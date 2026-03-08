# landscape.py
from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Optional, Tuple

import numpy as np


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.clip(x, -35.0, 35.0)
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray) -> np.ndarray:
    p = np.clip(p, 1e-8, 1.0 - 1e-8)
    return np.log(p / (1.0 - p))


def make_log_price_bins(
    p_min: float,
    p_max: float,
    n_bins: int,
) -> np.ndarray:
    if p_min <= 0 or p_max <= p_min:
        raise ValueError("Require 0 < p_min < p_max.")
    if n_bins < 2:
        raise ValueError("n_bins must be >= 2.")
    return np.exp(np.linspace(np.log(p_min), np.log(p_max), n_bins + 1))


def find_price_bin(values: np.ndarray, bin_edges: np.ndarray) -> np.ndarray:
    """
    Returns 0-based bin index in [0, K-1].
    Values <= first edge go to bin 0.
    Values >= last edge go to bin K-1.
    """
    k = len(bin_edges) - 1
    idx = np.searchsorted(bin_edges, values, side="right") - 1
    return np.clip(idx, 0, k - 1)


@dataclass
class CensoredAuctionDataset:
    segment_id: np.ndarray         # [N], int in [0, n_segments)
    bid: np.ndarray                # [N]
    is_win: np.ndarray             # [N], bool
    bid_bin: np.ndarray            # [N], 0-based
    event_bin: np.ndarray          # [N], 0-based, only used if is_win=True
    observed_price: np.ndarray     # [N], NaN for losses


@dataclass
class SyntheticMarket:
    bin_edges: np.ndarray          # [K+1]
    hazards_true: np.ndarray       # [S, K]
    pmf_true: np.ndarray           # [S, K]
    cdf_true: np.ndarray           # [S, K]
    prices: np.ndarray             # [N]
    segment_id: np.ndarray         # [N]


class PiecewiseSurvivalLandscape:
    """
    Segment-level piecewise-logistic survival model.

    Hazard for segment s and price bin k:
        h[s, k] = sigmoid(alpha[k] + u[s, k])

    Then:
        PMF[s, k] = S_prev[s, k] * h[s, k]
        CDF[s, k] = 1 - prod_{j<=k}(1 - h[s, j])

    Training objective:
      - Win with event_bin = k:
            -sum_{j<k} log(1-h_j) - log(h_k)
      - Loss with bid_bin = m:
            -sum_{j<=m} log(1-h_j)

    This handles right-censoring from lost auctions naturally.
    """

    def __init__(
        self,
        bin_edges: np.ndarray,
        n_segments: int,
        l2_alpha: float = 1e-4,
        l2_u: float = 1e-3,
        random_state: Optional[int] = None,
    ) -> None:
        self.bin_edges = np.asarray(bin_edges, dtype=float)
        if self.bin_edges.ndim != 1 or len(self.bin_edges) < 3:
            raise ValueError("bin_edges must be 1D with length >= 3.")
        self.k = len(self.bin_edges) - 1
        self.n_segments = int(n_segments)
        self.l2_alpha = float(l2_alpha)
        self.l2_u = float(l2_u)
        self.rng = np.random.default_rng(random_state)

        self.alpha = np.zeros(self.k, dtype=float)
        self.u = np.zeros((self.n_segments, self.k), dtype=float)

    # ----------------------------
    # Core probability utilities
    # ----------------------------
    def logits_for_segments(self, segment_ids: np.ndarray) -> np.ndarray:
        return self.alpha[None, :] + self.u[segment_ids]

    def hazards_for_segments(self, segment_ids: np.ndarray) -> np.ndarray:
        return sigmoid(self.logits_for_segments(segment_ids))

    @staticmethod
    def hazards_to_survival(h: np.ndarray) -> np.ndarray:
        """
        h: [B, K]
        returns S: [B, K+1]
        S[:, 0] = 1
        S[:, i+1] = P(Z > t_i)
        """
        bsz, k = h.shape
        s = np.ones((bsz, k + 1), dtype=h.dtype)
        for i in range(k):
            s[:, i + 1] = s[:, i] * (1.0 - h[:, i])
        return s

    @staticmethod
    def hazards_to_pmf(h: np.ndarray) -> np.ndarray:
        s = PiecewiseSurvivalLandscape.hazards_to_survival(h)
        return s[:, :-1] * h

    @staticmethod
    def hazards_to_cdf(h: np.ndarray) -> np.ndarray:
        s = PiecewiseSurvivalLandscape.hazards_to_survival(h)
        return 1.0 - s[:, 1:]

    def hazards_all(self) -> np.ndarray:
        segs = np.arange(self.n_segments)
        return self.hazards_for_segments(segs)

    def pmf_all(self) -> np.ndarray:
        return self.hazards_to_pmf(self.hazards_all())

    def cdf_all(self) -> np.ndarray:
        return self.hazards_to_cdf(self.hazards_all())

    # ----------------------------
    # Bid landscape API
    # ----------------------------
    def p_win(self, bid: np.ndarray | float, segment_id: np.ndarray | int) -> np.ndarray:
        """
        Piecewise-linear interpolation on the learned CDF.
        """
        bid_arr = np.asarray(bid, dtype=float)
        seg_arr = np.asarray(segment_id, dtype=int)

        bid_flat = np.atleast_1d(bid_arr)
        seg_flat = np.atleast_1d(seg_arr)
        if seg_flat.size == 1 and bid_flat.size > 1:
            seg_flat = np.full_like(bid_flat, seg_flat.item(), dtype=int)
        if bid_flat.shape != seg_flat.shape:
            raise ValueError("bid and segment_id must have compatible shapes.")

        cdf = self.cdf_all()[seg_flat]  # [B, K]
        edges = self.bin_edges
        idx = find_price_bin(bid_flat, edges)

        left_edge = edges[idx]
        right_edge = edges[idx + 1]
        frac = (bid_flat - left_edge) / np.maximum(right_edge - left_edge, 1e-12)
        frac = np.clip(frac, 0.0, 1.0)

        cdf_left = np.where(idx > 0, cdf[np.arange(len(idx)), idx - 1], 0.0)
        cdf_right = cdf[np.arange(len(idx)), idx]
        out = cdf_left + frac * (cdf_right - cdf_left)

        out = np.where(bid_flat <= edges[0], 0.0, out)
        out = np.where(bid_flat >= edges[-1], 1.0, out)
        return out.reshape(np.shape(bid_arr))

    def expected_spend(
        self,
        bid: np.ndarray | float,
        segment_id: np.ndarray | int,
        use_bin_midpoints: bool = True,
    ) -> np.ndarray:
        """
        Approximate E[price * 1{price <= bid} | segment].
        """
        bid_arr = np.asarray(bid, dtype=float)
        seg_arr = np.asarray(segment_id, dtype=int)

        bid_flat = np.atleast_1d(bid_arr)
        seg_flat = np.atleast_1d(seg_arr)
        if seg_flat.size == 1 and bid_flat.size > 1:
            seg_flat = np.full_like(bid_flat, seg_flat.item(), dtype=int)
        if bid_flat.shape != seg_flat.shape:
            raise ValueError("bid and segment_id must have compatible shapes.")

        pmf = self.pmf_all()[seg_flat]  # [B, K]
        centers = 0.5 * (self.bin_edges[:-1] + self.bin_edges[1:]) if use_bin_midpoints else self.bin_edges[1:]

        idx = find_price_bin(bid_flat, self.bin_edges)
        mask = np.arange(self.k)[None, :] < idx[:, None]
        out = (pmf * centers[None, :] * mask).sum(axis=1)

        # partial last bin
        left = self.bin_edges[idx]
        right = self.bin_edges[idx + 1]
        frac = (bid_flat - left) / np.maximum(right - left, 1e-12)
        frac = np.clip(frac, 0.0, 1.0)
        out += pmf[np.arange(len(idx)), idx] * centers[idx] * frac

        out = np.where(bid_flat <= self.bin_edges[0], 0.0, out)
        return out.reshape(np.shape(bid_arr))

    # ----------------------------
    # Training
    # ----------------------------
    def fit(
        self,
        data: CensoredAuctionDataset,
        lr: float = 0.05,
        n_epochs: int = 300,
        verbose: bool = True,
    ) -> Dict[str, list]:
        """
        Simple full-batch gradient descent.
        Compact and easy to read, suitable for simulator use.
        """
        s = data.segment_id.astype(int)
        is_win = data.is_win.astype(bool)
        event_bin = data.event_bin.astype(int)
        bid_bin = data.bid_bin.astype(int)

        history = {"loss": []}

        for epoch in range(n_epochs):
            logits = self.alpha[None, :] + self.u[s]     # [N, K]
            h = sigmoid(logits)

            # Grad wrt logits.
            # For each sample i:
            #   if win at k:
            #       dL/dz_j = h_j            for j < k
            #       dL/dz_k = h_k - 1
            #       dL/dz_j = 0              for j > k
            #   if loss censored at m:
            #       dL/dz_j = h_j            for j <= m
            #       dL/dz_j = 0              for j > m
            grad_logits = np.zeros_like(h)

            row_idx = np.arange(len(s))

            # losses
            for i in range(len(s)):
                if is_win[i]:
                    k = event_bin[i]
                    if k > 0:
                        grad_logits[i, :k] = h[i, :k]
                    grad_logits[i, k] = h[i, k] - 1.0
                else:
                    m = bid_bin[i]
                    grad_logits[i, : m + 1] = h[i, : m + 1]

            # Average gradients
            grad_alpha = grad_logits.mean(axis=0) + self.l2_alpha * self.alpha

            grad_u = np.zeros_like(self.u)
            np.add.at(grad_u, s, grad_logits)
            counts = np.bincount(s, minlength=self.n_segments).astype(float)
            counts = np.maximum(counts, 1.0)[:, None]
            grad_u = grad_u / len(s) + self.l2_u * self.u

            self.alpha -= lr * grad_alpha
            self.u -= lr * grad_u

            loss = self.nll(data)
            history["loss"].append(loss)

            if verbose and (epoch == 0 or (epoch + 1) % max(1, n_epochs // 10) == 0):
                print(f"epoch={epoch+1:4d}  loss={loss:.6f}")

        return history

    def nll(self, data: CensoredAuctionDataset) -> float:
        s = data.segment_id.astype(int)
        is_win = data.is_win.astype(bool)
        event_bin = data.event_bin.astype(int)
        bid_bin = data.bid_bin.astype(int)

        h = self.hazards_for_segments(s)
        eps = 1e-12
        logh = np.log(np.clip(h, eps, 1.0))
        log1mh = np.log(np.clip(1.0 - h, eps, 1.0))

        losses = np.zeros(len(s), dtype=float)
        for i in range(len(s)):
            if is_win[i]:
                k = event_bin[i]
                if k > 0:
                    losses[i] -= log1mh[i, :k].sum()
                losses[i] -= logh[i, k]
            else:
                m = bid_bin[i]
                losses[i] -= log1mh[i, : m + 1].sum()

        reg = 0.5 * self.l2_alpha * np.sum(self.alpha ** 2) + 0.5 * self.l2_u * np.sum(self.u ** 2)
        return float(losses.mean() + reg)

    # ----------------------------
    # Diagnostics
    # ----------------------------
    def true_vs_estimated_summary(
        self,
        true_cdf: np.ndarray,
        segment_id: int,
    ) -> Dict[str, np.ndarray]:
        est_cdf = self.cdf_all()[segment_id]
        return {
            "bin_right_edges": self.bin_edges[1:].copy(),
            "true_cdf": true_cdf[segment_id].copy(),
            "est_cdf": est_cdf.copy(),
            "abs_err": np.abs(true_cdf[segment_id] - est_cdf),
        }


# --------------------------------------
# Synthetic market generation utilities
# --------------------------------------
def sample_piecewise_market(
    n_segments: int,
    bin_edges: np.ndarray,
    n_auctions: int,
    segment_probs: Optional[np.ndarray] = None,
    random_state: Optional[int] = None,
) -> SyntheticMarket:
    """
    Create a synthetic market with segment-specific piecewise hazards,
    then sample exact prices by first sampling a bin, then uniformly
    within that bin.
    """
    rng = np.random.default_rng(random_state)
    k = len(bin_edges) - 1

    if segment_probs is None:
        segment_probs = np.ones(n_segments) / n_segments
    segment_probs = np.asarray(segment_probs, dtype=float)
    segment_probs = segment_probs / segment_probs.sum()

    # Global base hazards rising with price bin, but not too close to 1
    base = np.linspace(-2.2, -0.2, k)

    # Segment offsets: some easier markets, some harder
    seg_shift = rng.normal(0.0, 0.6, size=(n_segments, 1))
    bin_wiggle = rng.normal(0.0, 0.35, size=(n_segments, k))
    logits = base[None, :] + seg_shift + 0.5 * bin_wiggle
    hazards = sigmoid(logits)

    model = PiecewiseSurvivalLandscape(bin_edges=bin_edges, n_segments=n_segments)
    pmf = model.hazards_to_pmf(hazards)
    cdf = model.hazards_to_cdf(hazards)

    seg = rng.choice(n_segments, size=n_auctions, p=segment_probs)

    prices = np.empty(n_auctions, dtype=float)
    for s_idx in range(n_segments):
        mask = seg == s_idx
        n = int(mask.sum())
        if n == 0:
            continue
        event_bins = rng.choice(k, size=n, p=pmf[s_idx] / pmf[s_idx].sum())
        left = bin_edges[event_bins]
        right = bin_edges[event_bins + 1]
        prices[mask] = rng.uniform(left, right)

    return SyntheticMarket(
        bin_edges=bin_edges,
        hazards_true=hazards,
        pmf_true=pmf,
        cdf_true=cdf,
        prices=prices,
        segment_id=seg,
    )


def simulate_censored_auctions(
    market: SyntheticMarket,
    bids: np.ndarray,
) -> CensoredAuctionDataset:
    """
    Turn exact prices into censored win/loss observations.

    Win if bid >= price.
    If win, we keep observed_price and event_bin(price).
    If loss, only know price > bid, so event_bin is unused.
    """
    if len(bids) != len(market.prices):
        raise ValueError("bids and market.prices must have the same length.")

    bid = np.asarray(bids, dtype=float)
    price = market.prices
    is_win = bid >= price

    bid_bin = find_price_bin(bid, market.bin_edges)
    event_bin = find_price_bin(price, market.bin_edges)

    observed_price = np.where(is_win, price, np.nan)

    return CensoredAuctionDataset(
        segment_id=market.segment_id.copy(),
        bid=bid,
        is_win=is_win,
        bid_bin=bid_bin,
        event_bin=event_bin,
        observed_price=observed_price,
    )


def sample_bids_from_value_model(
    segment_id: np.ndarray,
    base_value_by_segment: np.ndarray,
    noise_scale: float = 0.25,
    random_state: Optional[int] = None,
) -> np.ndarray:
    """
    Simple synthetic bid model for experiments:
        bid = positive noisy value proxy by segment
    """
    rng = np.random.default_rng(random_state)
    base = base_value_by_segment[segment_id]
    log_bid = np.log(np.maximum(base, 1e-4)) + rng.normal(0.0, noise_scale, size=len(segment_id))
    return np.exp(log_bid)