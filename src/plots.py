import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
IMAGES = ROOT / "images"
DATA = ROOT / "data"

def true_p_win_from_prices(
    prices: np.ndarray,
    bids: np.ndarray,
) -> np.ndarray:
    """
    Empirical true win curve from exact synthetic prices:
        p_win(b) = P(price <= b)
    """
    prices_sorted = np.sort(prices)
    return np.searchsorted(prices_sorted, bids, side="right") / len(prices_sorted)


def true_expected_spend_from_prices(
    prices: np.ndarray,
    bids: np.ndarray,
) -> np.ndarray:
    """
    Empirical true expected spend curve from exact synthetic prices:
        E[ price * 1{price <= b} ]
    """
    prices_sorted = np.sort(prices)
    csum = np.cumsum(prices_sorted)

    idx = np.searchsorted(prices_sorted, bids, side="right") - 1
    out = np.zeros_like(bids, dtype=float)

    valid = idx >= 0
    out[valid] = csum[idx[valid]] / len(prices_sorted)
    return out


def plot_true_vs_estimated_curves(
    model,
    market,
    segment_id: int,
    n_grid: int = 200,
) -> None:
    """
    Plot:
      1) true vs estimated win curve
      2) true vs estimated expected spend curve
    for a single segment.
    """
    seg_mask = market.segment_id == segment_id
    seg_prices = market.prices[seg_mask]

    if len(seg_prices) == 0:
        raise ValueError(f"No samples found for segment {segment_id}.")

    bmin = model.bin_edges[0]
    bmax = model.bin_edges[-1]
    bid_grid = np.exp(np.linspace(np.log(bmin), np.log(bmax), n_grid))

    # True curves from exact synthetic prices
    true_pwin = true_p_win_from_prices(seg_prices, bid_grid)
    true_esp = true_expected_spend_from_prices(seg_prices, bid_grid)

    # Estimated curves from learned landscape
    est_pwin = model.p_win(bid_grid, segment_id=segment_id)
    est_esp = model.expected_spend(bid_grid, segment_id=segment_id)

    # Plot win curve
    plt.figure(figsize=(8, 5))
    plt.semilogx(bid_grid, true_pwin, label="True win curve")
    plt.semilogx(bid_grid, est_pwin, label="Estimated win curve")
    plt.xlabel("Bid")
    plt.ylabel("P(win | bid, segment)")
    plt.title(f"True vs Estimated Win Curve (segment={segment_id})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(IMAGES / "win_curve.png", dpi=150)
    plt.show()

    # Plot expected spend curve
    plt.figure(figsize=(8, 5))
    plt.semilogx(bid_grid, true_esp, label="True expected spend")
    plt.semilogx(bid_grid, est_esp, label="Estimated expected spend")
    plt.xlabel("Bid")
    plt.ylabel("E[spend | bid, segment]")
    plt.title(f"True vs Estimated Spend Curve (segment={segment_id})")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    plt.savefig(IMAGES / "expected_spend.png", dpi=150)
    plt.show()


def plot_multiple_segments(
    model,
    market,
    segments: list[int] | np.ndarray,
    n_grid: int = 200,
) -> None:
    """
    Overlay true/estimated win curves for multiple segments,
    then overlay true/estimated spend curves for multiple segments.
    """
    bmin = model.bin_edges[0]
    bmax = model.bin_edges[-1]
    bid_grid = np.exp(np.linspace(np.log(bmin), np.log(bmax), n_grid))

    # Win curves
    plt.figure(figsize=(9, 6))
    for seg in segments:
        seg_mask = market.segment_id == seg
        seg_prices = market.prices[seg_mask]
        if len(seg_prices) == 0:
            continue

        true_pwin = true_p_win_from_prices(seg_prices, bid_grid)
        est_pwin = model.p_win(bid_grid, segment_id=seg)

        plt.semilogx(bid_grid, true_pwin, label=f"seg={seg} true")
        plt.semilogx(bid_grid, est_pwin, linestyle="--", label=f"seg={seg} est")

    plt.xlabel("Bid")
    plt.ylabel("P(win | bid, segment)")
    plt.title("True vs Estimated Win Curves")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    plt.savefig(IMAGES / "seg_win_curves.png", dpi=150)
    plt.show()

    # Spend curves
    plt.figure(figsize=(9, 6))
    for seg in segments:
        seg_mask = market.segment_id == seg
        seg_prices = market.prices[seg_mask]
        if len(seg_prices) == 0:
            continue

        true_esp = true_expected_spend_from_prices(seg_prices, bid_grid)
        est_esp = model.expected_spend(bid_grid, segment_id=seg)

        plt.semilogx(bid_grid, true_esp, label=f"seg={seg} true")
        plt.semilogx(bid_grid, est_esp, linestyle="--", label=f"seg={seg} est")

    plt.xlabel("Bid")
    plt.ylabel("E[spend | bid, segment]")
    plt.title("True vs Estimated Spend Curves")
    plt.grid(True, alpha=0.3)
    plt.legend(ncol=2, fontsize=9)
    plt.tight_layout()

    plt.savefig(IMAGES / "seg_spend_curves.png", dpi=150)
    plt.show()

def plot_spend_ratio(
    model,
    market,
    segment_id: int,
    n_grid: int = 200,
) -> None:
    seg_mask = market.segment_id == segment_id
    seg_prices = market.prices[seg_mask]

    bmin = model.bin_edges[0]
    bmax = model.bin_edges[-1]
    bid_grid = np.exp(np.linspace(np.log(bmin), np.log(bmax), n_grid))

    true_esp = true_expected_spend_from_prices(seg_prices, bid_grid)
    est_esp = model.expected_spend(bid_grid, segment_id=segment_id)

    ratio = est_esp / np.maximum(true_esp, 1e-8)

    plt.figure(figsize=(8, 5))
    plt.semilogx(bid_grid, ratio)
    plt.xlabel("Bid")
    plt.ylabel("Estimated / True expected spend")
    plt.title(f"Spend Curve Ratio (segment={segment_id})")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    plt.savefig(IMAGES / "spend_ratio2.png", dpi=150)
    plt.show()