import numpy as np

def km_data_from_auctions(
    bid: np.ndarray,
    is_win: np.ndarray,
    observed_price: np.ndarray,
):
    """
    Convert auction logs into Kaplan-Meier inputs.

    For wins:
        observed_time = clearing price
        event_observed = True

    For losses:
        observed_time = bid
        event_observed = False
    """
    bid = np.asarray(bid, dtype=float)
    is_win = np.asarray(is_win, dtype=bool)
    observed_price = np.asarray(observed_price, dtype=float)

    if not (len(bid) == len(is_win) == len(observed_price)):
        raise ValueError("bid, is_win, observed_price must have same length.")

    observed_time = np.where(is_win, observed_price, bid)
    event_observed = is_win.copy()
    return observed_time, event_observed


class KaplanMeierLandscape:
    """
    Kaplan-Meier estimator for auction market-price survival.

    Input data format:
      - if auction is won: observe clearing price z
      - if auction is lost: only know z > bid, so censored at bid

    After fitting:
      survival(t) = P(price > t)
      p_win(b)    = P(price <= b) = 1 - survival(b)
    """

    def __init__(self):
        self.times_ = None          # sorted unique event/censor times
        self.survival_ = None       # survival after each time
        self.event_times_ = None    # sorted unique event times only
        self.event_survival_ = None

    def fit(self, observed_time: np.ndarray, event_observed: np.ndarray):
        """
        Parameters
        ----------
        observed_time : array-like, shape [N]
            - win  -> observed clearing price
            - loss -> censoring bid
        event_observed : bool array-like, shape [N]
            True for win/event, False for right-censored loss
        """
        t = np.asarray(observed_time, dtype=float)
        e = np.asarray(event_observed, dtype=bool)

        if t.ndim != 1 or e.ndim != 1 or len(t) != len(e):
            raise ValueError("observed_time and event_observed must be 1D arrays of same length.")

        order = np.argsort(t)
        t = t[order]
        e = e[order]

        unique_times = np.unique(t)

        survival_vals = []
        s = 1.0

        for time in unique_times:
            # at risk just before time: all with observed_time >= time
            n_at_risk = np.sum(t >= time)

            # events exactly at time
            d_events = np.sum((t == time) & e)

            # KM update only depends on events, censoring only affects risk set
            if n_at_risk > 0 and d_events > 0:
                s *= (1.0 - d_events / n_at_risk)

            survival_vals.append(s)

        self.times_ = unique_times
        self.survival_ = np.array(survival_vals, dtype=float)

        # Keep event-only grid too
        event_mask = np.array([np.any((t == time) & e) for time in unique_times])
        self.event_times_ = unique_times[event_mask]
        self.event_survival_ = self.survival_[event_mask]

        return self

    def survival(self, x):
        """
        Right-continuous step survival estimate S(x)=P(price>x)
        """
        x = np.asarray(x, dtype=float)
        out = np.ones_like(x, dtype=float)

        if self.times_ is None:
            raise RuntimeError("Call fit() first.")

        idx = np.searchsorted(self.times_, x, side="right") - 1
        mask = idx >= 0
        out[mask] = self.survival_[idx[mask]]
        return out

    def p_win(self, bid):
        """
        Estimated win probability P(price <= bid)
        """
        return 1.0 - self.survival(bid)


class SegmentKaplanMeierLandscape:
    def __init__(self):
        self.models_ = {}
        self.segments_ = None

    def fit(
        self,
        segment_id: np.ndarray,
        bid: np.ndarray,
        is_win: np.ndarray,
        observed_price: np.ndarray,
    ):
        segment_id = np.asarray(segment_id, dtype=int)
        bid = np.asarray(bid, dtype=float)
        is_win = np.asarray(is_win, dtype=bool)
        observed_price = np.asarray(observed_price, dtype=float)

        if not (len(segment_id) == len(bid) == len(is_win) == len(observed_price)):
            raise ValueError("All inputs must have same length.")

        self.segments_ = np.unique(segment_id)
        self.models_ = {}

        for s in self.segments_:
            mask = segment_id == s
            obs_t, evt = km_data_from_auctions(
                bid=bid[mask],
                is_win=is_win[mask],
                observed_price=observed_price[mask],
            )
            km = KaplanMeierLandscape().fit(obs_t, evt)
            self.models_[int(s)] = km

        return self

    def p_win(self, bid, segment_id):
        bid_arr = np.asarray(bid, dtype=float)
        seg_arr = np.asarray(segment_id, dtype=int)

        bid_flat = np.atleast_1d(bid_arr)
        seg_flat = np.atleast_1d(seg_arr)

        if seg_flat.size == 1 and bid_flat.size > 1:
            seg_flat = np.full_like(bid_flat, seg_flat.item(), dtype=int)

        if bid_flat.shape != seg_flat.shape:
            raise ValueError("bid and segment_id must have compatible shapes.")

        out = np.zeros_like(bid_flat, dtype=float)
        for i, (b, s) in enumerate(zip(bid_flat, seg_flat)):
            if int(s) not in self.models_:
                raise KeyError(f"Segment {s} not found.")
            out[i] = self.models_[int(s)].p_win(b)

        return out.reshape(np.shape(bid_arr))