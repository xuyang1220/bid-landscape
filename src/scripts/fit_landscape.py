import numpy as np
from src.piecewise_landscape import (make_log_price_bins, simulate_censored_auctions, sample_piecewise_market, 
sample_bids_from_value_model, PiecewiseSurvivalLandscape
)
from src.km_landscape import SegmentKaplanMeierLandscape
import src.plots as plots
# --------------------------------------
# Example usage
# --------------------------------------
def fit_piecewise() -> None:
    rng = np.random.default_rng(7)

    # 1) Define price bins
    bin_edges = make_log_price_bins(p_min=0.01, p_max=20.0, n_bins=100)

    # 2) Create synthetic market
    market = sample_piecewise_market(
        n_segments=8,
        bin_edges=bin_edges,
        n_auctions=500_000,
        random_state=7,
    )

    # 3) Simulate bids
    base_value_by_segment = np.exp(rng.normal(np.log(1.2), 0.5, size=8))
    bids = sample_bids_from_value_model(
        segment_id=market.segment_id,
        base_value_by_segment=base_value_by_segment,
        noise_scale=0.65,
        random_state=11,
    )

    # 4) Convert to censored auction logs
    data = simulate_censored_auctions(market, bids)

    # 5) Fit landscape estimator
    model = PiecewiseSurvivalLandscape(
        bin_edges=bin_edges,
        n_segments=8,
        l2_alpha=1e-4,
        l2_u=5e-4,
        random_state=3,
    )
    model.fit(data, lr=0.08, n_epochs=250, verbose=True)
    model.fit_bin_mean_price(data, min_count=20)
    #print(model.bin_mean_price)

    # 6) Compare recovered CDF on one segment
    for seg in range(8):
        summary = model.true_vs_estimated_summary(market.cdf_true, segment_id=seg)
        mae = summary["abs_err"].mean()
        print(f"\nsegment={seg}  CDF bin-wise MAE={mae:.4f}")

    # 7) Query p_win / expected_spend
    test_bids = np.array([0.05, 0.10, 0.25, 0.50, 1.0, 2.0, 5.0])
    pwin = model.p_win(test_bids, segment_id=seg)
    esp = model.expected_spend(test_bids, segment_id=seg)

    print("\nSample landscape:")
    for b, pw, es in zip(test_bids, pwin, esp):
        print(f"bid={b:>5.2f}  p_win={pw:>7.4f}  expected_spend={es:>7.4f}")
    
    # 8) Plots
    # Single segment diagnostic
    plots.plot_true_vs_estimated_curves(model, market, segment_id=0)

    # Multi-segment overlay
    plots.plot_multiple_segments(model, market, segments=[0, 1, 2, 3])

    # Estimated spend error ratio.
    plots.plot_spend_ratio(model, market, segment_id=0)


def fit_km() -> None:
    rng = np.random.default_rng(7)

    # 1) Define price bins
    bin_edges = make_log_price_bins(p_min=0.01, p_max=20.0, n_bins=20)

    # 2) Create synthetic market
    market = sample_piecewise_market(
        n_segments=8,
        bin_edges=bin_edges,
        n_auctions=50_000,
        random_state=7,
    )

    # 3) Simulate bids
    base_value_by_segment = np.exp(rng.normal(np.log(1.2), 0.5, size=8))
    bids = sample_bids_from_value_model(
        segment_id=market.segment_id,
        base_value_by_segment=base_value_by_segment,
        noise_scale=0.65,
        random_state=11,
    )

    # 4) Convert to censored auction logs
    data = simulate_censored_auctions(market, bids)

    # 5) Fit Kaplan Meier landscape estimator
    km_seg = SegmentKaplanMeierLandscape().fit(
        segment_id=data.segment_id,
        bid=data.bid,
        is_win=data.is_win,
        observed_price=data.observed_price,
    )

    test_bids = np.array([0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0])
    print(km_seg.p_win(test_bids, segment_id=0))

    # 6) Plot the curves.
    plots.plot_km_vs_true(km_seg, market, segment_id = 1)

if __name__ == "__main__":
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", type=str, default="km")
    args = ap.parse_args()
    if args.mode == "km":
        fit_km()
    else:
        fit_piecewise()