# Bid Landscape + Auction Bidding Simulator

This project implements a simplified **ads auction and bidding simulator** to study how modern bidding strategies work, including:

- Bid landscape estimation
- Segment-level HDMI-style modeling
- Budget pacing
- tCPA bidding
- Max Conversions bidding
- Shadow price optimization
- Auction market shocks and mix shifts

The simulator reproduces many behaviors seen in real advertising systems.

---

# 1. Market Simulator

We simulate a **second-price auction market**.

Each auction has:

- feature vector `x`
- predicted conversion probability `p_conv`
- market price `M`

Market prices follow a **log-normal distribution**:

$
\log M \sim \mathcal{N}(\mu(x), \sigma^2)
$

where

$
\mu(x) = w^\top x + b
$

This produces realistic CPM price distributions.

---

# 2. Segment-Based Bid Landscape (HDMI Style)

To approximate the hidden market distribution we:

1. Generate a **segment id** from the feature space.

```python
score = X @ u
seg_id = quantile_bin(score)
```

2. Estimate per-segment landscape parameters:
$$
\log M \mid s \sim \mathcal{N}(\mu_s, \sigma_s^2)
$$

3. Compute auction statistics. Also known as bid landscape.
- Win probability
$$
P(\text{win} \mid b, s) = \Phi\!\left(\frac{\log b - \mu_s}{\sigma_s}\right)
$$
- Expected payment
$$
\mathbb{E}[\text{cost} \mid b, s] =
\exp\!\left(\mu_s + \tfrac{1}{2}\sigma_s^2\right)
\Phi\!\left(\frac{\log b - \mu_s - \sigma_s^2}{\sigma_s}\right)
$$

# 3. Auction Simulation
Given bid $b$ and market price $M$:
```
win = (b >= M)
pay = M if win else 0
```
Spend per step:
```
spend = Σ pay / 1000
```
Conversions are simulated with:
```
conv ~ Bernoulli(p_conv)
```
Conversion delay is also supported.

# 4. Budget Pacing

Budget is enforced using either:

- PID-style pacing multiplier
```
bid = base_bid * alpha_pace
```
where alpha_pace adjusts to match spend trajectory.

- Shadow price solver

We solve:
$$
\sum_i \mathbb{E}[\text{pay} \mid b_i] = B
$$

to find the budget shadow price, using Binary Search:
$$
b_i = \frac{V_i}{\lambda_{\text{budget}}}
$$

where $V_i$ is value per impression.

This produces very accurate expected spend.

# 5. tCPA Bidding

For target CPA campaigns:
$$
b_i = 1000 \cdot p_i \cdot \frac{\lambda_{cpa}}{\lambda_{budget}}
$$

Two control variables exist: ${\lambda_{cpa}}$ enforces budget, ${\lambda_{budget}}$ enforces tCPA efficiency.
```
λ_budget  → solved from expected spend
λ_cpa     → adjusted from observed CPA
```

# 6. Max Conversions Bidding
Objective:
$$
\max \sum_i p_i \, P(\text{win} \mid b_i)
$$
subject to
$$
\sum_i \mathbb{E}[\text{pay}_i] \le B
$$

Bid rule simplifies to:
$$
b_i = \frac{1000 \cdot p_i}{\lambda_{budget}}
$$
No explicit CPA constraint.

# 7. Shadow tCPA Design
A realistic Max Conversions implementation uses:

## Slow loop (every ~6 hours)

Solve shadow tCPA
$$
b_i = 1000 \cdot p_i \cdot tCPA_{\text{shadow}}
$$
so expected spend matches budget.

## Fast loop (minutes)

Apply pacing:
```
bid = base_bid * alpha_pace
```
Final bid:
```
bid = 1000 * p_conv * shadow_tcpa * alpha_pace
```

# 8. Segment Mix Shift Experiments

We tested robustness using market shocks:

## Traffic shock

Auction per minutes double.

## Price shock

Market prices double.

## Segment mix shift

Cheap inventory disappears and expensive segments dominate.

Segment-level landscapes correctly track spend under these shifts, while global landscapes fail.

# 9. Key Findings
## Segment-level HDMI landscapes are important

Global landscapes cannot handle supply mix shifts.

Segment conditioning fixes expected spend prediction.

## Expected spend ≠ realized spend

Differences arise from:

- model misspecification

- lognormal approximation error

- stochastic auctions

- bid clipping

- within-segment heterogeneity

Real systems often apply spend calibration factors. 

Shadow price bidding works well by solving for the budget dual variable: $\lambda_{budget}$, it allows the system to precisely control expected spend.

Max Conversions vs tCPA: Max Conversions typically yields:

- more conversions

- less predictable CPA

# 10. Future Extensions

Potential improvements:

- mixture bid landscapes

- reinforcement learning bidding

- multi-channel budget allocation

- auction competition models

# Summary

This simulator provides a minimal but realistic framework for studying:

- bid landscapes

- pacing controllers

- shadow price optimization

- tCPA and Max Conversions bidding

It captures many of the dynamics observed in real advertising bidding systems.