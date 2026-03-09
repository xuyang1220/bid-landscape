# Piecewise Bid Landscape Estimation & Simulation 

This project builds a **synthetic auction environment and bid landscape estimator** similar to the components used in modern DSP / ads bidding systems.  
The goal is to understand how **censored auction observations → bid landscape estimation → bidding / pacing control** interact.

---

# 1. Problem Setup

In an ad auction, the hidden variable is the **market clearing price**

$$
Z = \text{market price}
$$

We place a bid ***b***.

Observation:

| Outcome | Information observed |
|------|------|
| Win | observe price \(z\) |
| Loss | only know \(z > b\) |

Thus auction logs are **right-censored observations**.

The key quantity to estimate is the **bid landscape**

$$
p(\text{win} \mid b, s)
=
P(Z \le b \mid s)
$$

where  

`s` = segment / cohort (publisher, device, geo, etc).

---

# 2. Synthetic Auction Market

To experiment with landscape estimation, we built a **synthetic market simulator**.

### Market design

Each segment has its own price distribution:

$$
Z_s \sim F_s(z)
$$

Implementation:

1. price axis discretized into bins
2. segment-specific **hazard functions**

$$
h_k(s) = P(Z \in bin_k \mid Z \ge bin_k, s)
$$

3. prices sampled from the resulting PMF

This creates realistic properties:

- heterogeneous segments
- skewed price distributions
- controllable market difficulty

---

# 3. Censored Auction Logs

Bids are generated from a synthetic value model.

Auction logs contain:

| field | meaning |
|------|------|
| `bid` | bid price |
| `segment_id` | market segment |
| `is_win` | auction outcome |
| `observed_price` | clearing price if win |
| `event_bin` | price bin if win |
| `bid_bin` | censoring bin if loss |

Losses only provide a **lower bound on price**.

---

# 4. Bid Landscape Estimation

We implemented a **piecewise survival model**.

### Hazard model

$$
h_k(s) = \sigma(\alpha_k + u_{s,k})
$$

where

- k = price bin
- s = segment
- $\sigma$ = sigmoid

### Survival function

$$
S_k(s) = P(Z > t_k)
=
\prod_{j\le k}(1 - h_j(s))
$$

### Win probability

$$
p(\text{win} \mid b, s)
=
1 - S(b)
$$

This guarantees:

- monotonic win curves
- valid probability bounds
- natural handling of censored losses

---

# 5. Training Objective

Training uses the **survival likelihood**.

### Win observation

If price falls in bin \(k\)

$$
\log L =
\sum_{j<k}\log(1-h_j) + \log h_k
$$

### Loss observation

If loss occurs at bid bin \(m\)

$$
\log L =
\sum_{j\le m}\log(1-h_j)
$$

Losses contribute **survival terms** only.

---

# 6. Expected Spend Estimation

Expected spend is approximated by

$$
E[\text{spend} \mid b,s]
=
\sum_{k: t_k \le b} P_k(s) \mu_k(s)
$$

where

- $P_k(s)$ = probability mass in bin
- $\mu_k(s)$ = mean price in bin

Two implementations tested:

### Bin midpoint

$$
\mu_k = \frac{left_k + right_k}{2}
$$

### Empirical bin mean

$$
\mu_k = E[Z \mid Z \in bin_k]
$$

In the current simulator both give similar results because prices are **uniformly sampled within bins**, making the midpoint equal to the mean.

---

# 7. Diagnostics

Several plots were implemented to evaluate the estimator.

### Win curves

Compare

- true win curve
- estimated win curve

Result: **very good recovery of win probabilities**

---

### Spend curves

Compare

- true expected spend
- estimated expected spend

Result:

- reasonable shape
- but underestimation at high bids due to
  - tail estimation difficulty
  - PMF error in high price bins

Spend is **more sensitive to tail errors** than win probability.

---

# 8. Kaplan–Meier Estimator

We also implemented a **Kaplan–Meier landscape estimator**.

Kaplan–Meier estimates the survival function:

$$
S(z) = P(Z > z)
$$

from censored data.

Formula:

$$
S(z)
=
\prod_{z_i \le z}
\left(1 - \frac{d_i}{n_i}\right)
$$

where

- $d_i$ = number of wins at price $z_i$
- $n_i$ = auctions still at risk

Then

$$
p(\text{win} \mid b) = 1 - S(b)
$$

Kaplan–Meier is essentially a **non-parametric hazard estimator**.

Our hazard model is a **smoothed / feature-aware version** of this idea.

---

# 9. Key Insights

### 1. Auction data is censored

Losses only reveal

$$
price > bid
$$

which requires survival modeling.

---

### 2. Win curves are easier than spend curves

Small tail errors barely affect

$$
p(\text{win})
$$

but strongly affect

$$
E[\text{spend}]
$$

---

### 3. Hazard models are natural for auctions

They provide:

- monotonic win curves
- smooth landscapes
- segment conditioning
- easy handling of censoring

---

# 10. System Architecture

The simulator now implements the full pipeline:

```
synthetic market
↓
auction logs (censored)
↓
bid landscape estimator
↓
win probability / expected spend
↓
bidding + pacing simulation
```

This mirrors the **core modeling loop used in real DSP bidding systems**.

---

# Next Steps

Potential improvements:

- better tail modeling for spend
- finer price bins
- segment regularization