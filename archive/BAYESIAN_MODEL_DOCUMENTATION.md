# Bayesian Model Documentation: Route Delay Analysis

## Overview

This document provides a theoretical explanation of the hierarchical Bayesian model used to analyze route delay uncertainty in `analysis1.ipynb`. The model implements **partial pooling** to estimate route-specific mean delays while accounting for varying data volumes across routes.

### Quick Summary

**Model Type**: Hierarchical Bayesian Model (Multilevel Model) with Partial Pooling

**Goal**: Estimate posterior distributions of mean delays for each route and quantify uncertainty

**Key Feature**: Variance scales with sample size (`σ²/n`), so observations with more trains have lower variance

**Main Result**: Uncertainty (credible interval width) decreases as route usage (number of trains) increases

---

## Problem Statement

We observe daily delay data for multiple routes, where:
- Each route has a different number of observations (trains)
- We want to estimate the **posterior distribution** of mean delay for each route
- We want to quantify **posterior uncertainty** (e.g., credible interval width) and compare it across routes with high vs. low data volume

---

## Model Structure

### Hierarchical Bayesian Model (Multilevel Model)

The model uses a **hierarchical structure** with three levels:

1. **Hyperparameters** (top level): Global parameters governing all routes
2. **Route-level parameters** (middle level): Route-specific mean delays
3. **Observations** (bottom level): Observed daily mean delays

### Visual Representation

```
Level 1 (Hyperparameters):
    μ₀ ~ Normal(0, 10²)          ──┐
    τ  ~ HalfNormal(5)            │ Global parameters
    σ  ~ HalfNormal(10)           ┘
              │
              │ (informs)
              ▼
Level 2 (Route Parameters):
    μ_route[1] ~ Normal(μ₀, τ²)  ──┐
    μ_route[2] ~ Normal(μ₀, τ²)     │ Route-specific means
    ...                             │ (R routes total)
    μ_route[R] ~ Normal(μ₀, τ²)  ──┘
              │
              │ (informs)
              ▼
Level 3 (Observations):
    y[1] ~ Normal(μ_route[route[1]], σ²/n[1])
    y[2] ~ Normal(μ_route[route[2]], σ²/n[2])
    ...
    y[N] ~ Normal(μ_route[route[N]], σ²/n[N])
```

**Key relationships:**
- All routes share the same hyperparameters `μ₀` and `τ` (partial pooling)
- Each route has its own mean `μ_route[r]`, but they're related through the common prior
- Observations have variance that scales inversely with sample size `n[i]`

### Mathematical Formulation

#### Level 1: Hyperpriors (Global Parameters)

```
μ₀ ~ Normal(0, 10²)          # Global mean delay
τ  ~ HalfNormal(5)           # Between-route standard deviation
σ  ~ HalfNormal(10)          # Observation-level standard deviation
```

**Interpretation:**
- **μ₀**: The overall average delay across all routes (centered at 0 with wide prior)
- **τ**: Controls how much route-specific means can vary from the global mean
- **σ**: The base standard deviation for individual observations

#### Level 2: Route-Specific Parameters

```
μ_route[r] ~ Normal(μ₀, τ²)  for r = 1, ..., R
```

**Interpretation:**
- Each route `r` has its own mean delay `μ_route[r]`
- These means are drawn from a common distribution centered at `μ₀` with spread `τ`
- This creates **partial pooling**: routes share information through the common prior

#### Level 3: Likelihood (Observations)

```
y_obs[i] ~ Normal(μ_route[route_idx[i]], σ²/n[i])
```

where:
- `y_obs[i]` is the observed mean delay for observation `i`
- `route_idx[i]` maps observation `i` to its route
- `n[i]` is the number of trains for observation `i`

**Key Feature: The variance scales with sample size**

The variance is `σ²/n[i]`, meaning:
- Observations with more trains (`n[i]` large) have **lower variance** (more precise)
- Observations with fewer trains (`n[i]` small) have **higher variance** (less precise)

This reflects the statistical principle that the standard error of a mean decreases as `1/√n`.

### Why σ²/n?

The model assumes that `y_obs[i]` represents a **mean delay** computed from `n[i]` trains. If individual train delays have variance `σ²`, then:

- The variance of the mean of `n` independent observations is `σ²/n`
- The standard error is `σ/√n`

This is a fundamental result from probability theory: if `X₁, ..., Xₙ ~ Normal(μ, σ²)` are independent, then:
```
X̄ = (X₁ + ... + Xₙ)/n ~ Normal(μ, σ²/n)
```

In the model, `y_obs[i]` is treated as this sample mean, so its variance is `σ²/n[i]`.

---

## Why Partial Pooling?

### Complete Pooling vs. No Pooling vs. Partial Pooling

1. **Complete Pooling**: All routes share the same mean
   - Ignores route-specific differences
   - Underfits the data

2. **No Pooling**: Each route estimated independently
   - Routes with few observations get extreme estimates
   - Overfits to noise in sparse data

3. **Partial Pooling** (this model): Routes share information but can differ
   - Routes with many observations: estimates dominated by their own data
   - Routes with few observations: estimates **shrunk** toward global mean `μ₀`
   - Balances between route-specific and global information

### Shrinkage Effect

For a route with few observations:
- The posterior mean `E[μ_route[r] | data]` is pulled toward `μ₀`
- This prevents extreme estimates driven by small sample noise
- The amount of shrinkage depends on:
  - Number of observations for that route
  - The between-route variance `τ²`
  - The observation variance `σ²`

---

## Full Joint Posterior Distribution

The complete Bayesian model specifies the joint posterior:

```
p(μ₀, τ, σ, μ_route | y, n, route_idx) ∝ p(y | μ_route, σ, n) × p(μ_route | μ₀, τ) × p(μ₀) × p(τ) × p(σ)
```

Where:
- **Likelihood**: `p(y | μ_route, σ, n) = ∏ᵢ Normal(y[i] | μ_route[route_idx[i]], σ²/n[i])`
- **Route-level prior**: `p(μ_route | μ₀, τ) = ∏ᵣ Normal(μ_route[r] | μ₀, τ²)`
- **Hyperpriors**: `p(μ₀) × p(τ) × p(σ)`

### Marginal Posterior for Route Means

For each route `r`, we obtain the marginal posterior:

```
p(μ_route[r] | y, n, route_idx) = ∫∫∫ p(μ₀, τ, σ, μ_route[r] | y, n, route_idx) dμ₀ dτ dσ
```

This marginal posterior:
- Combines information from route `r`'s own data
- Incorporates information from other routes through the hierarchical prior
- Accounts for uncertainty in hyperparameters `μ₀`, `τ`, and `σ`

### Analytical Insight (Conjugate Case)

In the simplified case where hyperparameters are fixed, the posterior for `μ_route[r]` is:

```
μ_route[r] | data ~ Normal(μ_post[r], σ_post²[r])
```

where:
- **Posterior mean**: `μ_post[r] = (n_total[r]/σ² × ȳ[r] + 1/τ² × μ₀) / (n_total[r]/σ² + 1/τ²)`
  - Weighted average of route-specific mean `ȳ[r]` and global mean `μ₀`
  - Weights depend on data precision (`n_total[r]/σ²`) vs. prior precision (`1/τ²`)
  
- **Posterior variance**: `σ_post²[r] = 1 / (n_total[r]/σ² + 1/τ²)`
  - Decreases as `n_total[r]` increases (more data → less uncertainty)
  - Increases as `τ²` increases (weaker pooling → more uncertainty)

---

## Posterior Uncertainty

### Quantification

The model estimates **posterior distributions** for each `μ_route[r]`, from which we compute:

- **90% Highest Density Interval (HDI)**: The narrowest interval containing 90% of posterior probability
- **Uncertainty = HDI width**: `HDI_upper - HDI_lower`

### Expected Behavior

1. **Frequent routes** (many trains):
   - Narrow credible intervals (low uncertainty)
   - Estimates driven primarily by route-specific data
   - Less shrinkage toward global mean

2. **Infrequent routes** (few trains):
   - Wide credible intervals (high uncertainty)
   - Estimates shrunk toward global mean
   - More uncertainty due to sparse data

### Mathematical Intuition

The posterior variance of `μ_route[r]` depends on:
- **Prior precision**: `1/τ²` (from hierarchical prior)
- **Data precision**: `n_total[r]/σ²` (from route's total observations)

Posterior precision = Prior precision + Data precision

Routes with more data have higher data precision, leading to:
- Tighter posterior distributions
- Narrower credible intervals
- Lower uncertainty

---

## Model Assumptions

1. **Normality**: Delays (or mean delays) are normally distributed
2. **Independence**: Observations are conditionally independent given route parameters
3. **Homoscedasticity**: Base variance `σ²` is constant across routes (though effective variance `σ²/n` varies)
4. **Hierarchical structure**: Route means come from a common distribution

---

## Prior Choices

### μ₀ ~ Normal(0, 10²)
- Centered at 0 (neutral prior)
- Standard deviation of 10 (wide, weakly informative)
- Allows delays to be positive or negative

### τ ~ HalfNormal(5)
- Constrained to be positive (standard deviation)
- Mode at 0, but allows moderate variation
- Controls strength of partial pooling

### σ ~ HalfNormal(10)
- Observation-level standard deviation
- Wide prior allows flexibility

---

## Computational Details

- **Sampler**: NUTS (No-U-Turn Sampler), a Hamiltonian Monte Carlo method
- **Chains**: 4 parallel chains
- **Iterations**: 2000 tuning + 2000 sampling per chain
- **Target acceptance rate**: 0.9 (high, for better mixing)

---

## Key Results

1. **Uncertainty decreases with route usage**: More trains → narrower credible intervals
2. **Partial pooling prevents overfitting**: Sparse routes are shrunk toward global mean
3. **Quantitative comparison**: Can compare uncertainty between frequent vs. infrequent routes

---

## References

- Gelman, A., et al. (2013). *Bayesian Data Analysis* (3rd ed.). Chapter 5: Hierarchical Models
- McElreath, R. (2020). *Statistical Rethinking* (2nd ed.). Chapter 13: Models With Memory

