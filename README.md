# Volatility-Trading
Predicting SPX implied volatility term structure using PCA and AR(1) on 15 years of options data. Backtests two strategies: VIX shorting and calendar spreads.

## Forecasting Implied Volatility
In this section, discuss the process of forecasting the IV

Concise: What is implied volalitilty and why do we use this to create a trading strategy.

Log-variance: $log(IV^2)$, will be forecasted rather than IV directly, as it ensures positivity and is more consistent with the log-normal assumptions underlying option pricing.

We forecast the term structure of ATM implied variance across 5 maturities: 14, 30, 60, 90, and 180 DTE. The key forecasting target is the slope of this term structure, defined as 

$$\log(\sigma^2_{14d}) - \log(\sigma^2_{180d})$$

A negative slope means the term structure is in contango (short-term IV < long-term IV), positive means backwardation.

### Data
Two data sources are used:
- SPX options data, every listed strike and expiry from 2010 to 2025. 2.3M rows in total, included with its bid, ask, IV, Greeks, volume, etc.
- SPX daily close prices from 2010 to 2025.

### Pre Processing
The risk-free rate $r$ and forward price $F$ are derived by regressing the put-call parity:

$$C - P = e^{-r\tau}(F - K)$$

Expand:

$$C - P = Fe^{-r\tau} - Ke^{-r\tau}$$

Written as a linear regression $y = a + bK$:

$$\underbrace{C - P}_{y} = \underbrace{Fe^{-r\tau}}_{a} + \underbrace{(-e^{-r\tau})}_{b} \cdot K$$

Log-moneyness is defined as $\log(K/F)$. It defines how far a strike is from the forward price on a log scale. We use this because option pricing is built on log-normal returns, and normalising by the forward $F$ rather than spot $S$ accounts for cost of carry.

IV mid is $(bid_{IV} + ask_{IV})/2$, and this is used as a clean estimate of the IV for each option.

The Black-76 model gives delta, gamma, vega, and theta for each option.

### Methodology
For each trading day, ATM implied variance is computed per expiry as the vega-weighted average:

$$\text{var}_{ATM} = \frac{\sum_i \text{var}_i \cdot \text{vega}_i}{\sum_i \text{vega}_i}$$

The nearest real expiry to each target maturity (14, 30, 60, 90, 180 DTE) 
is selected, giving a 5-point variance curve per day.

Then, principal compononent analysis is applied to the demeaned log-variance panel with 3 components:

$$v_t \approx \mu + Bf_t, \quad B \in \mathbb{R}^{5 \times 3}, \quad f_t \in \mathbb{R}^3$$

The three factors capture the **level** (PC1), **slope** (PC2), 
and **curvature** (PC3) of the term structure.

Each factor is forecast independently using a rolling AR(1) 
with a 2-year (504-day) window at a 5-day horizon:

$$f_{i,t+5} = \phi_i f_{i,t}$$

Mean-reversion is verified with the Augmented Dicker-Fuller tests and half-life analysis 
before applying AR(1). The forecast is projected back to 
log-variance space using the loadings $B$, from which the 
slope is reconstructed.

### Important Results
All three principal components are stationary (ADF p-value = 0.000), 
confirming AR(1) is appropriate. The 5-day ahead slope forecast achieves 
a correlation of **0.652** with the realized slope. Applying the PC1 regime 
filter improves the hit rate from 86.5% to **90.7%**, confirming that 
filtering out high-stress regimes meaningfully improves forecast reliability.




## VIX Strategy
Data:
- VX continuous futures.

## Calendar Spread Strategy
30, 60 day

options data filtered





## Team
team picture with names + degree
certificate
