import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import statsmodels.api as sm
from dataclasses import dataclass
from sklearn.decomposition import PCA
from statsmodels.graphics.tsaplots import plot_acf
from statsmodels.tsa.stattools import adfuller


# ─────────────────────────────────────────────
# CONFIGURATION
# ─────────────────────────────────────────────

CSV_PATH    = r"df_fwd_full.csv"
VX_PATH     = r"VX_full_1day_continuous_absolute_adjusted.csv"

TARGET_DTES = [14, 30, 60, 90, 180]
MAX_DTE     = 220
TICKER      = "SPX"

H           = 5       # forecast horizon (days)
WINDOW      = 504     # rolling training window (~2 trading years)
N_FACTORS   = 3       # PCA components retained
# PC3 is excluded from AR(1) forecasting: its half-life of 1.6 days makes it
# indistinguishable from noise at a 5-day horizon.
USE_FACTORS = [0, 1]

SIG_WIN     = 60      # rolling window for signal z-score normalisation
PC1_WIN     = 252     # rolling window for PC1 regime filter (1 trading year)
# SIG_Z_TH filters out low-conviction forecasts: only signals whose absolute
# z-score exceeds 0.5 standard deviations are considered actionable.
SIG_Z_TH    = 0.5
# PC1_Z_TH excludes high-stress regimes: when PC1 is more than 1 std above
# its 1-year rolling mean, the term structure is in an abnormal state and
# model predictions become less reliable.
PC1_Z_TH    = 1.0
REBAL_STEP  = 5       # rebalancing frequency (days)

# Calendar spread parameters
AGGRESSIVENESS   = 5     # RuleConfig master knob: 1 (conservative) to 10 (aggressive)
INITIAL_CASH     = 100_000
R_DEFAULT        = 0.04  # risk-free rate used in Black-Scholes Greeks
PBO_SPLITS       = 10    # number of data splits for PBO validation
PBO_TRIALS       = 150   # number of random IS/OOS trials for PBO


# ─────────────────────────────────────────────
# FUNCTIONS
# ─────────────────────────────────────────────

def load_data(path: str) -> pd.DataFrame:
    """
    Load the SPX options dataset, retain only the columns required for the
    term structure and filter out rows with missing or non-positive values.
    Implied variance (iv_mid^2) is pre-computed here for efficiency.
    """
    df = pd.read_csv(path)
    df = df[df["ticker"] == TICKER].copy()

    df["trade_date"] = pd.to_datetime(df["trade_date"])
    df["expiration"]  = pd.to_datetime(df["expiration"])

    needed = ["expiration", "dte", "iv_mid", "vega", "log_moneyness_fwd"]
    df = df.dropna(subset=needed)
    df = df[(df["iv_mid"] > 0) & (df["vega"] > 0) & (df["dte"] > 0)]

    df["variance"] = df["iv_mid"] ** 2
    return df


def build_term_structure(df: pd.DataFrame) -> pd.DataFrame:
    """
    Build a daily panel of ATM implied variance across five fixed maturities.

    ATM is defined as the strike with the smallest absolute log forward-moneyness
    (log(K/F)), which is the theoretically correct ATM definition under
    Black-76 / forward-based pricing.

    For each expiration, the ATM variance is aggregated using a vega-weighted
    average. Vega weighting ensures that options with higher price sensitivity
    to volatility contribute more to the estimate, reducing the impact of
    illiquid or wide-spread contracts.

    Each target DTE (14, 30, 60, 90, 180) is matched to the nearest available
    expiration to avoid interpolation artefacts.
    """
    records = []

    for trade_date, g_day in df.groupby("trade_date"):
        exp_points = []

        for exp, g_exp in g_day.groupby("expiration"):
            m_abs    = np.abs(g_exp["log_moneyness_fwd"])
            atm_mask = m_abs == m_abs.min()
            atm      = g_exp[atm_mask]

            var_atm  = np.sum(atm["variance"] * atm["vega"]) / np.sum(atm["vega"])
            dte_rep  = float(np.median(g_exp["dte"]))
            exp_points.append((exp, dte_rep, var_atm))

        if not exp_points:
            continue

        exp_df = pd.DataFrame(exp_points, columns=["expiration", "dte_rep", "variance_atm"])

        for target in TARGET_DTES:
            idx = (exp_df["dte_rep"] - target).abs().idxmin()
            row = exp_df.loc[idx]
            records.append({
                "trade_date":        trade_date,
                "target_dte":        target,
                "actual_expiration": row["expiration"],
                "actual_dte":        row["dte_rep"],
                "variance":          row["variance_atm"],
            })

    ts_long  = pd.DataFrame(records)
    ts_panel = (
        ts_long
        .pivot(index="trade_date", columns="target_dte", values="variance")
        .sort_index()
    )
    ts_panel.columns = [f"var_{c}d" for c in ts_panel.columns]
    return ts_panel


def fit_pca(ts_panel: pd.DataFrame, n_components: int = N_FACTORS):
    """
    Fit a PCA on the log-variance term structure.

    Log-variance is preferred over raw variance because it is approximately
    normally distributed, stabilises heteroskedasticity across maturities,
    and ensures the reconstructed surface remains strictly positive.
    The data is mean-centred before decomposition, as required by PCA.
    """
    X      = np.log(ts_panel).dropna()
    X_mean = X.mean()
    Xc     = X - X_mean
    pca    = PCA(n_components=n_components).fit(Xc)
    F      = pca.transform(Xc)
    return X, X_mean, pca, F


def rolling_ar1_backtest(X: pd.DataFrame, window: int, h: int, n_factors: int, use_factors: list):
    """
    Strict walk-forward backtest: at each step t, PCA and AR(1) parameters
    are estimated on the preceding [t-window, t) block only. No future
    information is used at any point, eliminating look-ahead bias.

    For each selected factor, an AR(1) is iterated h times to produce an
    h-step-ahead forecast. h_eval = h-1 selects the prediction at exactly
    the target horizon (index 0 = 1-step, index h-1 = h-step).

    Factors not in use_factors (here PC3) are left at zero, meaning they
    are implicitly forecast as mean-reverting to zero within the horizon.
    The reconstructed term structure forecast is obtained by projecting the
    factor forecasts back through the current loading matrix.
    """
    dates  = X.index
    cols   = X.columns
    T      = len(X)
    h_eval = h - 1  # index of the h-step-ahead observation within the test window

    F_true_list, F_hat_list = [], []
    X_true_list, X_hat_list = [], []

    for t in range(window, T - h):
        train = X.iloc[t - window : t]
        test  = X.iloc[t : t + h]

        mu  = train.mean()
        pca = PCA(n_components=n_factors).fit(train - mu)

        F_train = pca.transform(train - mu)
        F_test  = pca.transform(test  - mu)

        F_fore = np.zeros((h, n_factors))
        for k in use_factors:
            y   = F_train[1:, k]
            x   = sm.add_constant(F_train[:-1, k])
            res = sm.OLS(y, x).fit()
            a, b = res.params

            x0 = F_train[-1, k]
            for step in range(h):
                x0 = a + b * x0
                F_fore[step, k] = x0

        X_hat = (F_fore @ pca.components_) + mu.values

        F_true_list.append(F_test[h_eval])
        F_hat_list.append(F_fore[h_eval])
        X_true_list.append(test.iloc[h_eval].values)
        X_hat_list.append(X_hat[h_eval])

    bt_dates    = dates[window + h_eval : window + h_eval + len(F_true_list)]
    factor_cols = [f"PC{i+1}" for i in range(n_factors)]

    F_true = pd.DataFrame(F_true_list, index=bt_dates, columns=factor_cols)
    F_hat  = pd.DataFrame(F_hat_list,  index=bt_dates, columns=[c + "_hat" for c in factor_cols])
    X_true = pd.DataFrame(X_true_list, index=bt_dates, columns=cols)
    X_hat  = pd.DataFrame(X_hat_list,  index=bt_dates, columns=[c + "_hat" for c in cols])

    return F_true, F_hat, X_true, X_hat


def half_life(series: np.ndarray) -> float:
    y   = series[1:]
    x   = sm.add_constant(series[:-1])
    phi = sm.OLS(y, x).fit().params[1]
    return -np.log(2) / np.log(phi)


def rmse(a, b) -> float:
    return np.sqrt(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def hit_rate(pred: pd.Series, true: pd.Series) -> float:
    idx = pred != 0
    return (np.sign(pred[idx]) == np.sign(true[idx])).mean()


def masked_corr(pred: pd.Series, true: pd.Series) -> float:
    idx = pred != 0
    return np.corrcoef(pred[idx], true[idx])[0, 1]


def load_vx(path: str, start_date: str = "2010-01-01") -> pd.Series:
    """
    Load the VX continuous front-month future (close price, absolute adjusted).
    P&L is expressed in VIX points, not percentage returns, consistent with
    how VIX futures are traded and margined.
    """
    df = pd.read_csv(path)
    df["Date"]  = pd.to_datetime(df["Date"], errors="coerce")
    df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
    df = df.dropna(subset=["Date", "Close"]).sort_values("Date")
    df = df[df["Date"] >= pd.to_datetime(start_date)]
    return df.set_index("Date")["Close"]


def sharpe(x: pd.Series) -> float:
    return np.sqrt(252) * x.mean() / x.std() if x.std() != 0 else np.nan


def max_drawdown(cum: pd.Series) -> float:
    return (cum - cum.cummax()).min()


# ── Calendar spread: trading rule
# ─────────────────────────────────────────────

@dataclass
class RuleConfig:
    """
    All tunable parameters for the calendar spread signal in one place.

    aggressiveness (1–10): master knob that scales entry thresholds and
    position sizing simultaneously. Higher values lower the minimum forecast
    required to enter and widen the PC1 regime window, resulting in more
    active days and larger average contract sizes.
    max_short is capped below max_long because positive-slope forecasts
    (backwardation regime) are historically rarer and less reliable than
    negative-slope ones.
    """
    aggressiveness: int = AGGRESSIVENESS
    max_long:       int = 3
    max_short:      int = 2


def generate_signal(row: pd.Series, config: RuleConfig) -> dict:
    """
    Core calendar spread trading rule.

    LONG calendar  (sell 30d, buy 60d): entered when slope_forecast < 0
      (term structure in contango) and PC1 confirms a calm or moderate regime.
    SHORT calendar (buy 30d, sell 60d): entered when slope_forecast > 0 and
      PC1 confirms a stressed regime (backwardation is genuine, not noise).
    FLAT: when conviction falls below threshold or the regime is ambiguous.

    Position size (0 to max_long / max_short contracts) scales with the
    absolute magnitude of slope_forecast and is penalised by one level when
    PC1 exceeds the aggressiveness-dependent stress threshold.
    All thresholds are interpolated linearly between the conservative
    (aggressiveness=1) and aggressive (aggressiveness=10) anchors.
    """
    forecast = row["slope_forecast"]
    pc1      = row["pc1"]
    agg      = config.aggressiveness

    min_forecast       = np.interp(agg, [1, 10], [0.40, 0.05])
    pc1_max_long       = np.interp(agg, [1, 10], [0.5,  2.5])
    pc1_min_short      = np.interp(agg, [1, 10], [2.0,  0.8])
    min_forecast_short = np.interp(agg, [1, 10], [0.25, 0.05])

    size_thresholds = [
        np.interp(agg, [1, 10], [0.30, 0.10]),
        np.interp(agg, [1, 10], [0.50, 0.30]),
        np.interp(agg, [1, 10], [0.70, 0.50]),
    ]

    pc1_stress_penalty_threshold = np.interp(agg, [1, 10], [0.3, 1.5])

    abs_forecast = abs(forecast)
    long_size    = 0
    short_size   = 0

    if forecast < 0 and abs_forecast >= min_forecast and pc1 < pc1_max_long:
        base_size = sum(abs_forecast >= t for t in size_thresholds)
        if pc1 > pc1_stress_penalty_threshold:
            base_size = max(base_size - 1, 0)
        long_size = min(base_size, config.max_long)

    elif forecast > 0 and abs_forecast >= min_forecast_short and pc1 > pc1_min_short:
        base_size  = sum(abs_forecast >= t for t in size_thresholds)
        short_size = min(base_size, config.max_short)

    return {"long_calendars": long_size, "short_calendars": short_size}


def run_signals(df: pd.DataFrame, config: RuleConfig = None) -> pd.DataFrame:
    """Apply trading rule row-by-row; returns df augmented with position columns."""
    if config is None:
        config = RuleConfig()
    signals = df.apply(lambda row: generate_signal(row, config), axis=1, result_type="expand")
    result  = df.copy()
    result["long_calendars"]  = signals["long_calendars"]
    result["short_calendars"] = signals["short_calendars"]
    result["net_position"]    = result["long_calendars"] - result["short_calendars"]
    return result


def signal_diagnostics(df: pd.DataFrame, config: RuleConfig = None):
    """Print summary statistics for a signal run."""
    if config is None:
        config = RuleConfig()
    result     = run_signals(df, config)
    total      = len(result)
    active     = (result["net_position"] != 0).sum()
    long_days  = (result["long_calendars"] > 0).sum()
    short_days = (result["short_calendars"] > 0).sum()

    print(f"{'='*55}")
    print(f"  Calendar Spread Rule — Aggressiveness = {config.aggressiveness}")
    print(f"{'='*55}")
    print(f"  Period:      {result['date'].min().date()} to {result['date'].max().date()}")
    print(f"  Total days:  {total}")
    print(f"  Active days: {active} ({active/total*100:.1f}%)")
    print(f"    Long:      {long_days} ({long_days/total*100:.1f}%)")
    print(f"    Short:     {short_days} ({short_days/total*100:.1f}%)")
    print(f"    Flat:      {total - active} ({(total-active)/total*100:.1f}%)")
    print()

    for side, col, mx in [("Long",  "long_calendars",  config.max_long),
                           ("Short", "short_calendars", config.max_short)]:
        active_side = result[result[col] > 0]
        if len(active_side) > 0:
            print(f"  {side} size distribution:")
            for s in range(1, mx + 1):
                n = (active_side[col] == s).sum()
                print(f"    Size {s}: {n} days ({n/len(active_side)*100:.0f}%)")
            print(f"    Avg size: {active_side[col].mean():.2f}")
        print()

    return result


# ── Calendar spread: simulation engine
# ─────────────────────────────────────────────

class CalendarSpreadSimulator:
    """Tracks cash, positions, daily P&L and trade history for the simulation."""

    def __init__(self, initial_cash: float = INITIAL_CASH):
        self.long_calendars        = 0
        self.short_calendars       = 0
        self.cash                  = initial_cash
        self.initial_cash          = initial_cash
        self.daily_pnl             = []
        self.daily_portfolio_value = []
        self.transaction_costs     = 0
        self.trades                = []

    def get_portfolio_value(self, long_cal_value: float, short_cal_value: float) -> float:
        position_value = (self.long_calendars  * long_cal_value
                        - self.short_calendars * short_cal_value)
        return self.cash + position_value

    def get_current_positions(self) -> dict:
        return {"long_calendars":  self.long_calendars,
                "short_calendars": self.short_calendars,
                "cash":            self.cash}


class EnhancedSimulator(CalendarSpreadSimulator):
    """Extends CalendarSpreadSimulator with daily Greeks tracking."""

    def __init__(self, initial_cash: float = INITIAL_CASH):
        super().__init__(initial_cash)
        self.daily_vega  = []
        self.daily_theta = []
        self.daily_gamma = []
        self.daily_delta = []


def execute_trades(simulator: CalendarSpreadSimulator, row: pd.Series,
                   use_costs: bool = True) -> float:
    """
    Reconcile current simulator positions to the signal target.

    Each calendar spread is decomposed into its two legs. When use_costs=True,
    buys cross the spread (pay ask, receive bid); slippage is measured as the
    deviation of the execution price from mid. When use_costs=False, all trades
    execute at mid (useful for cost-free baseline comparisons).
    """
    target_long  = row["signal"]["long_calendars"]
    target_short = row["signal"]["short_calendars"]
    long_trade   = target_long  - simulator.long_calendars
    short_trade  = target_short - simulator.short_calendars

    if use_costs:
        long_cal_buy_price   = row["option_60dte_ask"] - row["option_30dte_bid"]
        long_cal_sell_price  = row["option_60dte_bid"] - row["option_30dte_ask"]
        short_cal_buy_price  = row["option_30dte_ask"] - row["option_60dte_bid"]
        short_cal_sell_price = row["option_30dte_bid"] - row["option_60dte_ask"]
    else:
        long_cal_mid  = row["option_60dte_mid"] - row["option_30dte_mid"]
        short_cal_mid = row["option_30dte_mid"] - row["option_60dte_mid"]
        long_cal_buy_price = long_cal_sell_price   = long_cal_mid
        short_cal_buy_price = short_cal_sell_price = short_cal_mid

    long_mid   = row["option_60dte_mid"] - row["option_30dte_mid"]
    short_mid  = row["option_30dte_mid"] - row["option_60dte_mid"]
    total_cost = 0.0

    if long_trade > 0:
        simulator.cash -= long_trade * long_cal_buy_price
        simulator.long_calendars += long_trade
        total_cost += abs(long_cal_buy_price - long_mid) * long_trade
        simulator.trades.append({"date": row["date"], "type": "BUY_LONG_CAL",
                                  "quantity": long_trade, "price": long_cal_buy_price})
    elif long_trade < 0:
        simulator.cash += abs(long_trade) * long_cal_sell_price
        simulator.long_calendars += long_trade
        total_cost += abs(long_cal_sell_price - long_mid) * abs(long_trade)
        simulator.trades.append({"date": row["date"], "type": "SELL_LONG_CAL",
                                  "quantity": abs(long_trade), "price": long_cal_sell_price})

    if short_trade > 0:
        simulator.cash += short_trade * short_cal_sell_price
        simulator.short_calendars += short_trade
        total_cost += abs(short_cal_sell_price - short_mid) * short_trade
        simulator.trades.append({"date": row["date"], "type": "OPEN_SHORT_CAL",
                                  "quantity": short_trade, "price": short_cal_sell_price})
    elif short_trade < 0:
        simulator.cash -= abs(short_trade) * short_cal_buy_price
        simulator.short_calendars += short_trade
        total_cost += abs(short_cal_buy_price - short_mid) * abs(short_trade)
        simulator.trades.append({"date": row["date"], "type": "CLOSE_SHORT_CAL",
                                  "quantity": abs(short_trade), "price": short_cal_buy_price})

    simulator.transaction_costs += total_cost
    return total_cost


def handle_expiration(simulator: CalendarSpreadSimulator,
                      current_day_idx: int, df_data: pd.DataFrame) -> float:
    """
    Settle positions every 30 trading days to approximate real option expiration.

    Long calendar: the short 30d leg settles at intrinsic value; the 60d leg
    is closed at mid. Short calendar: the long 30d leg settles at intrinsic;
    the short 60d leg closes at mid. Cash flows are credited to the simulator
    and all open contracts are zeroed out after settlement.
    """
    if (current_day_idx + 1) % 30 != 0:
        return 0.0

    row             = df_data.iloc[current_day_idx]
    intrinsic_30    = max(row["spx_price"] - row["strike"], 0)
    expiration_cost = 0.0

    if simulator.long_calendars > 0:
        cf = (-simulator.long_calendars * intrinsic_30
              + simulator.long_calendars * row["option_60dte_mid"])
        simulator.cash += cf
        expiration_cost += abs(simulator.long_calendars * intrinsic_30)
        simulator.trades.append({"date": row["date"], "type": "EXPIRATION_LONG_CAL",
                                  "quantity": simulator.long_calendars,
                                  "total_cash_flow": cf})
        simulator.long_calendars = 0

    if simulator.short_calendars > 0:
        cf = (simulator.short_calendars * intrinsic_30
              - simulator.short_calendars * row["option_60dte_mid"])
        simulator.cash += cf
        expiration_cost += abs(simulator.short_calendars * row["option_60dte_mid"])
        simulator.trades.append({"date": row["date"], "type": "EXPIRATION_SHORT_CAL",
                                  "quantity": simulator.short_calendars,
                                  "total_cash_flow": cf})
        simulator.short_calendars = 0

    return expiration_cost


def calculate_performance_metrics(simulator: CalendarSpreadSimulator) -> dict:
    """Compute standard performance metrics from simulator state."""
    pv       = np.array(simulator.daily_portfolio_value)
    ret_pct  = (pv[-1] / simulator.initial_cash - 1) * 100
    d_ret    = np.diff(pv) / pv[:-1]
    sr       = (d_ret.mean() / d_ret.std() * np.sqrt(252)
                if d_ret.std() > 0 else 0)
    run_max  = np.maximum.accumulate(pv)
    max_dd   = np.min((pv - run_max) / run_max * 100)
    win_rate = sum(p > 0 for p in simulator.daily_pnl) / len(simulator.daily_pnl) * 100

    return {
        "total_return_dollar":     pv[-1] - simulator.initial_cash,
        "total_return_pct":        ret_pct,
        "sharpe_ratio":            sr,
        "max_drawdown_pct":        max_dd,
        "num_trades":              len(simulator.trades),
        "win_rate_pct":            win_rate,
        "avg_daily_pnl":           np.mean(simulator.daily_pnl),
        "total_transaction_costs": simulator.transaction_costs,
        "final_cash":              simulator.cash,
        "final_portfolio_value":   pv[-1],
    }


def print_performance_report(metrics: dict):
    """Pretty-print performance metrics."""
    print("\n" + "="*60)
    print("PERFORMANCE METRICS")
    print("="*60)
    print(f"\nReturns:")
    print(f"  Total Return:           ${metrics['total_return_dollar']:,.2f}")
    print(f"  Total Return (%):       {metrics['total_return_pct']:.2f}%")
    print(f"  Avg Daily P&L:          ${metrics['avg_daily_pnl']:,.2f}")
    print(f"\nRisk Metrics:")
    print(f"  Sharpe Ratio:           {metrics['sharpe_ratio']:.3f}")
    print(f"  Max Drawdown:           {metrics['max_drawdown_pct']:.2f}%")
    print(f"\nTrading Activity:")
    print(f"  Total Trades:           {metrics['num_trades']}")
    print(f"  Win Rate:               {metrics['win_rate_pct']:.2f}%")
    print(f"  Transaction Costs:      ${metrics['total_transaction_costs']:,.2f}")
    print(f"\nFinal State:")
    print(f"  Cash:                   ${metrics['final_cash']:,.2f}")
    print(f"  Portfolio Value:        ${metrics['final_portfolio_value']:,.2f}")
    print("="*60)


def plot_clean_performance(simulator: CalendarSpreadSimulator):
    """Four-panel performance dashboard: cumulative return, drawdown, monthly P&L, summary."""
    pv       = np.array(simulator.daily_portfolio_value)
    returns  = (pv / simulator.initial_cash - 1) * 100
    run_max  = np.maximum.accumulate(pv)
    drawdown = (pv - run_max) / run_max * 100

    d_ret    = np.diff(pv) / pv[:-1]
    sr       = d_ret.mean() / d_ret.std() * np.sqrt(252) if d_ret.std() > 0 else 0
    win_rate = sum(p > 0 for p in simulator.daily_pnl) / len(simulator.daily_pnl) * 100

    days_per_month  = 21
    num_months      = len(simulator.daily_pnl) // days_per_month
    monthly_returns = [sum(simulator.daily_pnl[i*days_per_month:(i+1)*days_per_month])
                       for i in range(num_months)]

    fig, axes = plt.subplots(2, 2, figsize=(14, 9))
    fig.suptitle("Calendar Spread Strategy Performance", fontsize=16, fontweight="bold")

    ax1 = axes[0, 0]
    ax1.plot(returns, linewidth=2.5, color="#2E86AB")
    ax1.axhline(0, color="black", linestyle="--", linewidth=1, alpha=0.7)
    ax1.fill_between(range(len(returns)), returns, 0, alpha=0.25, color="#2E86AB")
    ax1.set_title("Cumulative Returns")
    ax1.set_ylabel("Return (%)")
    ax1.set_xlabel("Trading Days")
    ax1.grid(True, alpha=0.4)

    ax2 = axes[0, 1]
    ax2.fill_between(range(len(drawdown)), drawdown, 0, alpha=0.7, color="#E63946")
    ax2.plot(drawdown, linewidth=1.5, color="#8B0000", alpha=0.8)
    ax2.set_title("Drawdown")
    ax2.set_ylabel("Drawdown (%)")
    ax2.set_xlabel("Trading Days")
    ax2.grid(True, alpha=0.4)

    ax3 = axes[1, 0]
    colors = ["#E63946" if r < 0 else "#06A77D" for r in monthly_returns]
    ax3.bar(range(1, len(monthly_returns) + 1), monthly_returns,
            color=colors, alpha=0.85, edgecolor="black", linewidth=1)
    ax3.axhline(0, color="black", linewidth=1.5)
    ax3.set_title("Monthly P&L")
    ax3.set_ylabel("P&L ($)")
    ax3.set_xlabel("Month")
    ax3.grid(True, alpha=0.4, axis="y")

    ax4 = axes[1, 1]
    ax4.axis("off")
    summary = (
        f"  PERFORMANCE SUMMARY\n"
        f"  {'─'*32}\n\n"
        f"  Total Return:     {returns[-1]:>10.2f}%\n\n"
        f"  Sharpe Ratio:     {sr:>10.2f}\n\n"
        f"  Max Drawdown:     {drawdown.min():>10.2f}%\n\n"
        f"  Win Rate:         {win_rate:>10.1f}%\n\n"
        f"  Total Trades:     {len(simulator.trades):>10}\n\n"
        f"  Txn Costs:    ${simulator.transaction_costs:>12,.0f}\n"
    )
    ax4.text(0.05, 0.5, summary, fontsize=11, family="monospace",
             verticalalignment="center")

    plt.tight_layout()
    plt.show()


def calculate_greeks_bs(S: float, K: float, T: float, r: float, sigma: float) -> dict:
    """Black-Scholes Greeks for a call option (vega per 1% vol move)."""
    if T <= 0 or sigma <= 0:
        return {"delta": 0.0, "gamma": 0.0, "vega": 0.0, "theta": 0.0}
    d1 = (np.log(S / K) + (r + 0.5 * sigma**2) * T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma * np.sqrt(T)
    return {
        "delta": norm.cdf(d1),
        "gamma": norm.pdf(d1) / (S * sigma * np.sqrt(T)),
        "vega":  S * norm.pdf(d1) * np.sqrt(T) / 100,
        "theta": (-S * norm.pdf(d1) * sigma / (2 * np.sqrt(T))
                  - r * K * np.exp(-r * T) * norm.cdf(d2)) / 252,
    }


# ── PBO validation
# ─────────────────────────────────────────────

def run_single_backtest(df_data: pd.DataFrame, start_idx: int, end_idx: int) -> float:
    """Run a contained simulation on a data slice; returns annualised Sharpe."""
    sim = CalendarSpreadSimulator()

    for idx in range(start_idx, end_idx):
        row              = df_data.iloc[idx]
        days_since_start = idx - start_idx

        if days_since_start > 0 and days_since_start % 30 == 0:
            intrinsic_30 = max(row["spx_price"] - row["strike"], 0)
            if sim.long_calendars > 0:
                sim.cash += (-sim.long_calendars * intrinsic_30
                             + sim.long_calendars * row["option_60dte_mid"])
                sim.long_calendars = 0
            if sim.short_calendars > 0:
                sim.cash += (sim.short_calendars * intrinsic_30
                             - sim.short_calendars * row["option_60dte_mid"])
                sim.short_calendars = 0

        execute_trades(sim, row, use_costs=True)

        long_cal_mid  = row["option_60dte_mid"] - row["option_30dte_mid"]
        short_cal_mid = row["option_30dte_mid"] - row["option_60dte_mid"]
        pv            = sim.get_portfolio_value(long_cal_mid, short_cal_mid)

        sim.daily_pnl.append(pv - sim.daily_portfolio_value[-1]
                             if sim.daily_portfolio_value else 0)
        sim.daily_portfolio_value.append(pv)

    if len(sim.daily_portfolio_value) < 20:
        return 0.0
    d_ret = np.diff(sim.daily_portfolio_value) / sim.daily_portfolio_value[:-1]
    d_ret = d_ret[np.isfinite(d_ret)]
    return (d_ret.mean() / d_ret.std() * np.sqrt(252)
            if len(d_ret) > 0 and d_ret.std() > 0 else 0.0)


def calculate_pbo(df_data: pd.DataFrame, n_splits: int = PBO_SPLITS,
                  n_trials: int = PBO_TRIALS):
    """
    Probability of Backtest Overfitting via combinatorial symmetric cross-validation.

    At each trial, data splits are randomly assigned 50/50 to in-sample (IS)
    and out-of-sample (OOS) blocks. PBO is the fraction of trials in which the
    median OOS Sharpe falls below the median IS Sharpe, indicating that the
    in-sample selection advantage does not transfer out of sample. A PBO below
    30% suggests low overfitting risk; above 50% is a red flag.
    """
    total      = len(df_data)
    split_size = total // n_splits
    splits     = [(i * split_size, min((i + 1) * split_size, total))
                  for i in range(n_splits) if split_size >= 40]
    n_valid    = len(splits)
    print(f"PBO: {n_valid} splits of ~{split_size} days each")

    if n_valid < 4:
        print("Not enough data for PBO validation.")
        return None, None, None

    is_sharpes_all, oos_sharpes_all = [], []

    for trial in range(n_trials):
        n_is    = n_valid // 2
        is_idx  = np.random.choice(n_valid, size=n_is, replace=False)
        oos_idx = np.array([i for i in range(n_valid) if i not in is_idx])

        is_s  = [s for i in is_idx
                 for s in [run_single_backtest(df_data, *splits[i])]
                 if np.isfinite(s)]
        oos_s = [s for i in oos_idx
                 for s in [run_single_backtest(df_data, *splits[i])]
                 if np.isfinite(s)]

        if is_s and oos_s:
            is_sharpes_all.append(is_s)
            oos_sharpes_all.append(oos_s)

        if (trial + 1) % 10 == 0:
            print(f"  PBO trial {trial + 1}/{n_trials} complete")

    if not is_sharpes_all:
        print("No valid PBO trials.")
        return None, None, None

    oos_worse = sum(np.median(oos) < np.median(is_)
                    for is_, oos in zip(is_sharpes_all, oos_sharpes_all))
    pbo = oos_worse / len(is_sharpes_all)
    return pbo, is_sharpes_all, oos_sharpes_all


# ─────────────────────────────────────────────
# PIPELINE
# ─────────────────────────────────────────────

# ── 1. Load and clean data
df = load_data(CSV_PATH)

# ── 2. Build variance term structure
# The output is a (4022 × 5) daily panel of ATM implied variance at five
# standardised maturities, spanning the full sample from 2010 to 2025.
ts_panel = build_term_structure(df)
print("Term structure shape:", ts_panel.shape)
print(ts_panel.head())

# ── 3. In-sample PCA
# Three factors explain the vast majority of term structure variance.
# In-sample reconstruction RMSE in log-variance ranges from 0.002 (14d)
# to 0.043 (60d), confirming that the 3-factor structure provides a tight
# fit across all maturities.
X, X_mean, pca_full, F_full = fit_pca(ts_panel, n_components=N_FACTORS)

plt.figure(figsize=(6, 4))
plt.plot(np.cumsum(pca_full.explained_variance_ratio_), marker="o")
plt.axhline(0.95, color="r", linestyle="--", label="95%")
plt.title("PCA – Cumulative Explained Variance")
plt.ylabel("Explained variance ratio")
plt.xlabel("Number of factors")
plt.legend()
plt.tight_layout()
plt.show()

X_hat_full = pd.DataFrame(
    F_full @ pca_full.components_ + X_mean.values,
    index=X.index,
    columns=X.columns,
)

fig, axes = plt.subplots(2, 3, figsize=(14, 8))
for i, col in enumerate(X.columns):
    ax = axes.flatten()[i]
    ax.scatter(X[col], X_hat_full[col], alpha=0.3)
    ax.plot([X[col].min(), X[col].max()], [X[col].min(), X[col].max()],
            color="red", linestyle="--")
    ax.set_title(col)
    ax.set_xlabel("True log-variance")
    ax.set_ylabel("Reconstructed")
plt.suptitle("PCA (3 factors) – In-sample reconstruction")
plt.tight_layout()
plt.show()

recon_rmse = np.sqrt(((X - X_hat_full) ** 2).mean())
print("\nRMSE per maturity (log-variance):")
print(recon_rmse)

plt.figure(figsize=(6, 4))
for i in range(N_FACTORS):
    plt.plot(X.columns, pca_full.components_[i], marker="o", label=f"PC{i+1}")
plt.title("PCA Loadings")
plt.ylabel("Loading")
plt.legend()
plt.tight_layout()
plt.show()

# ── 4. Factor time-series analysis
# All three factors are stationary (ADF p-value = 0.000 in all cases),
# which validates the use of mean-reverting AR(1) models for forecasting.
# Half-lives differ substantially across factors: PC1 (level) reverts in
# 15.3 days, PC2 (slope) in 3.8 days, and PC3 (curvature) in only 1.6 days.
# The short half-life of PC3 means its signal is fully dissipated well
# within the 5-day forecast horizon, which is why it is excluded from AR(1)
# modelling (USE_FACTORS = [0, 1]).
fig, axes = plt.subplots(3, 1, figsize=(12, 8), sharex=True)
for i in range(N_FACTORS):
    axes[i].plot(X.index, F_full[:, i])
    axes[i].set_title(f"PCA Factor {i+1}")
plt.tight_layout()
plt.show()

fig, axes = plt.subplots(3, 1, figsize=(10, 8))
for i in range(N_FACTORS):
    plot_acf(F_full[:, i], lags=30, ax=axes[i])
    axes[i].set_title(f"ACF – PC{i+1}")
plt.tight_layout()
plt.show()

print("\nADF test:")
for i in range(N_FACTORS):
    _, pval, *_ = adfuller(F_full[:, i])
    print(f"  PC{i+1} | p-value: {pval:.4f}")

print("\nHalf-life:")
for i in range(N_FACTORS):
    hl = half_life(F_full[:, i])
    print(f"  PC{i+1}: {hl:.1f} days")

# ── 5. Rolling walk-forward backtest
# PCA and AR(1) parameters are re-estimated at every step on a 504-day
# rolling window. No future data enters any estimation, ensuring the
# results are fully out-of-sample.
# Factor accuracy at H=5 days: PC1 Corr=0.834, PC2 Corr=0.557.
# The term structure slope (var_14d − var_180d) achieves a forecast
# correlation of 0.652 with its realised values, providing a viable
# directional signal for trading.
F_true, F_hat, X_true, X_hat = rolling_ar1_backtest(
    X, window=WINDOW, h=H, n_factors=N_FACTORS, use_factors=USE_FACTORS
)

slope_true = X_true["var_14d"] - X_true["var_180d"]
slope_hat  = X_hat["var_14d_hat"] - X_hat["var_180d_hat"]

print("\nFactor accuracy (H-step):")
for i in range(N_FACTORS):
    f = F_true[f"PC{i+1}"].values
    g = F_hat[f"PC{i+1}_hat"].values
    print(f"  PC{i+1}: RMSE={rmse(f, g):.4f}  Corr={np.corrcoef(f, g)[0, 1]:.3f}")

print(f"\nSlope correlation (true vs forecast): {np.corrcoef(slope_true, slope_hat)[0, 1]:.3f}")

# ── 6. Slope diagnostics
# These plots decompose forecast quality by volatility regime (PC1 level)
# and signal strength, providing insight into where the model adds value
# and where it degrades. The regime-coloured scatter isolates whether
# forecast errors concentrate in high-stress periods.
pc1   = F_true["PC1"]
pc1_q = pc1.quantile([0.33, 0.66])
regime = pd.cut(pc1,
                bins=[-np.inf, pc1_q.iloc[0], pc1_q.iloc[1], np.inf],
                labels=["Low level", "Mid level", "High level"])

plt.figure(figsize=(7, 6))
for label, color in zip(["Low level", "Mid level", "High level"], ["green", "orange", "red"]):
    idx = regime == label
    plt.scatter(slope_true[idx], slope_hat[idx], alpha=0.4, label=label, color=color)
mn = min(slope_true.min(), slope_hat.min())
mx = max(slope_true.max(), slope_hat.max())
plt.plot([mn, mx], [mn, mx], "k--")
plt.xlabel("Realized slope (log-var)")
plt.ylabel("Forecast slope (log-var)")
plt.title("Slope forecast vs realized — coloured by PC1 regime")
plt.legend()
plt.tight_layout()
plt.show()

error = slope_hat - slope_true

plt.figure(figsize=(7, 5))
plt.scatter(pc1, error, alpha=0.4)
plt.axhline(0, color="k", linestyle="--")
plt.xlabel("PC1 (level)")
plt.ylabel("Forecast error (slope)")
plt.title("Slope forecast error vs PC1 level")
plt.tight_layout()
plt.show()

wrong_sign = np.sign(slope_hat) != np.sign(slope_true)

plt.figure(figsize=(7, 5))
plt.scatter(pc1[~wrong_sign], slope_true[~wrong_sign], alpha=0.3, label="Correct sign", color="blue")
plt.scatter(pc1[wrong_sign],  slope_true[wrong_sign],  alpha=0.6, label="Wrong sign",   color="red")
plt.axhline(0, color="k", linestyle="--")
plt.xlabel("PC1 (level)")
plt.ylabel("Realized slope")
plt.title("Wrong-sign forecasts vs PC1 level")
plt.legend()
plt.tight_layout()
plt.show()

plt.figure(figsize=(7, 5))
plt.scatter(np.abs(slope_hat), np.abs(error), alpha=0.4)
plt.xlabel("|Forecast slope|")
plt.ylabel("|Forecast error|")
plt.title("Forecast error vs signal strength")
plt.tight_layout()
plt.show()

rw_hat = slope_true.shift(1).loc[slope_hat.index]

plt.figure(figsize=(7, 6))
plt.scatter(rw_hat - slope_true, slope_hat - slope_true, alpha=0.4)
plt.axhline(0, color="k", linestyle="--")
plt.axvline(0, color="k", linestyle="--")
plt.xlabel("Random Walk error")
plt.ylabel("AR(1) error")
plt.title("AR(1) vs Random Walk errors (slope)")
plt.tight_layout()
plt.show()

# ── 7. Signal construction and evaluation
# The signal is built in two layers:
#   Layer 1 — noise filter: only forecasts whose absolute z-score exceeds 0.5
#     (normalised over a 60-day window) are treated as actionable. This removes
#     low-conviction, near-zero predictions.
#   Layer 2 — regime filter: forecasts are suppressed when PC1 is more than
#     1 standard deviation above its 1-year mean, i.e., during elevated
#     volatility regimes where model assumptions are least likely to hold.
# The regime filter reduces coverage from 89.9% to 70.1% while improving
# hit-rate from 0.865 to 0.907, confirming that the filtered-out observations
# are disproportionately low-quality predictions.
# Note: masked_corr returns NaN because the z-scored signal, after zeroing
# out filtered observations, has near-zero variance in the active subset,
# making correlation numerically undefined. Hit-rate is the more appropriate
# metric in this context.
signal_z  = slope_hat / slope_hat.rolling(SIG_WIN).std()
signal_ok = signal_z.abs() > SIG_Z_TH

pc1_z     = (pc1 - pc1.rolling(PC1_WIN).mean()) / pc1.rolling(PC1_WIN).std()
regime_ok = pc1_z < PC1_Z_TH

signal_nofilter = signal_z * signal_ok
signal_filtered = signal_z * signal_ok * regime_ok

print("\nSignal evaluation:")
print(f"  Coverage (no filter) : {(signal_nofilter != 0).mean():.2%}")
print(f"  Coverage (filtered)  : {(signal_filtered != 0).mean():.2%}")
print(f"  Hit-rate (no filter) : {hit_rate(signal_nofilter, slope_true):.3f}")
print(f"  Hit-rate (filtered)  : {hit_rate(signal_filtered, slope_true):.3f}")
print(f"  Corr (no filter)     : {masked_corr(signal_nofilter, slope_true):.3f}")
print(f"  Corr (filtered)      : {masked_corr(signal_filtered, slope_true):.3f}")

plt.figure(figsize=(12, 4))
plt.plot(slope_true,      label="Realized slope",      color="black", alpha=0.5)
plt.plot(signal_nofilter, label="Signal (no filter)",  alpha=0.6)
plt.plot(signal_filtered, label="Signal (PC1 filter)", alpha=0.9)
plt.legend()
plt.title("Slope forecast signal vs realized — with and without PC1 filter")
plt.tight_layout()
plt.show()

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

ax1.scatter(signal_nofilter, slope_true, alpha=0.3)
ax1.axline((0, 0), (1, 1), linestyle="--")
ax1.set_title("No filter")
ax1.set_xlabel("Forecast signal")
ax1.set_ylabel("Realized slope")

idx = signal_filtered != 0
ax2.scatter(signal_filtered[idx], slope_true[idx], alpha=0.3)
ax2.axline((0, 0), (1, 1), linestyle="--")
ax2.set_title("With PC1 filter")
ax2.set_xlabel("Forecast signal")
ax2.set_ylabel("Realized slope")

plt.tight_layout()
plt.show()

# ── 8. VIX futures strategy
# The signal drives a long/short position on the VX continuous front-month
# future, rebalanced every 5 days. P&L is expressed in VIX points, consistent
# with how VIX futures are margined and quoted.
# Position sizing is binary (sign of signal only) to keep the strategy simple
# and interpretable. A one-day execution lag (shift(1)) prevents using same-day
# information in the position decision.
# Out-of-sample results: Sharpe 0.36, Max Drawdown −20.86 VIX points,
# active on 68.7% of trading days. The benchmark (always short VIX) is
# included as a structural reference, given the well-documented negative
# risk premium embedded in VIX futures.
vx = load_vx(VX_PATH)

common_idx = slope_hat.index.intersection(F_true.index).intersection(vx.index)

vx           = vx.loc[common_idx].copy()
signal_strat = slope_hat.loc[common_idx].copy()
pc1_strat    = F_true.loc[common_idx, "PC1"].copy()

signal_z_strat  = signal_strat / signal_strat.rolling(SIG_WIN).std()
signal_ok_strat = signal_z_strat.abs() > SIG_Z_TH

pc1_z_strat     = (pc1_strat - pc1_strat.rolling(PC1_WIN).mean()) / pc1_strat.rolling(PC1_WIN).std()
regime_ok_strat = pc1_z_strat < PC1_Z_TH

final_signal = signal_z_strat.where(signal_ok_strat & regime_ok_strat, 0.0)

position = np.sign(final_signal)
position_5d = (
    position
    .iloc[::REBAL_STEP]
    .reindex(position.index, method="ffill")
    .shift(1)
    .fillna(0.0)
)

vx_ret  = vx.diff()
pnl     = (position_5d * vx_ret).fillna(0.0)
cum_pnl = pnl.cumsum()

print("\nStrategy metrics:")
print(f"  Sharpe               : {sharpe(pnl):.2f}")
print(f"  Max Drawdown         : {max_drawdown(cum_pnl):.2f}")
print(f"  Coverage (% active)  : {(position_5d != 0).mean() * 100:.1f}%")

plt.figure(figsize=(12, 4))
cum_pnl.plot()
plt.title("VIX Strategy P&L (PC2 signal + PC1 filter, 5d rebalance)")
plt.tight_layout()
plt.show()

plt.figure(figsize=(6, 4))
pnl.hist(bins=50)
plt.title("Distribution of daily P&L")
plt.tight_layout()
plt.show()

bench_pnl = -vx_ret.fillna(0.0)
bench_cum = bench_pnl.cumsum()

plt.figure(figsize=(12, 4))
bench_cum.plot(label="Always short VIX")
cum_pnl.plot(label="Signal-driven")
plt.legend()
plt.title("Strategy vs benchmark (always short VIX)")
plt.tight_layout()
plt.show()

# ── 9. Calendar spread strategy
# The calendar spread trades the term structure slope directly in option space.
# A LONG calendar (sell 30d, buy 60d) profits when short-term IV rises relative
# to long-term IV (contango normalises). A SHORT calendar profits in the
# opposite regime. Both legs are ATM calls, selected as the strike closest
# to spx_close within the respective DTE window.

# ── 9a. Build calendar spread dataset
# ATM calls are selected independently for the 30d and 60d legs: for each
# trade date, the option with the smallest DTE deviation from the target
# and the smallest absolute moneyness is retained. The two legs are then
# merged on trade_date to form a paired calendar spread record.
df_calls = df[df["option_type"].astype(str).str.upper().str.startswith("C")].copy()

df_30 = df_calls[(df_calls["dte"] >= 20) & (df_calls["dte"] <= 40)].copy()
df_30["dte_diff"]  = (df_30["dte"] - 30).abs()
df_30["moneyness"] = (df_30["strike"] - df_30["spx_close"]).abs()
df_30 = df_30.sort_values(["trade_date", "dte_diff", "moneyness"])
df_30_atm = df_30.drop_duplicates(subset="trade_date", keep="first")

df_60 = df_calls[(df_calls["dte"] >= 50) & (df_calls["dte"] <= 70)].copy()
df_60["dte_diff"]  = (df_60["dte"] - 60).abs()
df_60["moneyness"] = (df_60["strike"] - df_60["spx_close"]).abs()
df_60 = df_60.sort_values(["trade_date", "dte_diff", "moneyness"])
df_60_atm = df_60.drop_duplicates(subset="trade_date", keep="first")

print(f"30 DTE leg: {len(df_30_atm)} days")
print(f"60 DTE leg: {len(df_60_atm)} days")

df_calendar = pd.merge(
    df_30_atm[["trade_date", "strike", "bid", "ask", "iv_mid", "dte", "spx_close"]],
    df_60_atm[["trade_date", "strike", "bid", "ask", "iv_mid", "dte"]],
    on="trade_date",
    suffixes=("_30", "_60"),
    how="inner",
)

df_calendar["option_30dte_mid"] = (df_calendar["bid_30"] + df_calendar["ask_30"]) / 2
df_calendar["option_60dte_mid"] = (df_calendar["bid_60"] + df_calendar["ask_60"]) / 2
df_calendar["option_30dte_bid"] = df_calendar["bid_30"]
df_calendar["option_30dte_ask"] = df_calendar["ask_30"]
df_calendar["option_60dte_bid"] = df_calendar["bid_60"]
df_calendar["option_60dte_ask"] = df_calendar["ask_60"]

df_calendar = df_calendar.rename(columns={
    "trade_date": "date",
    "spx_close":  "spx_price",
    "strike_30":  "strike",
})

# ── 9b. Align with pipeline signals
# slope_hat and F_true["PC1"] are DatetimeIndex Series; merging on date
# ensures exact temporal alignment with no index mismatches.
slope_hat_df = slope_hat.rename("slope_forecast").reset_index()
slope_hat_df.columns = ["date", "slope_forecast"]

pc1_df = F_true["PC1"].rename("pc1").reset_index()
pc1_df.columns = ["date", "pc1"]

df_real = (
    df_calendar
    .merge(slope_hat_df, on="date", how="inner")
    .merge(pc1_df, on="date", how="inner")
    .reset_index(drop=True)
)

print(f"\nCalendar dataset: {df_real.shape[0]} trading days")
print(f"Date range: {df_real['date'].min().date()} to {df_real['date'].max().date()}")

# ── 9c. Generate signals
# Signals are generated row-by-row; rows with a missing slope forecast
# (arising from rolling-window warm-up) are assigned a flat position.
config = RuleConfig(aggressiveness=AGGRESSIVENESS)

df_real["signal"] = df_real.apply(
    lambda row: generate_signal(row, config)
    if pd.notna(row.get("slope_forecast"))
    else {"long_calendars": 0, "short_calendars": 0},
    axis=1,
)

for agg in [1, 3, 5, 7, 10]:
    signal_diagnostics(df_real, RuleConfig(aggressiveness=agg))

# ── 9d. Main simulation
print("\n" + "="*60)
print("RUNNING SIMULATION")
print("="*60)

simulator = CalendarSpreadSimulator()

for idx, row in df_real.iterrows():
    exp_cost = handle_expiration(simulator, idx, df_real)
    txn_cost = execute_trades(simulator, row, use_costs=True)

    long_cal_mid  = row["option_60dte_mid"] - row["option_30dte_mid"]
    short_cal_mid = row["option_30dte_mid"] - row["option_60dte_mid"]
    portfolio_val = simulator.get_portfolio_value(long_cal_mid, short_cal_mid)

    daily_pnl_cal = (portfolio_val - simulator.daily_portfolio_value[-1]
                     if simulator.daily_portfolio_value else 0)

    simulator.daily_pnl.append(daily_pnl_cal)
    simulator.daily_portfolio_value.append(portfolio_val)
    simulator.transaction_costs += exp_cost

print(f"  Long calendars open : {simulator.long_calendars}")
print(f"  Short calendars open: {simulator.short_calendars}")
print(f"  Final portfolio     : ${simulator.daily_portfolio_value[-1]:,.2f}")
print(f"  Total trades        : {len(simulator.trades)}")

# ── 9e. Performance metrics and plots
metrics = calculate_performance_metrics(simulator)
print_performance_report(metrics)
plot_clean_performance(simulator)

# ── 9f. Greeks tracking
# iv_mid_30 and iv_mid_60 from the real option data feed Black-Scholes to
# compute net Greeks of each calendar spread position. Net vega of a long
# calendar is positive (long vol); net theta is negative (paying time decay
# on the far leg net of receipt from the near leg).
print("\nRunning simulation with Greeks tracking...")
sim_greeks = EnhancedSimulator()

for idx, row in df_real.iterrows():
    handle_expiration(sim_greeks, idx, df_real)
    execute_trades(sim_greeks, row, use_costs=True)

    long_cal_mid  = row["option_60dte_mid"] - row["option_30dte_mid"]
    short_cal_mid = row["option_30dte_mid"] - row["option_60dte_mid"]
    portfolio_val = sim_greeks.get_portfolio_value(long_cal_mid, short_cal_mid)

    S    = row["spx_price"]
    K    = row["strike"]
    iv30 = row.get("iv_mid_30", row.get("iv_mid", 0.15))
    iv60 = row.get("iv_mid_60", row.get("iv_mid", 0.15))

    g30 = calculate_greeks_bs(S, K, 30 / 252, R_DEFAULT, iv30)
    g60 = calculate_greeks_bs(S, K, 60 / 252, R_DEFAULT, iv60)

    net_vega  = (sim_greeks.long_calendars  * (g60["vega"]  - g30["vega"])
               - sim_greeks.short_calendars * (g30["vega"]  - g60["vega"]))
    net_theta = (sim_greeks.long_calendars  * (g60["theta"] - g30["theta"])
               - sim_greeks.short_calendars * (g30["theta"] - g60["theta"]))
    net_gamma = (sim_greeks.long_calendars  * (g60["gamma"] - g30["gamma"])
               - sim_greeks.short_calendars * (g30["gamma"] - g60["gamma"]))
    net_delta = (sim_greeks.long_calendars  * (g60["delta"] - g30["delta"])
               - sim_greeks.short_calendars * (g30["delta"] - g60["delta"]))

    sim_greeks.daily_vega.append(net_vega)
    sim_greeks.daily_theta.append(net_theta)
    sim_greeks.daily_gamma.append(net_gamma)
    sim_greeks.daily_delta.append(net_delta)

    daily_pnl_cal = (portfolio_val - sim_greeks.daily_portfolio_value[-1]
                     if sim_greeks.daily_portfolio_value else 0)
    sim_greeks.daily_pnl.append(daily_pnl_cal)
    sim_greeks.daily_portfolio_value.append(portfolio_val)

print("Greeks simulation complete.")

# ── 9g. P&L attribution and regime analysis
df_analysis = pd.DataFrame({
    "pnl":   sim_greeks.daily_pnl[1:],
    "vega":  sim_greeks.daily_vega[:-1],
    "theta": sim_greeks.daily_theta[:-1],
    "gamma": sim_greeks.daily_gamma[:-1],
    "delta": sim_greeks.daily_delta[:-1],
})

print("\nGreeks vs P&L correlations:")
for greek in ["vega", "theta", "gamma", "delta"]:
    print(f"  {greek.capitalize()}: {df_analysis[greek].corr(df_analysis['pnl']):.3f}")

in_pos = df_analysis["vega"] != 0
print("\nAverage Greeks exposure (when in position):")
for greek in ["vega", "theta", "gamma", "delta"]:
    fmt = ".4f" if greek in ["gamma", "delta"] else ".2f"
    print(f"  Avg {greek.capitalize()}: {df_analysis.loc[in_pos, greek].mean():{fmt}}")

df_real_analysis = df_real.copy()
df_real_analysis["daily_pnl"]      = sim_greeks.daily_pnl
df_real_analysis["portfolio_value"] = sim_greeks.daily_portfolio_value

df_real_analysis["pc1_regime"] = pd.cut(
    df_real_analysis["pc1"],
    bins=[-np.inf, -0.5, 0.5, np.inf],
    labels=["Low Vol (PC1<-0.5)", "Normal Vol", "High Vol (PC1>0.5)"],
)

print("\nPerformance by PC1 regime:")
for reg in ["Low Vol (PC1<-0.5)", "Normal Vol", "High Vol (PC1>0.5)"]:
    sub = df_real_analysis[df_real_analysis["pc1_regime"] == reg]
    if len(sub) > 0:
        print(f"  {reg}: Avg P&L = ${sub['daily_pnl'].mean():.2f}, "
              f"Win rate = {(sub['daily_pnl'] > 0).mean()*100:.1f}%")

df_real_analysis["signal_strength"] = df_real_analysis["slope_forecast"].abs()
print("\nP&L by signal strength quartile:")
for label, q_lo, q_hi in [("Q1 (Weakest)", 0.00, 0.25), ("Q2", 0.25, 0.50),
                            ("Q3", 0.50, 0.75), ("Q4 (Strongest)", 0.75, 1.00)]:
    lo  = df_real_analysis["signal_strength"].quantile(q_lo)
    hi  = df_real_analysis["signal_strength"].quantile(q_hi)
    sub = df_real_analysis[(df_real_analysis["signal_strength"] >= lo) &
                            (df_real_analysis["signal_strength"] <= hi)]
    print(f"  {label}: Avg P&L = ${sub['daily_pnl'].mean():.2f}, "
          f"Win rate = {(sub['daily_pnl'] > 0).mean()*100:.1f}%")

df_real_analysis["position_size"] = df_real_analysis["signal"].apply(
    lambda x: x["long_calendars"] - x["short_calendars"]
)
df_real_analysis["position_change"] = (
    df_real_analysis["position_size"] != df_real_analysis["position_size"].shift()
).cumsum()
holding_periods = df_real_analysis.groupby("position_change").size()

print(f"\nHolding period statistics:")
print(f"  Mean:   {holding_periods.mean():.1f} days")
print(f"  Median: {holding_periods.median():.1f} days")
print(f"  Min:    {holding_periods.min()} days")
print(f"  Max:    {holding_periods.max()} days")

# ── 9h. Six-panel advanced visualisation
fig, axes = plt.subplots(3, 2, figsize=(16, 12))
fig.suptitle("Advanced Performance Analysis with Greeks", fontsize=16, fontweight="bold")

axes[0, 0].plot(sim_greeks.daily_vega,  label="Vega",  alpha=0.7)
axes[0, 0].plot(sim_greeks.daily_theta, label="Theta", alpha=0.7)
axes[0, 0].axhline(0, color="black", linestyle="--", linewidth=0.5)
axes[0, 0].set_title("Greeks Exposure Over Time")
axes[0, 0].set_ylabel("Exposure")
axes[0, 0].legend()
axes[0, 0].grid(True, alpha=0.3)

vega_pnl_corr = df_analysis["vega"].corr(df_analysis["pnl"])
axes[0, 1].scatter(df_analysis["vega"], df_analysis["pnl"], alpha=0.3, s=10)
axes[0, 1].axhline(0, color="black", linestyle="--", linewidth=0.5)
axes[0, 1].axvline(0, color="black", linestyle="--", linewidth=0.5)
axes[0, 1].set_title(f"P&L vs Vega Exposure (corr={vega_pnl_corr:.3f})")
axes[0, 1].set_xlabel("Vega Exposure")
axes[0, 1].set_ylabel("Daily P&L ($)")
axes[0, 1].grid(True, alpha=0.3)

regime_pnl, regime_labels = [], []
for reg in ["Low Vol (PC1<-0.5)", "Normal Vol", "High Vol (PC1>0.5)"]:
    sub = df_real_analysis[df_real_analysis["pc1_regime"] == reg]["daily_pnl"]
    if len(sub) > 0:
        regime_pnl.append(sub.values)
        regime_labels.append(reg)
axes[1, 0].boxplot(regime_pnl, labels=regime_labels)
axes[1, 0].axhline(0, color="red", linestyle="--", linewidth=0.5)
axes[1, 0].set_title("P&L Distribution by PC1 Regime")
axes[1, 0].set_ylabel("Daily P&L ($)")
axes[1, 0].grid(True, alpha=0.3, axis="y")

axes[1, 1].scatter(df_real_analysis["signal_strength"],
                   df_real_analysis["daily_pnl"], alpha=0.3, s=10)
axes[1, 1].axhline(0, color="black", linestyle="--", linewidth=0.5)
axes[1, 1].set_title("Signal Strength vs P&L")
axes[1, 1].set_xlabel("|Slope Forecast|")
axes[1, 1].set_ylabel("Daily P&L ($)")
axes[1, 1].grid(True, alpha=0.3)

axes[2, 0].hist(holding_periods, bins=30, alpha=0.7, edgecolor="black")
axes[2, 0].axvline(holding_periods.mean(),   color="red",   linestyle="--", linewidth=2,
                    label=f"Mean: {holding_periods.mean():.1f}d")
axes[2, 0].axvline(holding_periods.median(), color="green", linestyle="--", linewidth=2,
                    label=f"Median: {holding_periods.median():.1f}d")
axes[2, 0].set_title("Holding Period Distribution")
axes[2, 0].set_xlabel("Days in Position")
axes[2, 0].set_ylabel("Frequency")
axes[2, 0].legend()
axes[2, 0].grid(True, alpha=0.3, axis="y")

cum_pnl_cal = np.cumsum(sim_greeks.daily_pnl)
axes[2, 1].plot(cum_pnl_cal, linewidth=2)
for reg, color in [("Low Vol (PC1<-0.5)", "green"),
                    ("Normal Vol",          "yellow"),
                    ("High Vol (PC1>0.5)",  "red")]:
    for period in df_real_analysis[df_real_analysis["pc1_regime"] == reg].index:
        axes[2, 1].axvspan(period - 0.5, period + 0.5, alpha=0.1, color=color)
axes[2, 1].set_title("Cumulative P&L with PC1 Regime Overlay")
axes[2, 1].set_xlabel("Trading Day")
axes[2, 1].set_ylabel("Cumulative P&L ($)")
axes[2, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

# ── 9i. PBO validation
# Note: with PBO_SPLITS=10 and PBO_TRIALS=150 this section takes ~10–20 minutes.
pbo_score, is_sharpes, oos_sharpes = calculate_pbo(df_real, PBO_SPLITS, PBO_TRIALS)

if pbo_score is not None:
    print(f"\n{'='*60}")
    print("PROBABILITY OF BACKTEST OVERFITTING (PBO)")
    print(f"{'='*60}")
    print(f"\nPBO Score: {pbo_score:.2%}")
    if pbo_score < 0.3:
        print("  Low overfitting risk (PBO < 30%)")
    elif pbo_score < 0.5:
        print("  Moderate overfitting risk (30% < PBO < 50%)")
    else:
        print("  High overfitting risk (PBO > 50%)")

    is_flat  = [s for trial in is_sharpes  for s in trial]
    oos_flat = [s for trial in oos_sharpes for s in trial]
    print(f"\nMedian IS Sharpe:  {np.median(is_flat):.3f}")
    print(f"Median OOS Sharpe: {np.median(oos_flat):.3f}")
    print(f"Mean IS Sharpe:    {np.mean(is_flat):.3f}")
    print(f"Mean OOS Sharpe:   {np.mean(oos_flat):.3f}")
    print(f"{'='*60}")

# ── 9j. Dynamic drawdown analysis
# The two worst drawdown troughs are identified automatically from the equity
# curve; each is examined over a ±100-day window to provide context on the
# market conditions (SPX range, timing) surrounding the loss period.
portfolio_arr = np.array(simulator.daily_portfolio_value)
running_max   = np.maximum.accumulate(portfolio_arr)
drawdown_arr  = (portfolio_arr - running_max) / running_max * 100
df_real["drawdown"]        = drawdown_arr
df_real["portfolio_value"] = portfolio_arr

trough_indices = np.argsort(drawdown_arr)[:2]

for rank, trough_idx in enumerate(trough_indices, start=1):
    start = max(trough_idx - 100, 0)
    end   = min(trough_idx + 100, len(df_real) - 1)
    sub   = df_real.iloc[start:end]
    worst = sub["drawdown"].idxmin()
    print(f"\nMAJOR DRAWDOWN #{rank}:")
    print(f"  Worst point : {df_real.loc[worst, 'date'].date()}")
    print(f"  Drawdown    : {df_real.loc[worst, 'drawdown']:.1f}%")
    print(f"  Period      : {sub['date'].min().date()} to {sub['date'].max().date()}")
    print(f"  SPX range   : {sub['spx_price'].min():.0f} – {sub['spx_price'].max():.0f}")
