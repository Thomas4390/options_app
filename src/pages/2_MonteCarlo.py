import streamlit as st
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import norm
import plotly.graph_objects as go
import time

@dataclass
class GBMParameters:
    spot: float
    mu: float
    sigma: float
    dt: float
    seed: int = 42

@dataclass
class OptionContract:
    strike: float
    maturity: float
    is_call: bool

class GBMSimulator:
    def __init__(self, params: GBMParameters) -> None:
        """
        Initialize the GBM simulator with given parameters for multiple simulations.
        """
        self.params = params

    def simulate_multiple(self, n_sim: int, nb_steps: int) -> np.ndarray:
        """
        Simulate multiple GBM paths.

        Parameters
        ----------
        n_sim : int
            Number of simulations.
        nb_steps : int
            Number of time steps per simulation.

        Returns
        -------
        np.ndarray
            A 2D array of shape (n_sim, nb_steps) with simulated paths.
        """
        np.random.seed(self.params.seed)
        S0 = self.params.spot
        mu = self.params.mu
        sigma = self.params.sigma
        dt = self.params.dt

        increments = (mu - 0.5*sigma**2)*dt + sigma*np.sqrt(dt)*np.random.randn(n_sim, nb_steps)
        log_prices = np.log(S0) + np.cumsum(increments, axis=1)
        return np.exp(log_prices)

class GreeksCalculator:
    @staticmethod
    def compute_price_vectorized(
        S_val: np.ndarray,
        K: np.ndarray,
        r: float,
        sigma: float,
        T: np.ndarray,
        is_call: np.ndarray
    ) -> np.ndarray:
        """
        Compute vectorized option prices under Black-Scholes assumptions.

        Parameters
        ----------
        S_val : np.ndarray
            Underlying price paths (n_sim, nb_steps, 4).
        K : np.ndarray
            Strike array.
        r : float
            Risk-free rate.
        sigma : float
            Volatility.
        T : np.ndarray
            Time-to-maturity array.
        is_call : np.ndarray
            Boolean array indicating call or put.

        Returns
        -------
        np.ndarray
            Option prices of shape (n_sim, nb_steps, 4).
        """
        payoff_call = np.maximum(S_val - K, 0)
        payoff_put = np.maximum(K - S_val, 0)

        safe_T = np.where(T>0,T,np.nan)
        with np.errstate(divide='ignore', invalid='ignore'):
            d1 = (np.log(S_val/K)+(r+0.5*sigma**2)*safe_T)/(sigma*np.sqrt(safe_T))
            d2 = d1 - sigma*np.sqrt(safe_T)

        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)

        call_price = S_val*cdf_d1 - K*np.exp(-r*T)*cdf_d2
        put_price = K*np.exp(-r*T)*norm.cdf(-d2)-S_val*norm.cdf(-d1)

        price = np.where(T>0, np.where(is_call, call_price, put_price),
                         np.where(is_call, np.maximum(S_val-K,0), np.maximum(K-S_val,0)))
        return price

def compute_iron_condor_initial(
    spot: float, K1: float, K2: float, K3: float, K4: float,
    r: float, sigma: float, maturity: float
) -> float:
    """
    Compute the initial cost of the Iron Condor strategy.
    """
    strikes = np.array([K1,K2,K3,K4])
    is_call = np.array([False,False,True,True])
    leg_factors = np.array([+1,-1,-1,+1])
    T0 = np.full_like(strikes, maturity)
    S0_arr = np.full_like(strikes, spot)

    payoff_call = np.maximum(S0_arr - strikes,0)
    payoff_put = np.maximum(strikes - S0_arr,0)
    safe_T = np.where(T0>0,T0,np.nan)
    d1 = (np.log(S0_arr/strikes)+(r+0.5*sigma**2)*safe_T)/(sigma*np.sqrt(safe_T))
    d2 = d1 - sigma*np.sqrt(safe_T)
    cdf_d1 = norm.cdf(d1)
    cdf_d2 = norm.cdf(d2)
    call_price = S0_arr*cdf_d1 - strikes*np.exp(-r*T0)*cdf_d2
    put_price = strikes*np.exp(-r*T0)*norm.cdf(-d2)-S0_arr*norm.cdf(-d1)
    price = np.where(T0>0, np.where(is_call,call_price,put_price),
                     np.where(is_call, payoff_call, payoff_put))
    initial_cost = np.sum(price*leg_factors)
    return initial_cost

def compute_final_pnl_expiry(
    K1: float, K2: float, K3: float, K4: float, initial_cost: float
) -> tuple[float,float,np.ndarray,np.ndarray,list]:
    """
    Compute final P&L at expiry, max profit, max loss, and break-even points for Iron Condor.
    """
    S_range = np.linspace(K1-20, K4+20, 300)
    payoff_long_put = np.maximum(K1 - S_range, 0)
    payoff_short_put = -np.maximum(K2 - S_range, 0)
    payoff_short_call = -np.maximum(S_range - K3, 0)
    payoff_long_call = np.maximum(S_range - K4, 0)
    final_intrinsic = payoff_long_put+payoff_short_put+payoff_short_call+payoff_long_call
    final_pnl_at_expiry = final_intrinsic - initial_cost
    max_profit = np.max(final_pnl_at_expiry)
    max_loss = np.min(final_pnl_at_expiry)

    sign_change_indices = np.where(np.diff(np.sign(final_pnl_at_expiry))!=0)[0]
    break_evens = []
    for idx in sign_change_indices:
        x0,x1 = S_range[idx],S_range[idx+1]
        y0,y1 = final_pnl_at_expiry[idx],final_pnl_at_expiry[idx+1]
        be = x0 - y0*(x1-x0)/(y1 - y0)
        break_evens.append(be)
    return max_profit, max_loss, final_pnl_at_expiry, S_range, break_evens

def find_exit_time_retro_for_one(
    pnl: np.ndarray, tp: float, sl: float, buff: int,
    max_profit: float, max_loss: float
) -> tuple[float | None, bool | None]:
    """
    Find exit time for a single simulation based on TP/SL conditions.
    """
    nb_steps = len(pnl)
    inside_range_indices = np.where((pnl>max_loss)&(pnl<max_profit))[0]
    if len(inside_range_indices)==0:
        return None, None
    start_monitoring_step = inside_range_indices[0]
    for t in range(start_monitoring_step, nb_steps):
        if pnl[t]>=tp or pnl[t]<=sl:
            exit_idx = t+buff
            if exit_idx<nb_steps:
                tp_hit = (pnl[t]>=tp)
                return exit_idx, tp_hit
            else:
                return None, None
    return None, None

def compute_var_es(pnl_array: np.ndarray, alpha: float=0.05) -> tuple[float,float]:
    """
    Compute VaR and ES at given alpha level.
    """
    sorted_pnl = np.sort(pnl_array)
    var_idx = int(alpha*len(sorted_pnl))
    var_idx = min(var_idx, len(sorted_pnl)-1)
    var = sorted_pnl[var_idx]
    es = np.mean(sorted_pnl[:var_idx]) if var_idx>0 else var
    return var, es

def probability_of_threshold_hit(pnl_array: np.ndarray, threshold: float) -> float:
    """
    Compute the probability that final P&L is above a given threshold.
    """
    return np.mean(pnl_array>=threshold)

def compute_percentiles(data: np.ndarray, lower: float=5, upper: float=95) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute lower percentile, mean, and upper percentile values over an array.
    """
    low_val = np.percentile(data, lower, axis=0)
    high_val = np.percentile(data, upper, axis=0)
    mean_val = np.mean(data, axis=0)
    return low_val, mean_val, high_val

st.title("Monte Carlo Simulation (Iron Condor) with TP/SL, VaR/ES, Percentile Bands")

st.sidebar.header("Iron Condor Parameters")
K1 = st.sidebar.number_input("Long Put (K1)", value=90.0)
K2 = st.sidebar.number_input("Short Put (K2)", value=95.0)
K3 = st.sidebar.number_input("Short Call (K3)", value=105.0)
K4 = st.sidebar.number_input("Long Call (K4)", value=110.0)
if not (K1<K2<K3<K4):
    st.error("K1<K2<K3<K4 condition not met.")

st.sidebar.header("Market and Simulation Parameters")
spot = st.sidebar.number_input("Initial Spot", value=100.0)
mu = st.sidebar.number_input("Mu (Annual Drift)", value=0.05)
sigma = st.sidebar.number_input("Sigma (Annual Vol)", value=0.2)
r = st.sidebar.number_input("Interest Rate (r)", value=0.01)
seed = st.sidebar.number_input("Seed", value=42)

time_interval = st.sidebar.selectbox("Time Interval", ["Hours","Minutes","Seconds"], index=0)
if time_interval == "Seconds":
    steps_per_day = 24*3600
elif time_interval == "Minutes":
    steps_per_day = 24*60
else:
    steps_per_day = 24

nb_days = st.sidebar.number_input("Number of Days", value=30)
nb_steps = int(nb_days*steps_per_day)
dt = (1/365)/steps_per_day

st.sidebar.header("Maturity")
maturity_days = st.sidebar.number_input("Maturity (Days)", value=30)
maturity = maturity_days/365

initial_cost = compute_iron_condor_initial(spot,K1,K2,K3,K4,r,sigma,maturity)
max_profit, max_loss, final_pnl_at_expiry, S_range, break_evens = compute_final_pnl_expiry(K1,K2,K3,K4,initial_cost)

# Only Absolute TP/SL
st.sidebar.header("Stop Loss / Take Profit (Absolute)")
take_profit_input = st.sidebar.number_input("Take Profit (TP)", value=float(max_profit))
stop_loss_input = st.sidebar.number_input("Stop Loss (SL)", value=float(max_loss))
buffer_steps = st.sidebar.number_input("Buffer (time steps)", value=1, min_value=0)

st.sidebar.header("Monte Carlo Settings")
n_sim = st.sidebar.number_input("Number of simulations", value=1000)

st.sidebar.header("Investment")
investment_amount = st.sidebar.number_input("Investment Amount", value=10000.0, min_value=0.0)

st.sidebar.header("Risk Measures")
alpha = st.sidebar.number_input("VaR/ES Alpha", value=0.05, min_value=0.0, max_value=1.0)
threshold_for_prob = st.sidebar.number_input("P&L Threshold for Probability", value=0.0)

st.sidebar.header("Percentile Bands")
user_percentile = st.sidebar.number_input("Percentile (e.g. 5 means 5/95)", value=5, min_value=1, max_value=49)
lower_percentile = user_percentile
upper_percentile = 100 - user_percentile

if "sim_index" not in st.session_state:
    st.session_state.sim_index = 0

run_clicked = st.button("Run Monte Carlo")
if run_clicked:
    start = time.time()
    gbm_params = GBMParameters(spot=spot, mu=mu, sigma=sigma, dt=dt, seed=seed)
    simulator = GBMSimulator(params=gbm_params)
    price_paths = simulator.simulate_multiple(n_sim, nb_steps)

    strikes = np.array([K1,K2,K3,K4])
    is_call = np.array([False,False,True,True])
    leg_factors = np.array([+1,-1,-1,+1])
    T_steps = np.maximum(maturity - np.arange(nb_steps)*dt,0.0)
    S_val = np.repeat(price_paths[:,:,None],4,axis=2)
    K_val = strikes[None,None,:]
    T_val = np.repeat(T_steps[None,:], n_sim, axis=0)
    T_val = np.repeat(T_val[:,:,None],4,axis=2)

    price_all = GreeksCalculator.compute_price_vectorized(S_val, K_val, r, sigma, T_val, is_call)*leg_factors
    position_values = price_all.sum(axis=2)
    pnl_all = position_values - initial_cost

    if initial_cost != 0:
        number_of_contracts = int(investment_amount // abs(initial_cost))
        if number_of_contracts < 1:
            number_of_contracts = 1
    else:
        st.warning("Initial cost is zero, cannot scale P&L.")
        number_of_contracts = 1

    scale_factor = number_of_contracts
    pnl_all *= scale_factor
    max_profit_scaled = max_profit * scale_factor
    max_loss_scaled = max_loss * scale_factor

    take_profit_scaled = take_profit_input * scale_factor
    stop_loss_scaled = stop_loss_input * scale_factor

    exit_indices = np.full(n_sim, np.nan)
    final_pnls = np.full(n_sim, np.nan)
    exit_types = np.full(n_sim, "", dtype=object)
    exit_times_days = np.full(n_sim, np.nan)

    for i in range(n_sim):
        exit_idx, tp_hit = find_exit_time_retro_for_one(
            pnl_all[i,:], take_profit_scaled, stop_loss_scaled, buffer_steps, max_profit_scaled, max_loss_scaled
        )
        if exit_idx is not None:
            final_pnls[i] = pnl_all[i, int(exit_idx)]
            exit_indices[i] = exit_idx
            exit_times_days[i] = exit_idx/ (steps_per_day)
            exit_types[i] = "TP" if tp_hit else "SL"
        else:
            final_pnls[i] = pnl_all[i,-1]

    end = time.time()

    st.session_state.price_paths = price_paths
    st.session_state.pnl_all = pnl_all
    st.session_state.exit_indices = exit_indices
    st.session_state.final_pnls = final_pnls
    st.session_state.exit_types = exit_types
    st.session_state.exit_times_days = exit_times_days
    st.session_state.max_profit = max_profit_scaled
    st.session_state.max_loss = max_loss_scaled
    st.session_state.final_pnl_at_expiry = final_pnl_at_expiry*scale_factor
    st.session_state.S_range = S_range
    st.session_state.break_evens = break_evens
    st.session_state.n_sim = n_sim
    st.session_state.lower_percentile = lower_percentile
    st.session_state.upper_percentile = upper_percentile
    st.session_state.alpha = alpha
    st.session_state.threshold_for_prob = threshold_for_prob
    st.session_state.take_profit_scaled = take_profit_scaled
    st.session_state.stop_loss_scaled = stop_loss_scaled
    st.session_state.sim_index = 0

    st.success(f"Simulations completed in {end - start:.2f} seconds.")
    st.write(f"Number of Contracts: {number_of_contracts}")
    st.write(f"You sold {number_of_contracts} contracts for the short put and short call legs.")
    st.write(f"You bought {number_of_contracts} contracts for the long put and long call legs.")

if 'final_pnls' in st.session_state:
    final_pnls = st.session_state.final_pnls
    exit_types = st.session_state.exit_types
    exit_indices = st.session_state.exit_indices
    exit_times_days = st.session_state.exit_times_days
    pnl_all = st.session_state.pnl_all
    price_paths = st.session_state.price_paths
    max_profit = st.session_state.max_profit
    max_loss = st.session_state.max_loss
    final_pnl_at_expiry = st.session_state.final_pnl_at_expiry
    S_range = st.session_state.S_range
    break_evens = st.session_state.break_evens
    n_sim = st.session_state.n_sim
    lower_percentile = st.session_state.lower_percentile
    upper_percentile = st.session_state.upper_percentile
    alpha = st.session_state.alpha
    threshold_for_prob = st.session_state.threshold_for_prob
    take_profit_scaled = st.session_state.take_profit_scaled
    stop_loss_scaled = st.session_state.stop_loss_scaled

    fig_expiry = go.Figure()
    fig_expiry.add_trace(go.Scatter(x=S_range,y=final_pnl_at_expiry, mode='lines', name='P&L Expiry'))
    fig_expiry.add_trace(go.Scatter(x=[S_range[0], S_range[-1]],
                                    y=[max_profit, max_profit],
                                    mode='lines',
                                    name=f"Max Profit={max_profit:.2f}",
                                    line=dict(color='green', dash='dash')))
    fig_expiry.add_trace(go.Scatter(x=[S_range[0], S_range[-1]],
                                    y=[max_loss, max_loss],
                                    mode='lines',
                                    name=f"Max Loss={max_loss:.2f}",
                                    line=dict(color='red', dash='dash')))

    for be in break_evens:
        fig_expiry.add_vline(x=be, line=dict(color='gray', dash='dot'), annotation_text=f"BE={be:.2f}")
    fig_expiry.update_layout(title="Payoff at Expiration (Scaled)", xaxis_title="Underlying Price", yaxis_title="P&L (Scaled)")
    st.plotly_chart(fig_expiry, use_container_width=True)

    tp_count = np.sum(exit_types=="TP")
    sl_count = np.sum(exit_types=="SL")
    no_exit_count = np.sum(exit_types=="")
    var, es = compute_var_es(final_pnls, alpha)
    prob_hit = probability_of_threshold_hit(final_pnls, threshold_for_prob)

    metrics_data = [
        ["Number of simulations", f"{n_sim}"],
        [f"TP triggered", f"{tp_count} ({tp_count/n_sim*100:.2f}%)"],
        [f"SL triggered", f"{sl_count} ({sl_count/n_sim*100:.2f}%)"],
        [f"No exit", f"{no_exit_count} ({no_exit_count/n_sim*100:.2f}%)"],
        [f"VaR at alpha={alpha}", f"{var:.2f}"],
        ["ES", f"{es:.2f}"],
        [f"Probability of final P&L ≥ {threshold_for_prob:.2f}", f"{prob_hit*100:.2f}%"]
    ]
    metrics_df = pd.DataFrame(metrics_data, columns=["Metric", "Value"])
    st.table(metrics_df)

    fig_pl_hist = go.Figure(data=[go.Histogram(x=final_pnls, nbinsx=50)])
    fig_pl_hist.update_layout(title="Final P&L Distribution (Scaled)", xaxis_title="P&L (Scaled)", yaxis_title="Frequency")
    st.plotly_chart(fig_pl_hist, use_container_width=True)

    tp_exits = exit_times_days[(~np.isnan(exit_times_days)) & (exit_types=="TP")]
    sl_exits = exit_times_days[(~np.isnan(exit_times_days)) & (exit_types=="SL")]

    col_tp, col_sl = st.columns(2)
    if len(tp_exits)>0:
        fig_exit_tp = go.Figure(data=[go.Histogram(x=tp_exits, nbinsx=50)])
        fig_exit_tp.update_layout(title="Exit Time Distribution (Days) - TP", xaxis_title="Time (days)", yaxis_title="Frequency")
        col_tp.plotly_chart(fig_exit_tp, use_container_width=True)
    else:
        col_tp.write("No TP exits.")

    if len(sl_exits)>0:
        fig_exit_sl = go.Figure(data=[go.Histogram(x=sl_exits, nbinsx=50)])
        fig_exit_sl.update_layout(title="Exit Time Distribution (Days) - SL", xaxis_title="Time (days)", yaxis_title="Frequency")
        col_sl.plotly_chart(fig_exit_sl, use_container_width=True)
    else:
        col_sl.write("No SL exits.")

    sim_index_select = st.number_input("Select Simulation Index", min_value=1, max_value=n_sim, value=1, step=1)
    st.session_state.sim_index = sim_index_select - 1
    current_i = st.session_state.sim_index
    st.write(f"Showing simulation {current_i+1} of {n_sim}")

    fig_single = go.Figure()
    fig_single.add_trace(go.Scatter(x=np.arange(pnl_all.shape[1]), y=price_paths[current_i],
                                    mode='lines', name='Price', line=dict(color='orange'), yaxis='y2'))
    fig_single.add_trace(go.Scatter(x=np.arange(pnl_all.shape[1]), y=pnl_all[current_i],
                                    mode='lines', name='P&L (Scaled)', line=dict(color='blue')))

    if not np.isnan(exit_indices[current_i]):
        x_exit = exit_indices[current_i]
        y_exit = pnl_all[current_i,int(x_exit)]
        tp_hit = (exit_types[current_i]=="TP")
        marker_symbol = "triangle-up" if tp_hit else "triangle-down"
        marker_color = "green" if tp_hit else "red"
        fig_single.add_trace(go.Scatter(
            x=[x_exit],
            y=[y_exit],
            mode='markers',
            marker=dict(symbol=marker_symbol, size=12, color=marker_color),
            name="Exit"
        ))

    fig_single.add_trace(go.Scatter(x=[0, pnl_all.shape[1]-1],
                                    y=[take_profit_scaled, take_profit_scaled],
                                    mode='lines',
                                    name=f"TP={take_profit_scaled:.2f}",
                                    line=dict(color='green', dash='dot')))
    fig_single.add_trace(go.Scatter(x=[0, pnl_all.shape[1]-1],
                                    y=[stop_loss_scaled, stop_loss_scaled],
                                    mode='lines',
                                    name=f"SL={stop_loss_scaled:.2f}",
                                    line=dict(color='red', dash='dot')))
    fig_single.add_trace(go.Scatter(x=[0, pnl_all.shape[1]-1],
                                    y=[max_profit, max_profit],
                                    mode='lines',
                                    name=f"Max Profit={max_profit:.2f}",
                                    line=dict(color='green', dash='dash')))
    fig_single.add_trace(go.Scatter(x=[0, pnl_all.shape[1]-1],
                                    y=[max_loss, max_loss],
                                    mode='lines',
                                    name=f"Max Loss={max_loss:.2f}",
                                    line=dict(color='red', dash='dash')))

    fig_single.update_layout(
        title=f"Simulation {current_i+1}",
        xaxis_title="Time step",
        yaxis_title="P&L (Scaled)",
        yaxis2=dict(title="Underlying", overlaying='y', side='right')
    )
    st.plotly_chart(fig_single, use_container_width=True)

    low_p, mean_p, high_p = compute_percentiles(price_paths, lower=lower_percentile, upper=upper_percentile)
    low_pnl, mean_pnl, high_pnl = compute_percentiles(pnl_all, lower=lower_percentile, upper=upper_percentile)

    st.subheader("Aggregated Mean Paths and Percentile Bands")

    st.write("**Aggregated Underlying Paths (Price)**")
    fig_agg_price = go.Figure()
    fig_agg_price.add_trace(go.Scatter(x=np.arange(pnl_all.shape[1]), y=high_p, line=dict(color='orange',width=1), name=f'Price {upper_percentile}th', fill=None))
    fig_agg_price.add_trace(go.Scatter(x=np.arange(pnl_all.shape[1]), y=low_p, line=dict(color='orange',width=1), name=f'Price {lower_percentile}th', fill='tonexty', fillcolor='rgba(255,165,0,0.2)'))
    fig_agg_price.add_trace(go.Scatter(x=np.arange(pnl_all.shape[1]), y=mean_p, line=dict(color='orange',width=3), name='Price Mean'))
    fig_agg_price.update_layout(
        title=f"Aggregated Underlying Price Paths (Mean and {lower_percentile}th/{upper_percentile}th Percentiles)",
        xaxis_title="Time step",
        yaxis_title="Price"
    )
    st.plotly_chart(fig_agg_price, use_container_width=True)

    st.write("**Aggregated P&L Paths**")
    fig_agg_pnl = go.Figure()
    fig_agg_pnl.add_trace(go.Scatter(x=np.arange(pnl_all.shape[1]), y=high_pnl, line=dict(color='blue',width=1), name=f'P&L {upper_percentile}th', fill=None))
    fig_agg_pnl.add_trace(go.Scatter(x=np.arange(pnl_all.shape[1]), y=low_pnl, line=dict(color='blue',width=1), name=f'P&L {lower_percentile}th', fill='tonexty', fillcolor='rgba(0,0,255,0.2)'))
    fig_agg_pnl.add_trace(go.Scatter(x=np.arange(pnl_all.shape[1]), y=mean_pnl, line=dict(color='blue',width=3), name='P&L Mean'))

    fig_agg_pnl.add_trace(go.Scatter(x=[0, pnl_all.shape[1]-1],
                                    y=[max_profit, max_profit],
                                    mode='lines',
                                    name=f"Max Profit={max_profit:.2f}",
                                    line=dict(color='green', dash='dash')))
    fig_agg_pnl.add_trace(go.Scatter(x=[0, pnl_all.shape[1]-1],
                                    y=[max_loss, max_loss],
                                    mode='lines',
                                    name=f"Max Loss={max_loss:.2f}",
                                    line=dict(color='red', dash='dash')))

    fig_agg_pnl.update_layout(
        title=f"Aggregated P&L Paths (Mean and {lower_percentile}th/{upper_percentile}th Percentiles)",
        xaxis_title="Time step",
        yaxis_title="P&L (Scaled)"
    )
    st.plotly_chart(fig_agg_pnl, use_container_width=True)

    df_res = pd.DataFrame({
        "sim_id": np.arange(n_sim),
        "final_pnl_scaled": final_pnls,
        "exit_step": exit_indices,
        "exit_time_days": exit_times_days,
        "exit_type": exit_types
    })
    st.download_button("Download Monte Carlo Results", df_res.to_csv(index=False).encode('utf-8'),
                       "montecarlo_results.csv")
