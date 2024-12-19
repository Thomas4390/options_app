import streamlit as st
import numpy as np
import pandas as pd
from dataclasses import dataclass
from scipy.stats import norm
import plotly.graph_objects as go
import time
import math

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
        Initialize the GBM simulator with given parameters.
        """
        self.params = params
        np.random.seed(self.params.seed)

    def simulate(self, nb_steps: int) -> np.ndarray:
        """
        Simulate a single path using a Geometric Brownian Motion (GBM) model.

        Parameters
        ----------
        nb_steps : int
            Number of time steps to simulate.

        Returns
        -------
        np.ndarray
            Array of simulated prices at each time step.
        """
        s0 = self.params.spot
        mu = self.params.mu
        sigma = self.params.sigma
        dt = self.params.dt
        eps = np.random.randn(nb_steps)
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * eps
        log_prices = np.log(s0) + np.cumsum(drift + diffusion)
        return np.exp(log_prices)

class GreeksCalculator:
    @staticmethod
    def compute_full_greeks(
        S: np.ndarray,
        K: np.ndarray,
        r: float,
        sigma: float,
        T: np.ndarray,
        is_call: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """
        Compute option price and Greeks (Delta, Gamma, Vega, Theta, Rho) using Black-Scholes.

        Parameters
        ----------
        S : np.ndarray
            Current underlying prices.
        K : np.ndarray
            Strike prices.
        r : float
            Risk-free interest rate.
        sigma : float
            Volatility of the underlying.
        T : np.ndarray
            Time-to-maturity for each option.
        is_call : np.ndarray
            Boolean array indicating if the option is call (True) or put (False).

        Returns
        -------
        tuple
            price, delta, gamma, vega, theta, rho as np.ndarrays.
        """
        safe_T = np.where(T > 0, T, np.nan)
        d1 = (np.log(S/K) + (r + 0.5 * sigma**2) * safe_T) / (sigma * np.sqrt(safe_T))
        d2 = d1 - sigma * np.sqrt(safe_T)
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)

        call_price = S * cdf_d1 - K * np.exp(-r*T)*cdf_d2
        put_price = K * np.exp(-r*T)*norm.cdf(-d2) - S * norm.cdf(-d1)
        payoff_call = np.maximum(S-K, 0)
        payoff_put = np.maximum(K-S, 0)

        price = np.where(T > 0, np.where(is_call, call_price, put_price),
                         np.where(is_call, payoff_call, payoff_put))

        delta = np.where(T > 0, np.where(is_call, cdf_d1, cdf_d1 - 1),
                         np.where(is_call, (S > K).astype(float), (-1)*(S < K).astype(float)))
        gamma = np.where(T > 0, pdf_d1/(S*sigma*np.sqrt(T)), 0.0)
        vega = np.where(T > 0, S*pdf_d1*np.sqrt(T), 0.0)
        with np.errstate(invalid='ignore'):
            first_term = -(S*pdf_d1*sigma)/(2*np.sqrt(T))
            call_theta = first_term - r*K*np.exp(-r*T)*cdf_d2
            put_theta = first_term + r*K*np.exp(-r*T)*norm.cdf(-d2)
        theta = np.where(T > 0, np.where(is_call, call_theta, put_theta), 0.0)
        rho = np.where(T > 0,
                       np.where(is_call, K*T*np.exp(-r*T)*cdf_d2, -K*T*np.exp(-r*T)*norm.cdf(-d2)),
                       0.0)

        return price, delta, gamma, vega, theta, rho

def compute_iron_condor_metrics(
    K1: float, K2: float, K3: float, K4: float,
    spot: float, r: float, sigma: float, maturity: float
) -> tuple[float, np.ndarray, np.ndarray, float, float, list]:
    """
    Compute initial cost, final P&L at expiry, strikes range, max profit, max loss,
    and break-even points for an Iron Condor.
    """
    strikes = np.array([K1, K2, K3, K4])
    is_call = np.array([False, False, True, True])
    leg_factors = np.array([+1, -1, -1, +1])
    T0 = np.full_like(strikes, maturity)

    price0, _, _, _, _, _ = GreeksCalculator.compute_full_greeks(
        S=np.full_like(strikes, spot), K=strikes, r=r, sigma=sigma, T=T0, is_call=is_call
    )
    initial_cost = np.sum(price0 * leg_factors)

    S_range = np.linspace(K1 - 20, K4 + 20, 300)
    payoff_long_put = np.maximum(K1 - S_range, 0)
    payoff_short_put = -np.maximum(K2 - S_range, 0)
    payoff_short_call = -np.maximum(S_range - K3, 0)
    payoff_long_call = np.maximum(S_range - K4, 0)
    final_intrinsic = payoff_long_put + payoff_short_put + payoff_short_call + payoff_long_call
    final_pnl_at_expiry = final_intrinsic - initial_cost
    max_profit = np.max(final_pnl_at_expiry)
    max_loss = np.min(final_pnl_at_expiry)

    sign_change_indices = np.where(np.diff(np.sign(final_pnl_at_expiry)) != 0)[0]
    break_evens = []
    for idx in sign_change_indices:
        x0, x1 = S_range[idx], S_range[idx+1]
        y0, y1 = final_pnl_at_expiry[idx], final_pnl_at_expiry[idx+1]
        be = x0 - y0*(x1 - x0)/(y1 - y0)
        break_evens.append(be)

    return initial_cost, final_pnl_at_expiry, S_range, max_profit, max_loss, break_evens

def find_exit_time_retro(
    pnl: np.ndarray,
    tp: float,
    sl: float,
    buff: int,
    max_profit: float,
    max_loss: float
) -> tuple[float | None, bool | None]:
    """
    Find retroactive exit time based on TP/SL conditions with a buffer.

    Parameters
    ----------
    pnl : np.ndarray
        P&L array over time steps.
    tp : float
        Take Profit threshold.
    sl : float
        Stop Loss threshold.
    buff : int
        Buffer steps before confirming the exit.
    max_profit : float
        Maximum profit possible.
    max_loss : float
        Maximum loss possible.

    Returns
    -------
    tuple
        exit_idx (or None), tp_hit (True if TP triggered, False if SL triggered, None if no exit)
    """
    nb_steps = len(pnl)
    inside_range_indices = np.where((pnl > max_loss) & (pnl < max_profit))[0]
    if len(inside_range_indices) == 0:
        return None, None
    start_monitoring_step = inside_range_indices[0]
    for t in range(start_monitoring_step, nb_steps):
        if pnl[t] >= tp or pnl[t] <= sl:
            exit_idx = t + buff
            if exit_idx < nb_steps:
                tp_hit = (pnl[t] >= tp)
                return exit_idx, tp_hit
            else:
                return None, None
    return None, None

def compute_break_evens_at_step(
    S_min: float, S_max: float,
    strikes: np.ndarray,
    is_call: np.ndarray,
    leg_factors: np.ndarray,
    initial_cost: float,
    r: float,
    sigma: float,
    T_current: float
) -> tuple[float, float]:
    """
    Compute instantaneous break-even points by scanning a price range.
    """
    S_test = np.linspace(S_min, S_max, 100)
    T_grid = np.full((100,4), T_current)
    K_grid = np.repeat(strikes[None, :], 100, axis=0)

    price_options, _, _, _, _, _ = GreeksCalculator.compute_full_greeks(
        S=S_test[:, None].repeat(4, axis=1),
        K=K_grid,
        r=r,
        sigma=sigma,
        T=T_grid,
        is_call=is_call
    )
    price_position = (price_options * leg_factors).sum(axis=1) - initial_cost
    sign_changes = np.where(np.diff(np.sign(price_position)) != 0)[0]
    be_points = []
    for idx in sign_changes:
        x0, x1 = S_test[idx], S_test[idx+1]
        y0, y1 = price_position[idx], price_position[idx+1]
        be = x0 - y0*(x1 - x0)/(y1 - y0)
        be_points.append(be)
    if len(be_points) == 2:
        be_points.sort()
        return be_points[0], be_points[1]
    elif len(be_points) == 1:
        return be_points[0], np.nan
    else:
        return np.nan, np.nan

st.title("Single Path Simulation (Iron Condor) - Enhanced Greeks and Payoff")

st.markdown("""
**How is the Initial Cost Calculated?**

The initial cost of the Iron Condor is computed by pricing each of the four legs at the start and summing them 
with their respective signs (+/-). If negative, you receive a credit; if positive, you pay a debit.
""")

st.sidebar.header("Iron Condor Parameters")
K1 = st.sidebar.number_input("Long Put (K1)", value=90.0)
K2 = st.sidebar.number_input("Short Put (K2)", value=95.0)
K3 = st.sidebar.number_input("Short Call (K3)", value=105.0)
K4 = st.sidebar.number_input("Long Call (K4)", value=110.0)
if not (K1 < K2 < K3 < K4):
    st.error("Invalid Iron Condor configuration")

st.sidebar.header("Market and Simulation Parameters")
maturity_days = st.sidebar.number_input("Maturity (Days)", value=30)
maturity = maturity_days / 365
spot = st.sidebar.number_input("Initial Spot", value=100.0)
mu = st.sidebar.number_input("Mu (Annual Drift)", value=0.05)
sigma = st.sidebar.number_input("Sigma (Annual Vol)", value=0.2)
r = st.sidebar.number_input("Interest Rate (r)", value=0.01)
seed = st.sidebar.number_input("Seed", value=42)

interval = st.sidebar.selectbox("Time Interval", ["Hours","Minutes","Seconds"], index=0)
if interval == "Seconds":
    steps_per_day = 24*3600
elif interval == "Minutes":
    steps_per_day = 24*60
else:
    steps_per_day = 24

nb_days = st.sidebar.number_input("Number of Days", value=30)
nb_steps = int(nb_days * steps_per_day)
dt = (1 / 365) / steps_per_day

st.sidebar.header("Investment")
investment_amount = st.sidebar.number_input("Investment Amount", value=10000.0, min_value=0.0)

# Only Absolute TP/SL
initial_cost, final_pnl_at_expiry, S_range, max_profit, max_loss, break_evens = compute_iron_condor_metrics(
    K1, K2, K3, K4, spot, r, sigma, maturity
)

st.sidebar.header("Stop Loss / Take Profit (Absolute)")
take_profit_input = st.sidebar.number_input("Take Profit (TP)", value=float(max_profit))
stop_loss_input = st.sidebar.number_input("Stop Loss (SL)", value=float(max_loss))
buffer_steps = st.sidebar.number_input("Buffer (time steps)", value=1, min_value=0)

st.subheader("Max Profit / Max Loss Decomposition")
decomp_data = [
    ["Initial Cost", f"{initial_cost:.2f}"],
    ["Max Profit", f"{max_profit:.2f}"],
    ["Max Loss", f"{max_loss:.2f}"],
    ["Break Evens", f"{break_evens}"]
]
decomp_df = pd.DataFrame(decomp_data, columns=["Metric", "Value"])
st.table(decomp_df)

if st.button("Run Simulation"):
    start = time.time()
    gbm_params = GBMParameters(spot=spot, mu=mu, sigma=sigma, dt=dt, seed=seed)
    simulator = GBMSimulator(params=gbm_params)
    prices = simulator.simulate(nb_steps)
    times = np.arange(nb_steps)

    strikes = np.array([K1,K2,K3,K4])
    is_call = np.array([False,False,True,True])
    leg_factors = np.array([+1,-1,-1,+1])
    T_arr_all = np.maximum(maturity - times*dt, 0.0)

    price_all = np.zeros((4, nb_steps))
    delta_all = np.zeros((4, nb_steps))
    gamma_all = np.zeros((4, nb_steps))
    vega_all = np.zeros((4, nb_steps))
    theta_all = np.zeros((4, nb_steps))
    rho_all = np.zeros((4, nb_steps))

    for t in range(nb_steps):
        S_t = prices[t]
        T_arr = np.full(4, T_arr_all[t])
        p, dlt, gma, vga, tht, ro = GreeksCalculator.compute_full_greeks(
            S=np.full(4, S_t), K=strikes, r=r, sigma=sigma, T=T_arr, is_call=is_call
        )
        p *= leg_factors
        dlt *= leg_factors
        gma *= leg_factors
        vga *= leg_factors
        tht *= leg_factors
        ro *= leg_factors

        price_all[:, t] = p
        delta_all[:, t] = dlt
        gamma_all[:, t] = gma
        vega_all[:, t] = vga
        theta_all[:, t] = tht
        rho_all[:, t] = ro

    total_pnl_steps = price_all.sum(axis=0) - initial_cost

    if initial_cost != 0:
        number_of_contracts = int(investment_amount // abs(initial_cost))
        if number_of_contracts < 1:
            number_of_contracts = 1
    else:
        st.warning("Initial cost is zero, cannot scale the P&L by investment amount.")
        number_of_contracts = 1

    scale_factor = number_of_contracts
    total_pnl_steps *= scale_factor
    max_profit_scaled = max_profit * scale_factor
    max_loss_scaled = max_loss * scale_factor

    take_profit_scaled = take_profit_input * scale_factor
    stop_loss_scaled = stop_loss_input * scale_factor

    exit_idx, tp_hit = find_exit_time_retro(
        total_pnl_steps, take_profit_scaled, stop_loss_scaled, buffer_steps, max_profit_scaled, max_loss_scaled
    )
    final_pnl = total_pnl_steps[-1]
    exit_time_days = None
    if exit_idx is not None:
        final_pnl = total_pnl_steps[exit_idx]
        exit_time_days = exit_idx / steps_per_day

    S_min = K1 - 20
    S_max = K4 + 20
    be_lower_dyn = np.full(nb_steps, np.nan)
    be_upper_dyn = np.full(nb_steps, np.nan)
    for t in range(nb_steps):
        T_current = T_arr_all[t]
        if T_current > 0:
            bl, bu = compute_break_evens_at_step(
                S_min, S_max, strikes, is_call, leg_factors, initial_cost, r, sigma, T_current
            )
            be_lower_dyn[t] = bl
            be_upper_dyn[t] = bu
        else:
            if len(break_evens) > 0:
                be_lower_dyn[t] = break_evens[0]
            if len(break_evens) > 1:
                be_upper_dyn[t] = break_evens[1]

    end = time.time()
    st.success(f"Simulation completed in {end - start:.2f} seconds.")
    st.write(f"Number of Contracts: {number_of_contracts}")
    st.write(f"You sold {number_of_contracts} contracts for the short put and short call legs.")
    st.write(f"You bought {number_of_contracts} contracts for the long put and long call legs.")

    final_pnl_at_expiry_scaled = final_pnl_at_expiry * scale_factor
    fig_expiry = go.Figure()
    fig_expiry.add_trace(go.Scatter(x=S_range, y=final_pnl_at_expiry_scaled, mode='lines', name='P&L Expiry (Scaled)'))
    fig_expiry.add_trace(go.Scatter(x=[S_range[0], S_range[-1]],
                                    y=[max_profit_scaled, max_profit_scaled],
                                    mode='lines',
                                    name=f"Max Profit={max_profit_scaled:.2f}",
                                    line=dict(color='green', dash='dash')))
    fig_expiry.add_trace(go.Scatter(x=[S_range[0], S_range[-1]],
                                    y=[max_loss_scaled, max_loss_scaled],
                                    mode='lines',
                                    name=f"Max Loss={max_loss_scaled:.2f}",
                                    line=dict(color='red', dash='dash')))
    for be in break_evens:
        fig_expiry.add_vline(x=be, line=dict(color='gray', dash='dot'),
                             annotation_text=f"BE={be:.2f}", annotation_position="top left")
    fig_expiry.update_layout(title="Payoff at Expiration (Scaled)", xaxis_title="Underlying Price", yaxis_title="P&L (Scaled)")
    st.plotly_chart(fig_expiry, use_container_width=True)

    fig_main = go.Figure()
    fig_main.add_trace(go.Scatter(x=times, y=total_pnl_steps, mode='lines', name='P&L (Scaled)', line=dict(color='blue')))
    fig_main.add_trace(go.Scatter(x=times, y=prices, mode='lines', name='Underlying', line=dict(color='orange'), yaxis='y2'))
    fig_main.add_trace(go.Scatter(x=[times[0], times[-1]],
                                  y=[take_profit_scaled, take_profit_scaled],
                                  mode='lines',
                                  name=f"TP={take_profit_scaled:.2f}",
                                  line=dict(color='green', dash='dot')))
    fig_main.add_trace(go.Scatter(x=[times[0], times[-1]],
                                  y=[stop_loss_scaled, stop_loss_scaled],
                                  mode='lines',
                                  name=f"SL={stop_loss_scaled:.2f}",
                                  line=dict(color='red', dash='dot')))
    fig_main.add_trace(go.Scatter(x=[times[0], times[-1]],
                                  y=[max_profit_scaled, max_profit_scaled],
                                  mode='lines',
                                  name=f"Max Profit={max_profit_scaled:.2f}",
                                  line=dict(color='green', dash='dash')))
    fig_main.add_trace(go.Scatter(x=[times[0], times[-1]],
                                  y=[max_loss_scaled, max_loss_scaled],
                                  mode='lines',
                                  name=f"Max Loss={max_loss_scaled:.2f}",
                                  line=dict(color='red', dash='dash')))
    fig_main.add_trace(go.Scatter(x=times, y=be_lower_dyn, mode='lines', name='BE lower dyn', line=dict(color='gray', dash='dot'), yaxis='y2'))
    fig_main.add_trace(go.Scatter(x=times, y=be_upper_dyn, mode='lines', name='BE upper dyn', line=dict(color='gray', dash='dot'), yaxis='y2'))

    if exit_idx is not None:
        marker_symbol = "triangle-up" if tp_hit else "triangle-down"
        marker_color = "green" if tp_hit else "red"
        fig_main.add_trace(go.Scatter(
            x=[exit_idx],
            y=[final_pnl],
            mode='markers',
            marker=dict(symbol=marker_symbol, size=12, color=marker_color),
            name="Exit"
        ))

    fig_main.update_layout(
        title="P&L, Underlying and Dynamic Break-Evens (Scaled)",
        xaxis_title="Time step",
        yaxis_title="P&L (Scaled)",
        yaxis2=dict(title="Underlying / Dynamic BE", overlaying='y', side='right')
    )
    st.plotly_chart(fig_main, use_container_width=True)

    st.write(f"Final P&L (Scaled): {final_pnl:.2f}")
    if exit_time_days is not None:
        st.write(f"Position closed after {exit_time_days:.2f} days via {'Take Profit' if tp_hit else 'Stop Loss'}.")
    else:
        st.write("No early exit triggered.")

    greek_tabs = st.tabs(["Delta","Gamma","Vega","Theta","Rho"])
    with greek_tabs[0]:
        fig_delta = go.Figure(go.Scatter(x=times, y=delta_all.sum(axis=0), mode='lines', name='Delta'))
        fig_delta.update_layout(title="Position Delta Over Time", xaxis_title="Time step", yaxis_title="Delta")
        st.plotly_chart(fig_delta, use_container_width=True)
    with greek_tabs[1]:
        fig_gamma = go.Figure(go.Scatter(x=times, y=gamma_all.sum(axis=0), mode='lines', name='Gamma'))
        fig_gamma.update_layout(title="Position Gamma Over Time", xaxis_title="Time step", yaxis_title="Gamma")
        st.plotly_chart(fig_gamma, use_container_width=True)
    with greek_tabs[2]:
        fig_vega = go.Figure(go.Scatter(x=times, y=vega_all.sum(axis=0), mode='lines', name='Vega'))
        fig_vega.update_layout(title="Position Vega Over Time", xaxis_title="Time step", yaxis_title="Vega")
        st.plotly_chart(fig_vega, use_container_width=True)
    with greek_tabs[3]:
        fig_theta = go.Figure(go.Scatter(x=times, y=theta_all.sum(axis=0), mode='lines', name='Theta'))
        fig_theta.update_layout(title="Position Theta Over Time", xaxis_title="Time step", yaxis_title="Theta")
        st.plotly_chart(fig_theta, use_container_width=True)
    with greek_tabs[4]:
        fig_rho = go.Figure(go.Scatter(x=times, y=rho_all.sum(axis=0), mode='lines', name='Rho'))
        fig_rho.update_layout(title="Position Rho Over Time", xaxis_title="Time step", yaxis_title="Rho")
        st.plotly_chart(fig_rho, use_container_width=True)

    df_res = pd.DataFrame({
        "time_step": times,
        "underlying": prices,
        "total_pnl_scaled": total_pnl_steps,
        "be_lower_dyn": be_lower_dyn,
        "be_upper_dyn": be_upper_dyn,
        "delta": delta_all.sum(axis=0),
        "gamma": gamma_all.sum(axis=0),
        "vega": vega_all.sum(axis=0),
        "theta": theta_all.sum(axis=0),
        "rho": rho_all.sum(axis=0)
    })
    if exit_time_days is not None:
        df_res["exit_step"] = exit_idx
        df_res["exit_time_days"] = exit_time_days
        df_res["exit_type"] = "TP" if tp_hit else "SL"
    st.download_button("Download Results", df_res.to_csv(index=False).encode('utf-8'), "simulation_results.csv")
