import numpy as np
import streamlit as st
from dataclasses import dataclass
from scipy.stats import norm
import plotly.graph_objects as go
import pandas as pd
import time

@dataclass
class GBMParameters:
    """
    Parameters for the Geometric Brownian Motion model.
    """
    spot: float
    mu: float
    sigma: float
    dt: float
    seed: int = 42

class GBMSimulator:
    def __init__(self, params: GBMParameters):
        self.params = params
        np.random.seed(self.params.seed)

    def simulate(self, nb_steps: int) -> np.ndarray:
        s0 = self.params.spot
        mu = self.params.mu
        sigma = self.params.sigma
        dt = self.params.dt

        eps = np.random.randn(nb_steps)
        drift = (mu - 0.5 * sigma**2) * dt
        diffusion = sigma * np.sqrt(dt) * eps

        log_prices = np.log(s0) + np.cumsum(drift + diffusion)
        prices = np.exp(log_prices)
        return prices

    @staticmethod
    def plot_prices(times: np.ndarray, prices: np.ndarray) -> go.Figure:
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=times, y=prices, mode='lines', name='Underlying Asset'))
        fig.update_layout(
            title="Simulated Underlying Asset Trajectory (GBM)",
            xaxis_title="Time (custom interval)",
            yaxis_title="Price",
            legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
        )
        fig.update_xaxes(showgrid=True)
        fig.update_yaxes(showgrid=True)
        return fig

@dataclass
class OptionContract:
    strike: float
    maturity: float
    is_call: bool

    def get_label(self) -> str:
        option_type = "Call" if self.is_call else "Put"
        return f"{option_type}, Mat={self.maturity:.6f}y, Strike={self.strike:.2f}"

class GreeksCalculator:
    @staticmethod
    def compute_d1_d2(S: np.ndarray, K: np.ndarray, r: float, sigma: float, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        safe_T = np.where(T > 0, T, np.nan)
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*safe_T) / (sigma*np.sqrt(safe_T))
        d2 = d1 - sigma*np.sqrt(safe_T)
        d1 = np.where(T > 0, d1, np.nan)
        d2 = np.where(T > 0, d2, np.nan)
        return d1, d2

    @staticmethod
    def compute_greeks(S: np.ndarray, K: np.ndarray, r: float, sigma: float, T: np.ndarray, is_call: np.ndarray) -> tuple:
        payoff = np.maximum((S - K), 0)
        payoff_put = np.maximum((K - S), 0)

        d1, d2 = GreeksCalculator.compute_d1_d2(S=S, K=K, r=r, sigma=sigma, T=T)

        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)

        call_price = S * cdf_d1 - K * np.exp(-r * T) * cdf_d2
        put_price = K * np.exp(-r * T) * norm.cdf(-d2) - S * norm.cdf(-d1)
        price = np.where(T > 0, np.where(is_call, call_price, put_price), np.where(is_call, payoff, payoff_put))

        delta = np.where(T > 0, np.where(is_call, cdf_d1, cdf_d1 - 1),
                         np.where(is_call, (S > K).astype(float), (-1)*(S < K).astype(float)))
        gamma = np.where(T > 0, pdf_d1 / (S * sigma * np.sqrt(T)), 0.0)
        vega = np.where(T > 0, S * pdf_d1 * np.sqrt(T), 0.0)

        with np.errstate(invalid='ignore'):
            first_term = -(S * pdf_d1 * sigma) / (2 * np.sqrt(T))
            call_theta = first_term - r * K * np.exp(-r * T) * cdf_d2
            put_theta = first_term + r * K * np.exp(-r * T) * norm.cdf(-d2)
        theta = np.where(T > 0, np.where(is_call, call_theta, put_theta), 0.0)

        rho = np.where(T > 0, np.where(is_call, K * T * np.exp(-r * T) * cdf_d2,
                                       -K * T * np.exp(-r * T) * norm.cdf(-d2)), 0.0)

        return price, delta, gamma, vega, theta, rho

def plot_all_options_prices(times: np.ndarray, options: list[OptionContract], option_prices_list: list[np.ndarray]) -> go.Figure:
    fig = go.Figure()
    for opt, opt_prices in zip(options, option_prices_list):
        fig.add_trace(go.Scatter(x=times, y=opt_prices, mode='lines', name=opt.get_label()))
    fig.update_layout(
        title="Option Prices Over Time",
        xaxis_title="Time (custom interval)",
        yaxis_title="Option Price",
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig

def plot_all_options_greek(times: np.ndarray, options: list[OptionContract], greek_values_list: list[np.ndarray], greek_name: str) -> go.Figure:
    fig = go.Figure()
    for opt, gv in zip(options, greek_values_list):
        fig.add_trace(go.Scatter(x=times, y=gv, mode='lines', name=opt.get_label()))
    fig.update_layout(
        title=f"Evolution of {greek_name} Over Time",
        xaxis_title="Time (custom interval)",
        yaxis_title=greek_name,
        legend=dict(yanchor="top", y=0.99, xanchor="left", x=0.01)
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig

def main():
    # Initialisation des variables de session
    if 'simulation_done' not in st.session_state:
        st.session_state.simulation_done = False
    if 'df' not in st.session_state:
        st.session_state.df = None
    if 'times' not in st.session_state:
        st.session_state.times = None
    if 'prices' not in st.session_state:
        st.session_state.prices = None
    if 'options' not in st.session_state:
        st.session_state.options = []
    if 'elapsed' not in st.session_state:
        st.session_state.elapsed = 0.0
    if 'params' not in st.session_state:
        st.session_state.params = {}

    # Gestion des changements de paramètres pour réinitialiser la simulation
    params_changed = False
    current_params = {
        "spot": st.sidebar.number_input("Initial Spot Price", value=100.0, step=1.0),
        "mu": st.sidebar.number_input("Mu (Annual Drift)", value=0.05, step=0.01),
        "sigma": st.sidebar.number_input("Sigma (Annual Volatility)", value=0.2, step=0.01),
        "nb_days": st.sidebar.number_input("Number of Days to Simulate", value=1, step=1),
        "seed": st.sidebar.number_input("Random Seed", value=42, step=1),
        "time_interval": st.sidebar.selectbox("Choose Time Interval", ["Seconds", "Minutes", "Hours"]),
        "option_count": st.sidebar.number_input("Number of Options", min_value=1, max_value=10, value=4),
        "r": st.sidebar.number_input("Interest Rate (r)", value=0.01, step=0.001)
    }

    if st.session_state.params != current_params:
        st.session_state.simulation_done = False
        st.session_state.df = None
        st.session_state.times = None
        st.session_state.prices = None
        st.session_state.options = []
        st.session_state.elapsed = 0.0
        st.session_state.params = current_params

    # Sidebar pour les paramètres
    st.sidebar.header("Underlying Asset (GBM) Parameters")
    spot = current_params["spot"]
    mu = current_params["mu"]
    sigma = current_params["sigma"]
    nb_days = current_params["nb_days"]
    seed = current_params["seed"]

    time_interval = current_params["time_interval"]
    if time_interval == "Seconds":
        steps_per_day = 24 * 3600
        xaxis_label = "Time (seconds)"
    elif time_interval == "Minutes":
        steps_per_day = 24 * 60
        xaxis_label = "Time (minutes)"
    else:  # Hours
        steps_per_day = 24
        xaxis_label = "Time (hours)"

    nb_steps = int(nb_days * steps_per_day)
    day_in_years = 1 / 365
    dt = day_in_years / steps_per_day

    st.sidebar.header("Options Parameters")
    option_count = int(current_params["option_count"])
    options = []
    for i in range(option_count):
        st.sidebar.subheader(f"Option {i + 1}")
        strike = st.sidebar.number_input(f"Strike (Option {i + 1})", value=100.0 + i * 5, step=1.0, key=f"strike_{i}")
        maturity_days = st.sidebar.number_input(f"Maturity (Days, Option {i + 1})", value=10 * (i + 1), step=1, key=f"maturity_{i}")
        is_call = st.sidebar.selectbox(f"Type (Option {i + 1})", ("Call", "Put"), index=0, key=f"type_{i}") == "Call"
        options.append(OptionContract(strike=strike, maturity=maturity_days / 365, is_call=is_call))

    r = current_params["r"]

    # Mettre à jour les options dans session_state
    st.session_state.options = options

    with st.expander("About this Application"):
        st.write("""
        This application simulates the price of an underlying asset using a Geometric Brownian Motion (GBM) model.
        You can configure the initial spot price, annual drift (mu), annual volatility (sigma),
        and the simulation duration in days. You can also choose the time increment for the simulation.

        Then, you can define multiple European vanilla options with given strike, maturity (in days), and type (Call or Put).
        The application computes their prices and Greeks (Delta, Gamma, Vega, Theta, Rho) over time.

        **Features Implemented:**
        - Interactive plots using Plotly.
        - Tabs to organize results.
        - Configurable time increment (seconds, minutes, hours).
        - Display of simulation parameters and download of results as CSV.
        - Measurement of simulation runtime.
        """)

    st.title("Underlying Asset and Options Simulation (GBM)")

    if st.button("Run Simulation"):
        start_time = time.time()

        gbm_params = GBMParameters(spot=spot, mu=mu, sigma=sigma, dt=dt, seed=seed)
        simulator = GBMSimulator(params=gbm_params)
        prices = simulator.simulate(nb_steps=nb_steps)
        times = np.arange(nb_steps)

        strikes = np.array([opt.strike for opt in options])
        maturities = np.array([opt.maturity for opt in options])
        is_call = np.array([opt.is_call for opt in options])

        option_prices_list = [np.zeros(nb_steps) for _ in options]
        deltas_list = [np.zeros(nb_steps) for _ in options]
        gammas_list = [np.zeros(nb_steps) for _ in options]
        vegas_list = [np.zeros(nb_steps) for _ in options]
        thetas_list = [np.zeros(nb_steps) for _ in options]
        rhos_list = [np.zeros(nb_steps) for _ in options]

        progress_bar = st.progress(0)
        for t in range(nb_steps):
            S = prices[t]
            T_arr = np.maximum(maturities - t * dt, 0.0)
            p, d, g, v, th, rh = GreeksCalculator.compute_greeks(
                S=np.full_like(strikes, S),
                K=strikes,
                r=r,
                sigma=sigma,
                T=T_arr,
                is_call=is_call
            )
            for i in range(len(options)):
                option_prices_list[i][t] = p[i]
                deltas_list[i][t] = d[i]
                gammas_list[i][t] = g[i]
                vegas_list[i][t] = v[i]
                thetas_list[i][t] = th[i]
                rhos_list[i][t] = rh[i]

            # Mise à jour de la barre de progression
            if nb_steps > 100 and t % (nb_steps // 100) == 0:
                progress_bar.progress(int((t / nb_steps) * 100))
            elif nb_steps <= 100:
                progress_bar.progress(int((t / nb_steps) * 100))

        end_time = time.time()
        elapsed = end_time - start_time

        # Construire le DataFrame complet
        df = pd.DataFrame({
            "time_step": times,
            "underlying_price": prices
        })
        for i in range(len(options)):
            df[f"option_{i+1}_price"] = option_prices_list[i]
            df[f"option_{i+1}_delta"] = deltas_list[i]
            df[f"option_{i+1}_gamma"] = gammas_list[i]
            df[f"option_{i+1}_vega"] = vegas_list[i]
            df[f"option_{i+1}_theta"] = thetas_list[i]
            df[f"option_{i+1}_rho"] = rhos_list[i]

        # Stocker les résultats dans session_state
        st.session_state.df = df
        st.session_state.times = times
        st.session_state.prices = prices
        st.session_state.elapsed = elapsed
        st.session_state.simulation_done = True

    # Afficher les résultats si la simulation est terminée
    if st.session_state.simulation_done:
        df = st.session_state.df
        times = st.session_state.times
        prices = st.session_state.prices
        elapsed = st.session_state.elapsed
        options = st.session_state.options

        st.subheader("Simulation Parameters")
        param_dict = {
            "Spot": spot,
            "Mu": mu,
            "Sigma": sigma,
            "Days": nb_days,
            "Interval": time_interval,
            "r": r,
            "Number of Options": len(options)
        }
        st.table(pd.DataFrame([param_dict]))

        # Bouton de téléchargement
        st.download_button(
            label="Download Simulation Results",
            data=df.to_csv(index=False).encode('utf-8'),
            file_name="simulation_results.csv",
            mime="text/csv"
        )

        st.info(f"Simulation completed in {elapsed:.2f} seconds.")

        # Onglets pour les visualisations
        tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8 = st.tabs(
            ["Underlying", "Option Prices", "Delta", "Gamma", "Vega", "Theta", "Rho", "About Options"])

        with tab1:
            fig_prices = GBMSimulator.plot_prices(times=times, prices=prices)
            fig_prices.update_layout(xaxis_title=f"Time ({time_interval.lower()})")
            st.plotly_chart(fig_prices, use_container_width=True)

        with tab2:
            fig_opt_prices = plot_all_options_prices(times=times, options=options,
                                                     option_prices_list=[df[f"option_{i+1}_price"] for i in range(len(options))])
            fig_opt_prices.update_layout(xaxis_title=f"Time ({time_interval.lower()})")
            st.plotly_chart(fig_opt_prices, use_container_width=True)

        with tab3:
            fig_delta = plot_all_options_greek(times=times, options=options,
                                               greek_values_list=[df[f"option_{i+1}_delta"] for i in range(len(options))],
                                               greek_name="Delta")
            fig_delta.update_layout(xaxis_title=f"Time ({time_interval.lower()})")
            st.plotly_chart(fig_delta, use_container_width=True)

        with tab4:
            fig_gamma = plot_all_options_greek(times=times, options=options,
                                               greek_values_list=[df[f"option_{i+1}_gamma"] for i in range(len(options))],
                                               greek_name="Gamma")
            fig_gamma.update_layout(xaxis_title=f"Time ({time_interval.lower()})")
            st.plotly_chart(fig_gamma, use_container_width=True)

        with tab5:
            fig_vega = plot_all_options_greek(times=times, options=options,
                                              greek_values_list=[df[f"option_{i+1}_vega"] for i in range(len(options))],
                                              greek_name="Vega")
            fig_vega.update_layout(xaxis_title=f"Time ({time_interval.lower()})")
            st.plotly_chart(fig_vega, use_container_width=True)

        with tab6:
            fig_theta = plot_all_options_greek(times=times, options=options,
                                               greek_values_list=[df[f"option_{i+1}_theta"] for i in range(len(options))],
                                               greek_name="Theta")
            fig_theta.update_layout(xaxis_title=f"Time ({time_interval.lower()})")
            st.plotly_chart(fig_theta, use_container_width=True)

        with tab7:
            fig_rho = plot_all_options_greek(times=times, options=options,
                                             greek_values_list=[df[f"option_{i+1}_rho"] for i in range(len(options))],
                                             greek_name="Rho")
            fig_rho.update_layout(xaxis_title=f"Time ({time_interval.lower()})")
            st.plotly_chart(fig_rho, use_container_width=True)

        with tab8:
            st.write("### Options Defined")
            opt_data = []
            for i, opt in enumerate(options):
                opt_data.append({
                    "Option": i + 1,
                    "Strike": opt.strike,
                    "Maturity (years)": opt.maturity,
                    "Type": "Call" if opt.is_call else "Put"
                })
            st.table(pd.DataFrame(opt_data))

if __name__ == "__main__":
    main()
