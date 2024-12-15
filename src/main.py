import numpy as np
import pandas as pd
from typing import List, Dict, Callable
from dataclasses import dataclass
from math import log, sqrt, exp, pi
import logging
from numba import njit
from tqdm import tqdm  # Pour la barre de progression
import plotly.graph_objects as go

logging.basicConfig(level=logging.INFO)

@dataclass(frozen=True)
class Option:
    """Représente une option européenne (call ou put)

    Attributes:
        strike (float): Prix d'exercice de l'option.
        maturity_days (float): Maturité de l'option exprimée en fraction d'année.
        option_type (str): Type de l'option, "call" ou "put".
    """
    strike: float
    maturity_days: float
    option_type: str = "call"


class DataLoader:
    """Class responsible for loading data from various file formats."""
    _read_funcs: Dict[str, Callable] = {
        'hdf': pd.read_hdf,
        'csv': pd.read_csv,
        'xlsx': pd.read_excel,
        'json': pd.read_json,
        'parquet': pd.read_parquet,
    }

    @staticmethod
    def load_data(data_path: str, **kwargs) -> pd.DataFrame:
        file_extension = data_path.split('.')[-1].lower()
        read_func = DataLoader._read_funcs.get(file_extension)
        if read_func:
            return read_func(data_path, **kwargs)
        else:
            raise ValueError(f"Unsupported file extension: {file_extension}")


class MarketDataHandler:
    """Charge et prépare les données du SPY."""
    def __init__(self, data: pd.DataFrame):
        """
        Args:
            data: DataFrame avec colonnes ['open', 'high', 'low', 'close', 'volume'] et un DatetimeIndex
        """
        # Assurer que l'index est un DatetimeIndex
        if not isinstance(data.index, pd.DatetimeIndex):
            if 'date' in data.columns:
                data['date'] = pd.to_datetime(data['date'])
                data.set_index('date', inplace=True)
            else:
                raise ValueError("Le DataFrame doit avoir un DatetimeIndex ou une colonne 'date'.")

        self.full_data = data.sort_index()

    def get_data_slice(self, start_time="13:00", end_time="15:40", freq='15min') -> pd.DataFrame:
        """
        Retourne une fenêtre temporelle filtrée et rééchantillonnée.
        """
        data = self.full_data.copy()
        mask = (data.index.time >= pd.to_datetime(start_time).time()) & (data.index.time <= pd.to_datetime(end_time).time())
        data = data.loc[mask]

        if freq is not None:
            data = data.resample(freq).agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            logging.info(f"Données rééchantillonnées en {freq}.")
        return data

    @staticmethod
    def get_spot_prices(data: pd.DataFrame) -> pd.Series:
        """
        Retourne la série de prix du sous-jacent ('close').
        """
        return data['close']


@njit
def erf(z: float) -> float:
    sign = 1.0 if z >= 0 else -1.0
    z_abs = abs(z)

    p = 0.3275911
    a1 = 0.254829592
    a2 = -0.284496736
    a3 = 1.421413741
    a4 = -1.453152027
    a5 = 1.061405429

    t = 1.0/(1.0 + p*z_abs)
    exp_term = exp(-z_abs*z_abs)
    poly = a1*t + a2*(t*t) + a3*(t**3) + a4*(t**4) + a5*(t**5)
    y = 1.0 - poly*exp_term
    return sign*y

@njit
def cdf_normal(x: float) -> float:
    return 0.5*(1.0 + erf(x/sqrt(2.0)))

@njit
def pdf_normal(x: float) -> float:
    return (1.0 / sqrt(2.0*pi)) * exp(-0.5*x*x)

@njit
def black_scholes_call_put(S: float, K: float, r: float, sigma: float, T: float, is_call: int):
    if T <= 0:
        payoff = max(0.0, S - K) if is_call == 1 else max(0.0, K - S)
        return payoff, 0.0, 0.0, 0.0, 0.0, 0.0

    d1 = (log(S/K) + (r + 0.5*sigma*sigma)*T)/(sigma*sqrt(T))
    d2 = d1 - sigma*sqrt(T)

    Nd1 = cdf_normal(d1)
    Nd2 = cdf_normal(d2)
    nd1 = pdf_normal(d1)

    if is_call == 1:
        price = S*Nd1 - K*exp(-r*T)*Nd2
        delta = Nd1
        rho_val = K*T*exp(-r*T)*Nd2
        Nd2_r = Nd2
    else:
        Nmd1 = cdf_normal(-d1)
        Nmd2 = cdf_normal(-d2)
        price = K*exp(-r*T)*Nmd2 - S*Nmd1
        delta = Nd1 - 1.0
        rho_val = -K*T*exp(-r*T)*Nmd2
        Nd2_r = -Nmd2

    gamma = nd1/(S*sigma*sqrt(T))
    theta = -(S*nd1*sigma)/(2.0*sqrt(T)) - r*K*exp(-r*T)*Nd2_r
    vega = S*nd1*sqrt(T)

    return price, delta, gamma, theta, vega, rho_val


class BlackScholesPricer:
    def __init__(self, r: float, sigma: float):
        self.r = r
        self.sigma = sigma

    def price_and_greeks(self, S: float, option: Option) -> Dict[str, float]:
        is_call = 1 if option.option_type == 'call' else 0
        T = option.maturity_days
        price, delta, gamma, theta, vega, rho = black_scholes_call_put(S, option.strike, self.r, self.sigma, T, is_call)
        return {
            'price': price,
            'delta': delta,
            'gamma': gamma,
            'theta': theta,
            'vega': vega,
            'rho': rho
        }


class SimulationEngine:
    def __init__(self, market_data_handler: MarketDataHandler, pricer: BlackScholesPricer, options: List[Option]):
        self.market_data_handler = market_data_handler
        self.pricer = pricer
        self.options = options
        self.results = {}

    def run_simulation(self, start_time: str = "13:00", end_time: str = "15:40", freq='15min'):
        data_slice = self.market_data_handler.get_data_slice(start_time=start_time, end_time=end_time, freq=freq)
        spot_series = MarketDataHandler.get_spot_prices(data_slice)

        # On ajoute 'spot' aux résultats pour visualiser la trajectoire du sous-jacent
        self.results = {opt: {'time': [], 'spot': [], 'price': [], 'delta': [], 'gamma': [], 'theta': [], 'vega': [], 'rho': []}
                        for opt in self.options}

        for t, S in tqdm(spot_series.items(), desc="Simulation en cours", unit="timestamps"):
            for opt in self.options:
                res = self.pricer.price_and_greeks(S, opt)
                # Ajout des résultats
                self.results[opt]['time'].append(t)
                self.results[opt]['spot'].append(S)
                for k, v in res.items():
                    self.results[opt][k].append(v)


class Visualizer:
    def __init__(self, results: Dict[Option, Dict[str, List[float]]]):
        self.results = results

    def to_dataframe(self, opt: Option) -> pd.DataFrame:
        """
        Convertit les résultats pour un seul option en DataFrame pour une manipulation plus aisée.
        """
        data = self.results[opt]
        df = pd.DataFrame({
            'time': data['time'],
            'spot': data['spot'],
            'price': data['price'],
            'delta': data['delta'],
            'gamma': data['gamma'],
            'theta': data['theta'],
            'vega': data['vega'],
            'rho': data['rho']
        })
        df.set_index('time', inplace=True)
        return df

    def plot_interactive(self, opt: Option):
        """
        Crée des graphiques interactifs avec plotly.
        On va tracer les trajectoires par jour. Chaque journée aura sa propre ligne.
        """
        df = self.to_dataframe(opt)

        # Extraire la date (sans l'heure) pour distinguer chaque journée
        df['date'] = df.index.date

        # Trajectoire du sous-jacent (SPY)
        fig_spot = go.Figure()
        for d in df['date'].unique():
            df_day = df[df['date'] == d]
            fig_spot.add_trace(go.Scatter(
                x=df_day.index,
                y=df_day['spot'],
                mode='lines',
                name=str(d),
                hovertemplate='Time: %{x}<br>Spot: %{y:.2f}<extra></extra>'
            ))
        fig_spot.update_layout(
            title="Trajectoire du sous-jacent (SPY) de 13h00 à 15h40",
            xaxis_title="Temps",
            yaxis_title="Spot Price"
        )

        # Prix de l'option
        fig_price = go.Figure()
        for d in df['date'].unique():
            df_day = df[df['date'] == d]
            fig_price.add_trace(go.Scatter(
                x=df_day.index,
                y=df_day['price'],
                mode='lines',
                name=str(d),
                hovertemplate='Time: %{x}<br>Price: %{y:.2f}<extra></extra>'
            ))
        fig_price.update_layout(
            title=f"Évolution du prix de l'option (Strike={opt.strike}, T={opt.maturity_days*365:.0f}j)",
            xaxis_title="Temps",
            yaxis_title="Option Price"
        )

        # Greeks : delta, gamma, theta, vega, rho
        # On peut faire un subplot, ou bien 5 graph séparés. Pour plus de clarté, on va faire 5 graph distincts.
        greeks = ['delta', 'gamma', 'theta', 'vega', 'rho']
        figs_greeks = {}
        for g in greeks:
            fig_g = go.Figure()
            for d in df['date'].unique():
                df_day = df[df['date'] == d]
                fig_g.add_trace(go.Scatter(
                    x=df_day.index,
                    y=df_day[g],
                    mode='lines',
                    name=str(d),
                    hovertemplate=f'Time: %{{x}}<br>{g.capitalize()}: %{{y:.6f}}<extra></extra>'
                ))
            fig_g.update_layout(
                title=f"Évolution de {g.capitalize()} (Strike={opt.strike}, T={opt.maturity_days*365:.0f}j)",
                xaxis_title="Temps",
                yaxis_title=g.capitalize()
            )
            figs_greeks[g] = fig_g

        # Affichage : dans un notebook, on utiliserait fig.show().
        # Si vous exécutez ce script standalone, assurez-vous de le faire dans un environnement qui supporte l'affichage (ex: Jupyter)
        fig_spot.show()
        fig_price.show()
        for g in greeks:
            figs_greeks[g].show()


if __name__ == "__main__":
    # Paramètres
    data_path = "../data/raw_data/SPY_1min_2009-2024.parquet"
    r = 0.01
    sigma = 0.2

    # Chargement des données via DataLoader
    data = DataLoader.load_data(data_path)
    mdh = MarketDataHandler(data)

    pricer = BlackScholesPricer(r=r, sigma=sigma)

    # Se concentrer sur une seule option
    option = Option(strike=450, maturity_days=30/365)
    options_list = [option]

    engine = SimulationEngine(market_data_handler=mdh, pricer=pricer, options=options_list)
    engine.run_simulation(start_time="13:00", end_time="15:40", freq="15min")

    viz = Visualizer(engine.results)
    # Création des graphiques interactifs
    viz.plot_interactive(option)
