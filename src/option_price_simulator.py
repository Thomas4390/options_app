import numpy as np
from dataclasses import dataclass
from scipy.stats import norm
from tqdm import tqdm
import matplotlib.pyplot as plt


@dataclass
class GBMParameters:
    """
    Paramètres pour le modèle de mouvement brownien géométrique.
    """
    spot: float
    mu: float            # drift annualisé
    sigma: float         # volatilité annualisée
    dt: float            # pas de temps en années (ici, seconde / nb_sec_annee)
    seed: int = 42


class GBMSimulator:
    """
    Cette classe gère la simulation de la trajectoire d'un sous-jacent selon un GBM.
    """
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
    def plot_prices(times: np.ndarray, prices: np.ndarray):
        plt.figure(figsize=(10, 5))
        plt.plot(times, prices, label='Sous-jacent')
        plt.title("Trajectoire du sous-jacent simulé (GBM)")
        plt.xlabel("Temps (secondes)")
        plt.ylabel("Prix")
        plt.grid(True)
        plt.legend()
        plt.show()


@dataclass
class OptionContract:
    """
    Classe définissant une option européenne simple.
    """
    strike: float
    maturity: float      # en années (ex: 1 jour ~ 1/365 si 1 jour)
    is_call: bool

    def get_label(self) -> str:
        option_type = "Call" if self.is_call else "Put"
        return f"{option_type}, Maturity={self.maturity:.6f}y, Strike={self.strike:.2f}"


class GreeksCalculator:
    """
    Classe pour calculer les greeks d'une option européenne vanille en utilisant la formule de Black-Scholes.
    """
    @staticmethod
    def d1(S: float, K: float, r: float, sigma: float, T: float) -> float:
        return (np.log(S/K) + (r + 0.5*sigma**2)*T) / (sigma * np.sqrt(T))

    @staticmethod
    def d2(S: float, K: float, r: float, sigma: float, T: float) -> float:
        return GreeksCalculator.d1(S=S, K=K, r=r, sigma=sigma, T=T) - sigma*np.sqrt(T)

    @staticmethod
    def price(S: float, K: float, r: float, sigma: float, T: float, is_call: bool) -> float:
        if T <= 0:
            payoff = max((S - K), 0) if is_call else max((K - S), 0)
            return payoff
        d1_val = GreeksCalculator.d1(S=S, K=K, r=r, sigma=sigma, T=T)
        d2_val = d1_val - sigma*np.sqrt(T)
        if is_call:
            return S*norm.cdf(d1_val) - K*np.exp(-r*T)*norm.cdf(d2_val)
        else:
            return K*np.exp(-r*T)*norm.cdf(-d2_val) - S*norm.cdf(-d1_val)

    @staticmethod
    def delta(S: float, K: float, r: float, sigma: float, T: float, is_call: bool) -> float:
        d1_val = GreeksCalculator.d1(S=S, K=K, r=r, sigma=sigma, T=T)
        return norm.cdf(d1_val) if is_call else norm.cdf(d1_val) - 1

    @staticmethod
    def gamma(S: float, K: float, r: float, sigma: float, T: float) -> float:
        if T <= 0:
            return 0.0
        d1_val = GreeksCalculator.d1(S=S, K=K, r=r, sigma=sigma, T=T)
        return norm.pdf(d1_val) / (S*sigma*np.sqrt(T))

    @staticmethod
    def vega(S: float, K: float, r: float, sigma: float, T: float) -> float:
        if T <= 0:
            return 0.0
        d1_val = GreeksCalculator.d1(S=S, K=K, r=r, sigma=sigma, T=T)
        return S * norm.pdf(d1_val)*np.sqrt(T)

    @staticmethod
    def theta(S: float, K: float, r: float, sigma: float, T: float, is_call: bool) -> float:
        if T <= 0:
            return 0.0
        d1_val = GreeksCalculator.d1(S=S, K=K, r=r, sigma=sigma, T=T)
        d2_val = d1_val - sigma * np.sqrt(T)
        first_term = - (S * norm.pdf(d1_val)*sigma) / (2*np.sqrt(T))
        if is_call:
            return first_term - r*K*np.exp(-r*T)*norm.cdf(d2_val)
        else:
            return first_term + r*K*np.exp(-r*T)*norm.cdf(-d2_val)

    @staticmethod
    def rho(S: float, K: float, r: float, sigma: float, T: float, is_call: bool) -> float:
        if T <= 0:
            return 0.0
        d2_val = GreeksCalculator.d2(S=S, K=K, r=r, sigma=sigma, T=T)
        if is_call:
            return K*T*np.exp(-r*T)*norm.cdf(d2_val)
        else:
            return -K*T*np.exp(-r*T)*norm.cdf(-d2_val)


def plot_all_options_prices(times: np.ndarray, options: list[OptionContract], option_prices_list: list[np.ndarray]):
    """
    Affiche l'évolution des prix pour toutes les options sur un même graphique.
    """
    plt.figure(figsize=(10, 5))
    for opt, opt_prices in zip(options, option_prices_list):
        plt.plot(times, opt_prices, label=opt.get_label())
    plt.title("Évolution du prix des options")
    plt.xlabel("Temps (secondes)")
    plt.ylabel("Prix de l'option")
    plt.grid(True)
    plt.legend()
    plt.show()


def plot_all_options_greek(times: np.ndarray, options: list[OptionContract], greek_values_list: list[np.ndarray], greek_name: str):
    """
    Affiche l'évolution d'un grec (delta, gamma, vega, theta, rho) pour toutes les options sur un même graphique.
    """
    plt.figure(figsize=(10, 5))
    for opt, gv in zip(options, greek_values_list):
        plt.plot(times, gv, label=opt.get_label())
    plt.title(f"Évolution du {greek_name} des options")
    plt.xlabel("Temps (secondes)")
    plt.ylabel(f"{greek_name}")
    plt.grid(True)
    plt.legend()
    plt.show()


def main():
    # Paramètres de base
    nb_seconds_per_day = 86400
    day_in_years = 1/365
    dt = day_in_years / nb_seconds_per_day

    # Paramètres du GBM
    gbm_params = GBMParameters(
        spot=100.0,
        mu=0.05,
        sigma=0.2,
        dt=dt
    )

    simulator = GBMSimulator(params=gbm_params)
    prices = simulator.simulate(nb_steps=nb_seconds_per_day)
    times = np.arange(nb_seconds_per_day)

    # Définition d'options
    option_list = [
        OptionContract(strike=100.0, maturity=10/365, is_call=True),
        OptionContract(strike=105.0, maturity=15/365, is_call=False),
        OptionContract(strike=95.0, maturity=30/365, is_call=True),
        OptionContract(strike=100.0, maturity=60/365, is_call=False)
    ]

    r = 0.01
    sigma = gbm_params.sigma

    # On va stocker les prix et greeks de toutes les options.
    option_prices_list = []
    deltas_list = []
    gammas_list = []
    vegas_list = []
    thetas_list = []
    rhos_list = []

    # Initialiser les arrays
    for _ in option_list:
        option_prices_list.append(np.zeros(nb_seconds_per_day))
        deltas_list.append(np.zeros(nb_seconds_per_day))
        gammas_list.append(np.zeros(nb_seconds_per_day))
        vegas_list.append(np.zeros(nb_seconds_per_day))
        thetas_list.append(np.zeros(nb_seconds_per_day))
        rhos_list.append(np.zeros(nb_seconds_per_day))

    # Boucle de simulation
    for t in tqdm(range(nb_seconds_per_day), desc='Calcul des greeks et prix'):
        S = prices[t]
        for i, opt in enumerate(option_list):
            opt_T = max(opt.maturity - t*dt, 0)
            opt_price = GreeksCalculator.price(S=S, K=opt.strike, r=r, sigma=sigma, T=opt_T, is_call=opt.is_call)
            delta = GreeksCalculator.delta(S=S, K=opt.strike, r=r, sigma=sigma, T=opt_T, is_call=opt.is_call)
            gamma = GreeksCalculator.gamma(S=S, K=opt.strike, r=r, sigma=sigma, T=opt_T)
            vega = GreeksCalculator.vega(S=S, K=opt.strike, r=r, sigma=sigma, T=opt_T)
            theta = GreeksCalculator.theta(S=S, K=opt.strike, r=r, sigma=sigma, T=opt_T, is_call=opt.is_call)
            rho = GreeksCalculator.rho(S=S, K=opt.strike, r=r, sigma=sigma, T=opt_T, is_call=opt.is_call)

            option_prices_list[i][t] = opt_price
            deltas_list[i][t] = delta
            gammas_list[i][t] = gamma
            vegas_list[i][t] = vega
            thetas_list[i][t] = theta
            rhos_list[i][t] = rho

    # Plot du sous-jacent
    GBMSimulator.plot_prices(times=times, prices=prices)

    # Plot de tous les prix d'options
    plot_all_options_prices(times=times, options=option_list, option_prices_list=option_prices_list)

    # Plot de chaque grec pour toutes les options
    plot_all_options_greek(times=times, options=option_list, greek_values_list=deltas_list, greek_name="Delta")
    plot_all_options_greek(times=times, options=option_list, greek_values_list=gammas_list, greek_name="Gamma")
    plot_all_options_greek(times=times, options=option_list, greek_values_list=vegas_list, greek_name="Vega")
    plot_all_options_greek(times=times, options=option_list, greek_values_list=thetas_list, greek_name="Theta")
    plot_all_options_greek(times=times, options=option_list, greek_values_list=rhos_list, greek_name="Rho")


if __name__ == "__main__":
    main()
