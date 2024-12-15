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
    def compute_d1_d2(S: np.ndarray, K: np.ndarray, r: float, sigma: float, T: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        # Pour éviter des warnings sur log(S/K) quand T=0 (on met T>0 de toute façon), on masque ces valeurs.
        safe_T = np.where(T > 0, T, np.nan)
        d1 = (np.log(S/K) + (r + 0.5*sigma**2)*safe_T) / (sigma*np.sqrt(safe_T))
        d2 = d1 - sigma*np.sqrt(safe_T)
        # On remet T=0 à d1, d2 => à T=0, ces valeurs n'ont pas de sens, mais on gèrera plus tard
        d1 = np.where(T > 0, d1, np.nan)
        d2 = np.where(T > 0, d2, np.nan)
        return d1, d2

    @staticmethod
    def compute_greeks(S: np.ndarray, K: np.ndarray, r: float, sigma: float, T: np.ndarray, is_call: np.ndarray) -> tuple:
        """
        Calcule en une fois price, delta, gamma, vega, theta, rho pour tous les contrats.
        S, K, T, is_call sont des arrays 1D.
        """
        # Gérer le cas T=0: l'option est à l'échéance
        payoff = np.maximum((S - K), 0)
        payoff_put = np.maximum((K - S), 0)

        # d1, d2
        d1, d2 = GreeksCalculator.compute_d1_d2(S=S, K=K, r=r, sigma=sigma, T=T)

        # Norm pdf et cdf
        pdf_d1 = norm.pdf(d1)
        cdf_d1 = norm.cdf(d1)
        cdf_d2 = norm.cdf(d2)

        # Price
        call_price = S*cdf_d1 - K*np.exp(-r*T)*cdf_d2
        put_price = K*np.exp(-r*T)*norm.cdf(-d2) - S*norm.cdf(-d1)
        # A l'échéance (T=0), price = payoff
        price = np.where(T > 0, np.where(is_call, call_price, put_price), np.where(is_call, payoff, payoff_put))

        # Delta
        delta = np.where(T > 0, np.where(is_call, cdf_d1, cdf_d1 - 1), np.where(is_call, (S > K).astype(float), (-1)*(S < K).astype(float)))

        # Gamma
        gamma = np.where(T > 0, pdf_d1/(S*sigma*np.sqrt(T)), 0.0)

        # Vega
        vega = np.where(T > 0, S*pdf_d1*np.sqrt(T), 0.0)

        # Theta
        # Pour Theta, utilisation de la formule standard
        # first_term = -(S * pdf_d1 * sigma) / (2*sqrt(T))
        # call_theta = first_term - r*K*exp(-rT)*cdf_d2
        # put_theta = first_term + r*K*exp(-rT)*cdf(-d2)
        # A T=0, Theta=0
        with np.errstate(invalid='ignore'):
            first_term = -(S*pdf_d1*sigma)/(2*np.sqrt(T))
            call_theta = first_term - r*K*np.exp(-r*T)*cdf_d2
            put_theta = first_term + r*K*np.exp(-r*T)*norm.cdf(-d2)
        theta = np.where(T > 0, np.where(is_call, call_theta, put_theta), 0.0)

        # Rho
        # call_rho = K*T*exp(-rT)*cdf_d2
        # put_rho = -K*T*exp(-rT)*cdf(-d2)
        rho = np.where(T > 0, np.where(is_call, K*T*np.exp(-r*T)*cdf_d2, -K*T*np.exp(-r*T)*norm.cdf(-d2)), 0.0)

        return price, delta, gamma, vega, theta, rho


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
        dt=dt,
        seed=42
    )

    simulator = GBMSimulator(params=gbm_params)
    prices = simulator.simulate(nb_steps=nb_seconds_per_day)
    times = np.arange(nb_seconds_per_day)

    # Définition d'options
    option_list = [
        OptionContract(strike=100.0, maturity=10 / 365, is_call=True),
        OptionContract(strike=105.0, maturity=15 / 365, is_call=False),
        OptionContract(strike=95.0, maturity=30 / 365, is_call=True),
        OptionContract(strike=100.0, maturity=60 / 365, is_call=False)
    ]

    r = 0.01
    sigma = gbm_params.sigma

    # Extraction des paramètres des options sous forme de tableaux
    strikes = np.array([opt.strike for opt in option_list])
    maturities = np.array([opt.maturity for opt in option_list])
    is_call = np.array([opt.is_call for opt in option_list])

    # Pré-allocation des résultats
    option_prices_list = [np.zeros(nb_seconds_per_day) for _ in option_list]
    deltas_list = [np.zeros(nb_seconds_per_day) for _ in option_list]
    gammas_list = [np.zeros(nb_seconds_per_day) for _ in option_list]
    vegas_list = [np.zeros(nb_seconds_per_day) for _ in option_list]
    thetas_list = [np.zeros(nb_seconds_per_day) for _ in option_list]
    rhos_list = [np.zeros(nb_seconds_per_day) for _ in option_list]

    # Au lieu de boucler sur les options en interne, on vectorise sur les options.
    # Boucle sur le temps
    for t in tqdm(range(nb_seconds_per_day), desc='Calcul des greeks et prix'):
        S = prices[t]
        # Calcul de T pour toutes les options vectoriellement
        T_arr = np.maximum(maturities - t*dt, 0.0)
        # Calcul vectorisé des greeks
        p, d, g, v, th, rh = GreeksCalculator.compute_greeks(
            S=np.full_like(strikes, S),
            K=strikes,
            r=r,
            sigma=sigma,
            T=T_arr,
            is_call=is_call
        )

        # Assignation aux arrays
        for i in range(len(option_list)):
            option_prices_list[i][t] = p[i]
            deltas_list[i][t] = d[i]
            gammas_list[i][t] = g[i]
            vegas_list[i][t] = v[i]
            thetas_list[i][t] = th[i]
            rhos_list[i][t] = rh[i]

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
