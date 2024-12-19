import streamlit as st

st.set_page_config(
    page_title="GBM and Options Simulation",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.title("GBM and Options Simulation Application")

st.write("""
This application simulates the behavior of an underlying asset using a **Geometric Brownian Motion (GBM)** model 
and analyzes multi-legged option strategies, such as the Iron Condor.

**Key Features:**
- **Single Path Simulation (Simulation):** Simulate a single path of the underlying and analyze an Iron Condor strategy in detail, including Greeks, dynamic break-evens, and scaled P&L according to a chosen investment amount.
- **Monte Carlo Simulation:** Run multiple simulations to perform statistical analysis of P&L, Greeks, risk measures (VaR, ES), percentile bands of paths, probabilities of hitting P&L thresholds, and view individual simulation scenarios by selecting their index.

**Interactive Visualizations:** Use interactive Plotly charts to explore results.
**Scenario Analysis:** Adjust parameters and instantly see the effects on P&L and Greeks.
**Greeks Analysis:** Evaluate Delta, Gamma, Vega, Theta, and Rho for each leg and the entire position.

**Navigation:**
Use the sidebar menu to navigate:
- **Simulation:** Single-path simulation page (enhanced Iron Condor analysis).
- **Monte Carlo Simulation:** Multiple simulations for statistical analyses and scenario exploration.
""")

st.info("Use the sidebar menu on the left to select a page and begin.")
