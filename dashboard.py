import datetime
import streamlit as st
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from InvestmentPortfolioStressTester import PortfolioStressTester

st.set_page_config(page_title="Risk Engine", layout="wide")

# Header
with st.container():
    st.title("⚙️Investment Portfolio Risk Engine")
    st.info("Analyse portfolio resilience through Multivariate Monte Carlo Simulations and Systemic Stress Testing")

## Upper configuration bar
# Using columns to pull key inputs out of the sidebar for better flow
with st.expander("Portfolio & Simulation Configuration", expanded=True):
    col1, col2, col3 = st.columns([2, 1, 1])
    with col1:
        tickers = st.multiselect(
            "Select Portfolio Assets", 
            ["GOOG", "NVDA", "AVGO", "BTC-USD", "TSLA", "SPY", "VOO"], 
            default=["GOOG", "NVDA", "AVGO", "BTC-USD"]
        )
    with col2:
        base = st.number_input("Initial Capital (USD$)", value=23270)
    with col3:
        dayHorizon = st.selectbox("Forecast Horizon (Days)", [30, 60, 90, 180, 365], index=0)
    st.write("Historical Data Lookback Period")
    # Define the options
    lookback_options = ["1M", "3M", "6M", "1Y", "3Y", "5Y"]

    # Group of buttons where the selected one stays highlighted
    selection = st.segmented_control("Select Period", options=lookback_options, default="3Y", label_visibility="collapsed")

    # Logic to map selection to actual start dates
    today = datetime.date.today()
    mapping = {
        "1M": today - datetime.timedelta(days=30),
        "3M": today - datetime.timedelta(days=90),
        "6M": today - datetime.timedelta(days=182),
        "1Y": today - datetime.timedelta(days=365),
        "3Y": today - datetime.timedelta(days=3*365),
        "5Y": today - datetime.timedelta(days=5*365),
    }
    # Update session state based on the segmented control
    st.session_state.start_date = mapping[selection]

    # Display the dates to confirm selection
    col_d1, col_d2 = st.columns(2)
    finalStart = st.session_state.start_date
    finalEnd = today

# Sidebar for stress test configuration
with st.sidebar:
    st.header("Stress Test Triggers")
    st.markdown("Adjust these factors to simulate a Market Crash event")
    shock_vol = st.slider("Volatility Multiplier", 1.0, 4.0, 2.0, help="Simulates market panic by inflating asset variance")
    mkt_gap = st.slider("Overnight Gap Down (%)", -50.0, 0.0, -5.0, 1.0)/100
    mean_shock = st.slider("Daily Negative Drift", -0.10, 0.0, -0.05, 0.01, help="Simulates sustained downward pressure/panic selling")
    simulations = st.select_slider("Simulation Timesteps", options=[500, 1000, 1500, 2000, 3000], value=1500)

# Main analysis engine
if tickers:
    # Handle Weights in a dedicated container
    with st.container():
        st.subheader("Asset Allocation")
        weight_cols = st.columns(len(tickers))
        weights = []
        for i, t in enumerate(tickers):
            with weight_cols[i]:
                w = st.number_input(f"% {t}", 0, 100, int(100/len(tickers)))
                weights.append(w/100)
        
        weight_sum = sum(weights)
        if not np.isclose(weight_sum, 1.0):
            st.warning(f"Weights normalised from {weight_sum*100:.1f}% to 100%.")
            weights = [w/weight_sum for w in weights]

    if st.button("Run Risk Analysis", width='stretch'):
        with st.spinner("Processing Market Data..."):
            # Date logic
            start = finalStart.strftime("%Y-%m-%d")
            end = finalEnd.strftime("%Y-%m-%d")
            tester = PortfolioStressTester(tickers, weights, base)
            tester.fetchData(start, end)
            
            general, annualReturn, annualVolatility = tester.runMonteCarloSimulation(dayHorizon, simulations)
            crash, annualReturnCrash, annualVolatilityCrash = tester.runMonteCarloSimulation(dayHorizon, simulations, shock_vol, mkt_gap, mean_shock)

            # Metrics
            var95 = base - np.percentile(general[-1, :], 5)
            stressvar95 = base - np.percentile(crash[-1, :], 5)
            # Sharpe Ratio (Risk free rate at 4%)
            sharpeGeneral = (annualReturn-0.0365)/annualVolatility
            sharpeCrash = (annualReturnCrash-0.0365)/annualVolatilityCrash

            m_col1, m_col2, m_col3, m_col4, m_col5 = st.columns(5)
            m_col1.metric("Starting Principal", f"${base:,.0f}")
            m_col2.metric("Normal VaR", f"${var95:,.2f}", help="The minimum amount that expected to lose in the worst 5% of outcomes")
            m_col3.metric("Stress VaR", f"${stressvar95:,.2f}", 
                          delta=f"{((stressvar95/var95)-1)*100:.1f}% vs Normal", delta_color="inverse")
            m_col4.metric("Normal Sharpe", f"{sharpeGeneral:.2f}", help="Measures risk-adjusted return. " \
            "A ratio > 1.0 is considered good, while > 2.0 is excellent. It tells whether returns are worth the volatility.")
            m_col5.metric("Stress Sharpe", f"{sharpeCrash:.2f}", delta=f"{sharpeCrash-sharpeGeneral:.2f}")

            # Tabs for organised results
            tab1, tab2, tab3, tab4 = st.tabs(["📈Simulation Forecast", "𝜌 Asset Correlations", "🛡️Hedging", "📑Raw Market Data",])

            with tab1:
                st.subheader("Monte Carlo Simulations")
                time_steps = np.arange(dayHorizon)
                fig = go.Figure()
                
                # Normal Market
                gen_median = np.median(general, axis=1)
                fig.add_trace(go.Scatter(x=time_steps, y=np.percentile(general, 95, axis=1), mode='lines', line=dict(color='blue', width=1), name='Upper Bound'))
                fig.add_trace(go.Scatter(x=time_steps, y=np.percentile(general, 5, axis=1), fill='tonexty', fillcolor='rgba(0,0,255,0.1)', name='Normal 90% CI'))
                fig.add_trace(go.Scatter(x=time_steps, y=gen_median, line=dict(color='blue', width=3), name='General Median'))
                
                # Crash Scenario
                fig.add_trace(go.Scatter(x=time_steps, y=np.median(crash, axis=1), line=dict(color='red', width=3, dash='dash'), name='Crash Median'))
                fig.add_trace(go.Scatter(x=time_steps, y=np.percentile(crash, 5, axis=1), fill='tonexty', fillcolor='rgba(255,0,0,0.2)', name='Extreme Tail Risk'))

                fig.update_layout(template="plotly_white", hovermode="x unified", height=600)
                st.plotly_chart(fig, width='stretch')

            with tab2:
                col_a, col_b = st.columns([1.2, 1]) # Adjusting ratio to give the heatmap more room
    
                with col_a:
                    st.subheader("Asset Correlation Matrix")
                    # Generate the figure
                    plt.style.use("dark_background")
                    fig_corr, ax_corr = plt.subplots(figsize=(10, 8))
                    # Using the logReturns from your tester class
                    corr = tester.logReturns.corr()
                    
                    # Plotting the heatmap
                    sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdYlGn", center=0, ax=ax_corr)
                    plt.title("Historical Correlation (Daily Log Returns)")
                    st.pyplot(fig_corr)
                    
                with col_b:
                    st.subheader("Risk Analysis Insights")
                    st.markdown(f"""
                    **What this tells:**
                    * **Diversification Strength:** Low values (<0.2) indicate assets move independently, providing better protection.
                    * **Systemic Risk:** High values (>0.7) suggest assets are likely to crash simultaneously during a Market Crash event.
                    
                    **Stress Scenario Note:**
                    In edge case simulation, these correlations are artificially inflated to simulate **Liquidity Contagion**, where investors sell all assets at once to raise cash.
                    """)

            with tab3:
                st.subheader("Optimal Hedge Suggestion")
                risk_gap = stressvar95 - var95
                hedge_req = (risk_gap / base) * 100
                st.warning(f"To neutralise the additional risk, consider reallocating **{hedge_req:.2f}%** to defensive/non-correlated assets.")
            
            with tab4:
                st.subheader("Raw Portfolio Data")
                raw_data = tester.data['Close'].dropna()
                st.dataframe(raw_data, width='stretch')
                csv = raw_data.to_csv().encode('utf-8')
                st.download_button(label="Download Raw Data as CSV", data=csv, file_name=f"portfolio_data_{datetime.date.today()}.csv", mime="text/csv")
                
                
else:
    st.warning("Select assets in the configuration bar to begin.")

