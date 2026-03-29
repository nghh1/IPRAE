import datetime
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from InvestmentPortfolioStressTester import PortfolioStressTester

st.set_page_config(page_title="Risk Engine", layout="wide")

# Header
with st.container():
    st.title("Investment Portfolio Risk Analyser", anchor=False)
## Upper configuration bar
# Using columns to pull key inputs out of the sidebar for better flow
with st.expander("Portfolio & Simulation Configuration", expanded=True):
    col1, col2 = st.columns([2, 2])
    with col1:
        selected_tickers = st.multiselect(
            "Select Portfolio Assets", 
            ["GOOG", "NVDA", "AVGO", "AAPL", "MSFT", "AMZN", "TSLA", "SPY", "VOO", "BTC-USD"], 
            default=["GOOG", "NVDA", "AVGO", "BTC-USD"]
        )
    with col2:
        custom_tickers_input = st.text_input("Add Custom Tickers (comma-separated)", value="", placeholder="e.g. AMD, TSM")
        custom_tickers = [t.strip().upper() for t in custom_tickers_input.split(',') if t.strip()]
        tickers = list(dict.fromkeys(selected_tickers + custom_tickers))
    col3, col4, col5 = st.columns([1,1,1])
    with col3:
        # Define the options
        lookback_options = ["1M", "3M", "6M", "1Y", "3Y", "5Y"]
        # Group of buttons where the selected one stays highlighted
        selection = st.segmented_control("Historical Data Lookback Period", options=lookback_options, default="3Y", width='stretch')
    with col4:
        base = st.number_input("Initial Capital (USD$)", value=23270)
    with col5:
        dayHorizon = st.selectbox("Forecast Horizon (Days)", [30, 60, 90, 180, 365], index=0)

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
    st.header("Factors Configuration")
    shock_vol = st.slider("Volatility Multiplier", 1.0, 4.0, 2.0, help="Amplifies historical price swings. 2.0x means the market becomes twice as jittery as usual.")
    mkt_gap = st.slider("Overnight Gap Down (%)", -50.0, 0.0, -5.0, 1.0, help="Simulates an immediate market price drop (x% gap down) before the simulation starts.")/100
    mean_shock = st.slider("Annualized Negative Drift", -1.0, 0.0, -0.20, 0.05, 
                           help="Simulates a sustained bear market trend (e.g., -0.20 is a 20% yearly drop)")
    simulations = st.select_slider("Simulation Timesteps", options=[500, 1000, 1500, 2000, 3000], value=1500)
    rebalance_toggle = st.checkbox("Enable Daily Rebalancing", value=False, help="If unchecked, simulates 'Buy and Hold' where winning assets grow to represent more of the portfolio.")

# Main analysis engine
if tickers:
    # Handle Weights in a dedicated container
    with st.container():
        st.subheader("Asset Allocation")
        weight_cols = st.columns(len(tickers))
        weights = []
        for i, t in enumerate(tickers):
            with weight_cols[i]:
                w = st.number_input(f"% {t}", min_value=0.0, max_value=100.0, value=100.0/len(tickers), format="%.2f")
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
            
            try:
                tester.fetchData(start, end)
            except Exception as e:
                # Catches KeyError (empty DataFrame) or any other yfinance fetch errors (e.g., ValueError for malformed inputs)
                st.error(f"🚨 **Data Fetch Error:** One or more assets could not be found or are malformed. Please verify the symbols. (Error: {e})")
                st.stop()

            # Validate that all tickers returned valid data 
            invalid_tickers = [t for t in tickers if t not in tester.logReturns.columns]
            
            if invalid_tickers:
                st.error(f"🚨 **Invalid assets:** {', '.join(invalid_tickers)}. "
                         "Please check for typos, then try again.")
                st.stop()

            general, annualReturn, annualVolatility = tester.runSimulation(dayHorizon, simulations, rebalance=rebalance_toggle)
            crash, annualReturnCrash, annualVolatilityCrash = tester.runSimulation(dayHorizon, simulations, shockVolatility=shock_vol, marketGap=mkt_gap, meanShock=mean_shock, rebalance=rebalance_toggle)

            # Metrics
            # Extract final values from all simulation paths
            final_general = general[-1, :]
            final_crash = crash[-1, :]
            # Value at Risk (VaR)
            var95 = base - np.percentile(final_general, 5)
            stressvar95 = base - np.percentile(final_crash, 5)
            # Conditional Value at Risk (CVaR / Expected Shortfall)
            # This calculates the AVERAGE loss of all scenarios in that bottom 5%
            tail_general = final_general[final_general <= np.percentile(final_general, 5)]
            cvar95 = base - np.mean(tail_general)
            tail_crash = final_crash[final_crash <= np.percentile(final_crash, 5)]
            stress_cvar95 = base - np.mean(tail_crash)
            # Sharpe Ratio
            var_percent_change = ((stressvar95 / var95) - 1) * 100
            cvar_percent_change = ((stress_cvar95 / cvar95) - 1) * 100
            var_delta_text = f"{var_percent_change:.1f}% vs Normal"
            cvar_delta_text = f"{cvar_percent_change:.1f}% vs Normal"
            # Sharpe Ratio Sanity Check
            # If vol is near zero, Sharpe is technically undefined;
            sharpeGeneral = (annualReturn - 0.0365) / (annualVolatility + 1e-9)
            sharpeCrash = (annualReturnCrash - 0.0365) / (annualVolatilityCrash + 1e-9)

            base_col1, base_col2, base_col3 = st.columns(3)
            with base_col1:
                st.metric("Normal VaR (95%)", f"${var95:,.2f}", 
                        help="The minimum loss expected in the worst 5% of normal outcomes.")
                
            with base_col2:
                st.metric("Normal CVaR (Tail Risk)", f"${cvar95:,.2f}", 
                        help="The average loss if you land in the worst 5% of normal outcomes.")
                
            with base_col3:
                st.metric("Normal Sharpe Ratio", f"{sharpeGeneral:.2f}", 
                        help="Measures how much return you get for every unit of volatility.")
            st.divider()
            stress_col1, stress_col2, stress_col3 = st.columns(3)
            with stress_col1:
                var_delta = f"{((stressvar95 / (var95 + 1e-9)) - 1) * 100:.1f}% vs Normal" if var95 > 0.01 else "New Risk"
                st.metric("Stress VaR", f"${stressvar95:,.2f}", 
                        delta=var_delta, delta_color="inverse", help="VaR when Volatility Multipliers and Gaps are applied.")

            with stress_col2:
                cvar_delta = f"{((stress_cvar95 / (cvar95 + 1e-9)) - 1) * 100:.1f}% vs Normal" if cvar95 > 0.01 else "New Risk"
                st.metric("Stress CVaR", f"${stress_cvar95:,.2f}", 
                        delta=cvar_delta, delta_color="inverse", help="CVaR when Volatility Multipliers and Gaps are applied.")

            with stress_col3:
                st.metric("Stress Sharpe Ratio", f"{sharpeCrash:.2f}", 
                        delta=f"{sharpeCrash - sharpeGeneral:.2f}", 
                        help="How the risk-reward efficiency collapses under stress.")

            # Tabs for organised results
            tab1, tab2, tab3, tab4, tab5 = st.tabs(["Historical Trends", "Probabilistic Forecast", "Risk Attribution", "Hedging", "Raw Market Data"])
            with tab1:
                st.subheader("Historical Asset Performance (Base 100)")
                st.markdown("""
                This chart visualises the relative growth of each asset over the selected lookback period. 
                All prices are normalised to start at **100**.
                """)
                historical_prices = tester.closePrices.dropna()
                normalised_prices = (historical_prices/historical_prices.iloc[0])*100
                historyFigure = go.Figure()
                for asset in normalised_prices.columns:
                    historyFigure.add_trace(go.Scatter(
                        x=normalised_prices.index,
                        y=normalised_prices[asset],
                        name=asset,
                        mode='lines',
                        hovertemplate='%{y:.2f} (Index)'
                    ))
                historyFigure.update_layout(
                    template='plotly',
                    xaxis_title='Timeline',
                    yaxis_title="Relative Growth (Base 100)",
                    hovermode='x unified',
                    height=600,
                    legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1),
                    margin=dict(l=0, r=0, t=0, b=0)
                )
                st.plotly_chart(historyFigure, width='stretch')
                with st.expander("How to Read this Trend"):
                    st.write(f"""
                    * Best Performer: The asset at the top of the chart has the highest cumulative return since {finalStart.strftime('%Y-%m-%d')}.
                    * Volatility Check: Assets with the most 'jagged' lines represent the highest historical risk in your portfolio.
                    """)

            with tab2:
                st.subheader("Historical Bootstrap Forecast")
                # Calculate the percentage of paths that end below the starting principal
                failure_rate = np.mean(general[-1, :] < base) * 100
                crash_failure_rate = np.mean(crash[-1, :] < base) * 100
                col_f1, col_f2 = st.columns(2)
                col_f1.metric("Probability of Loss (Normal)", f"{failure_rate:.1f}%")
                col_f2.metric("Probability of Loss (Stress)", f"{crash_failure_rate:.1f}%", delta=f"{crash_failure_rate - failure_rate:.1f}%", delta_color="inverse")
                time_steps = np.arange(general.shape[0])
                fig = go.Figure()
                
                fig.add_shape(
                    type="line", line=dict(color="white", width=2, dash="dot"), x0=0, x1=dayHorizon, y0=base, y1=base, name="Initial Principal"
                )
                # Normal Market Traces
                gen_median = np.median(general, axis=1)
                fig.add_trace(go.Scatter(x=time_steps, y=np.percentile(general, 95, axis=1), 
                                        mode='lines', line=dict(color='rgba(0,0,255,0.2)', width=1), showlegend=False, hoverinfo='skip'))
                fig.add_trace(go.Scatter(x=time_steps, y=np.percentile(general, 5, axis=1), 
                                        fill='tonexty', fillcolor='rgba(0,0,255,0.1)', name='Normal 90% Confidence'))
                fig.add_trace(go.Scatter(x=time_steps, y=gen_median, 
                                        line=dict(color='blue', width=3), name='General Median Outcome'))
                # Crash Scenario Traces
                fig.add_trace(go.Scatter(x=time_steps, y=np.median(crash, axis=1), 
                                        line=dict(color='red', width=3, dash='dash'), name='Stress Median Outcome'))
                fig.add_trace(go.Scatter(x=time_steps, y=np.percentile(crash, 5, axis=1), 
                                        fill='tonexty', fillcolor='rgba(255,0,0,0.2)', name='Extreme Tail Risk (5th Pct)'))
                fig.update_layout(
                    title="Simulated Portfolio Value Paths",
                    xaxis_title="Days into Future",
                    yaxis_title="Portfolio Value ($)",
                    template="plotly_white", 
                    hovermode="x unified", 
                    height=600,
                    legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
                )
                st.plotly_chart(fig, width='stretch')

            with tab3:
                st.divider()
                st.subheader("Risk Concentration & Diversification")
                # Unpack both the series and the new ratio
                risk_contrib, div_ratio = tester.calculateRiskContribution()
                m1, m2 = st.columns(2)
                m1.metric("Diversification Ratio", f"{div_ratio:.2f}", 
                        help="A ratio > 1.0 means your assets are offsetting each other's risks. Higher is better.")
                # Interpretation of the ratio
                if div_ratio < 1.1:
                    m2.error("Low Diversification: Assets are highly correlated.")
                elif div_ratio < 1.5:
                    m2.warning("Moderate Diversification.")
                else:
                    m2.success("Strong Diversification: Portfolio risk is significantly lower than individual asset risks.")
                c1, c2 = st.columns([2, 1])
                with c1:
                    # Plotting the Risk Contribution vs Allocation
                    comparison_df = pd.DataFrame({
                        "Asset Allocation (%)": [w * 100 for w in weights],
                        "Risk Contribution (%)": [r * 100 for r in risk_contrib]
                    }, index=tickers)
                    st.bar_chart(comparison_df)
                    
                with c2:
                    st.write("**Insight:**")
                    # Identify the "Risk Hog"
                    max_risk_asset = risk_contrib.idxmax()
                    risk_ratio = risk_contrib[max_risk_asset] / (1/len(tickers))
                    st.info(f"**{max_risk_asset}** is your primary risk driver, accounting for **{risk_contrib[max_risk_asset]*100:.1f}%** of total portfolio volatility.")
                    if risk_contrib[max_risk_asset] > (sum(weights)/len(tickers)) * 1.5:
                        st.warning("**Concentration Alert:** Your risk is heavily skewed toward one asset. Consider trimming this position to achieve 'Risk Parity'.")

            with tab4:
                st.subheader("Optimal Hedge Suggestion")
                risk_gap = stressvar95 - var95
                hedge_req = (risk_gap / base) * 100
                st.warning(f"To neutralise the unnecessary risk, consider reallocating **{hedge_req:.2f}%** to defensive/non-correlated assets.")

            with tab5:
                st.subheader("Raw Portfolio Data")
                raw_data = tester.closePrices.dropna()
                st.dataframe(raw_data, width='stretch')
                csv = raw_data.to_csv().encode('utf-8')
                st.download_button(label="Download Raw Data as CSV", data=csv, file_name=f"portfolio_data_{datetime.date.today()}.csv", mime="text/csv")
            
            risk_level = "High" if annualVolatility > 0.3 else "Moderate" if annualVolatility > 0.15 else "Low"
            main_threat = "Concentration" if risk_contrib.max() > 0.5 else "Systemic Market Risk"
            diversification_status = "Poor" if div_ratio < 1.2 else "Excellent"
            st.markdown(f"""
            * **Portfolio Risk Profile:** Your portfolio exhibits **{risk_level}** volatility. 
            * **Primary Vulnerability:** The biggest threat to your capital is **{main_threat}**.
            * **Diversification Quality:** Your diversification is **{diversification_status}** (Ratio: {div_ratio:.2f}).
            * **Stress Test Outcome:** In a severe market crash scenario, your portfolio could face an average tail loss of **${stress_cvar95:,.2f}** over the next {dayHorizon} days.
            """)
            if risk_contrib.max() > 0.6:
                st.success(f"**Actionable Advice:** Consider reducing your exposure to **{risk_contrib.idxmax()}** to lower your overall expected shortfall.")
                
else:
    st.warning("Select appropriate assets in the configuration bar to begin.")
