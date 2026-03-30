import datetime
import os
import streamlit as st
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
import requests
from DataPipeline import MarketDataPipeline

@st.cache_data(ttl=600, show_spinner=False)
def get_cached_market_data(tickers_tuple, start_date, end_date):
    try:
        api_key = st.secrets.get("api_keys", {}).get("ALPACA_API_KEY")
        api_secret = st.secrets.get("api_keys", {}).get("ALPACA_API_SECRET")
    except Exception:
        api_key = None
        api_secret = None
    
    # Pass them to the pipeline
    pipeline = MarketDataPipeline(list(tickers_tuple), start_date, end_date, api_key, api_secret)
    return pipeline.fetch_data()

st.set_page_config(page_title="Risk Engine", layout="wide")

with st.container():
    st.title("Investment Portfolio Risk Analyser", anchor=False)

# Upper configuration bar
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
        lookback_options = ["1M", "3M", "6M", "1Y", "3Y", "5Y"]
        selection = st.segmented_control("Historical Data Lookback Period", options=lookback_options, default="3Y", width='stretch')
    with col4:
        base = st.number_input("Initial Capital (USD$)", value=23270)
    with col5:
        dayHorizon = st.selectbox("Forecast Horizon (Days)", [30, 60, 90, 180, 365], index=0)

    today = datetime.date.today()
    mapping = {
        "1M": today - datetime.timedelta(days=30),
        "3M": today - datetime.timedelta(days=90),
        "6M": today - datetime.timedelta(days=182),
        "1Y": today - datetime.timedelta(days=365),
        "3Y": today - datetime.timedelta(days=3*365),
        "5Y": today - datetime.timedelta(days=5*365),
    }
    st.session_state.start_date = mapping[selection]
    finalStart = st.session_state.start_date
    finalEnd = today

# Sidebar for stress test configuration
with st.sidebar:
    st.header("Factors Configuration")
    shock_vol = st.slider("Volatility Multiplier", 1.0, 4.0, 2.0, help="Amplifies historical price swings. 2.0x means the market becomes twice as jittery as usual.")
    mkt_gap = st.slider("Overnight Gap Down (%)", -50.0, 0.0, -5.0, 1.0, help="Simulates an immediate market price drop (x% gap down) before the simulation starts.")/100
    mean_shock = st.slider("Annualized Negative Drift", -1.0, 0.0, -0.20, 0.05, help="Simulates a sustained bear market trend.")
    simulations = st.select_slider("Simulation Timesteps", options=[500, 1000, 1500, 2000, 3000], value=1500)
    rebalance_toggle = st.checkbox("Enable Daily Rebalancing", value=False)

# Main analysis engine
if tickers:
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

    # We use session state to remember if the user has loaded the data. 
    # If they tweak a slider, Streamlit reruns, sees data_loaded is True, and instantly recalculates 
    # the math without requiring them to click the button again.
    button_clicked = st.button("Load Data & Run Engine", width='stretch')
    
    # Reset state if tickers or dates change so we fetch fresh data
    current_config_hash = hash((tuple(tickers), finalStart, finalEnd))
    if st.session_state.get('config_hash') != current_config_hash:
        st.session_state['data_loaded'] = False
        st.session_state['config_hash'] = current_config_hash

    if button_clicked:
        st.session_state['data_loaded'] = True

    if st.session_state.get('data_loaded', False):
        start = finalStart.strftime("%Y-%m-%d")
        end = finalEnd.strftime("%Y-%m-%d")
        
        # Fetch the data securely and instantly via Cache
        with st.spinner("Fetching Market Data..."):
            try:
                clean_data = get_cached_market_data(tuple(tickers), start, end)
            except Exception as e:
                st.error(f"Data Fetch Error: {e}")
                st.stop()

        # Send the cached data to the backend math engine
        with st.spinner("Running Risk Engine..."):
            # Make a copy and format the datetime index as strings for JSON.
            df_for_json = clean_data.copy()
            df_for_json.index = df_for_json.index.astype(str)
            payload = {
                "tickers": tickers,
                "weights": weights,
                "base": base,
                "start_date": start,
                "end_date": end,
                "day_horizon": dayHorizon,
                "simulations": simulations,
                "shock_volatility": shock_vol,
                "market_gap": mkt_gap,
                "mean_shock": mean_shock,
                "rebalance": rebalance_toggle,
                "market_data": df_for_json.to_dict()
            }
            API_URL = os.getenv("API_URL", "http://localhost:8000/api/v1/simulate")
            try:
                response = requests.post(API_URL, json=payload)
                # Check if the API sent back a 400 or 500 error explicitly
                if response.status_code != 200:
                    # Extract our beautifully formatted error message from the JSON
                    error_msg = response.json().get("detail", "Unknown API Error")
                    st.error(f"**Engine Error:** {error_msg}")
                    st.stop()
                data = response.json()
            except requests.exceptions.ConnectionError:
                st.error("**Connection Error:** Could not connect to the FastAPI backend. Is `api.py` running in another terminal?")
                st.stop()
            except requests.exceptions.RequestException as e:
                st.error(f"**Network Error:** {e}")
                st.stop()
        # Convert the JSON lists back into NumPy arrays
        general = np.array(data["normal_simulation"])
        crash = np.array(data["stress_simulation"])

        history_df = pd.DataFrame(data["historical_prices"])
        history_df.index = pd.to_datetime(history_df.index)
        corr_matrix = pd.DataFrame(data["correlation_matrix"])
        risk_contrib = pd.Series(data["risk_contribution"])
        div_ratio = data["diversification_ratio"]

        final_general = general[-1, :]
        final_crash = crash[-1, :]
        var95 = base - np.percentile(final_general, 5)
        stressvar95 = base - np.percentile(final_crash, 5)
        
        tail_general = final_general[final_general <= np.percentile(final_general, 5)]
        cvar95 = base - np.mean(tail_general)
        tail_crash = final_crash[final_crash <= np.percentile(final_crash, 5)]
        stress_cvar95 = base - np.mean(tail_crash)

        risk_free_rate = 0.0365
        path_returns_general = ((final_general / base) - 1) * (252 / dayHorizon)
        path_returns_crash = ((final_crash / base) - 1) * (252 / dayHorizon)
        downside_general = np.minimum(0, path_returns_general - risk_free_rate)
        downside_crash = np.minimum(0, path_returns_crash - risk_free_rate)
        downside_vol_general = np.sqrt(np.mean(downside_general**2))
        downside_vol_crash = np.sqrt(np.mean(downside_crash**2))
        median_return_general = np.median(path_returns_general)
        median_return_crash = np.median(path_returns_crash)
        excess_gen = median_return_general - risk_free_rate
        excess_crash = median_return_crash - risk_free_rate
        sortinoGeneral = (excess_gen / (downside_vol_general + 1e-9)) if excess_gen > 0 else (excess_gen * downside_vol_general)
        sortinoCrash = (excess_crash / (downside_vol_crash + 1e-9)) if excess_crash > 0 else (excess_crash * downside_vol_crash)

        base_col1, base_col2, base_col3 = st.columns(3)
        with base_col1:
            st.metric("Normal VaR (95%)", f"${var95:,.2f}", help="The minimum loss expected in the worst 5% of normal outcomes.")
        with base_col2:
            st.metric("Normal CVaR (Tail Risk)", f"${cvar95:,.2f}", help="The average loss if you land in the worst 5% of normal outcomes.")
        with base_col3:
            st.metric("Normal Sortino Ratio", f"{sortinoGeneral:.2f}", help="Measures risk-adjusted return, but only penalizes downside volatility (crashes), ignoring upside spikes.")
        
        st.divider()
        
        stress_col1, stress_col2, stress_col3 = st.columns(3)
        with stress_col1:
            var_delta = f"{((stressvar95 / (var95 + 1e-9)) - 1) * 100:.1f}% vs Normal" if var95 > 0.01 else "New Risk"
            st.metric("Stress VaR", f"${stressvar95:,.2f}", delta=var_delta, delta_color="inverse", help="VaR when Volatility Multipliers and Gaps are applied.")
        with stress_col2:
            cvar_delta = f"{((stress_cvar95 / (cvar95 + 1e-9)) - 1) * 100:.1f}% vs Normal" if cvar95 > 0.01 else "New Risk"
            st.metric("Stress CVaR", f"${stress_cvar95:,.2f}", delta=cvar_delta, delta_color="inverse", help="CVaR when Volatility Multipliers and Gaps are applied.")
        with stress_col3:
            st.metric("Stress Sortino Ratio", f"{sortinoCrash:.2f}", delta=f"{sortinoCrash - sortinoGeneral:.2f}", help="How efficiently the portfolio handles downside risk under severe stress.")

        tab1, tab2, tab3, tab4, tab5 = st.tabs(["Historical Trends", "Probabilistic Forecast", "Risk Attribution", "Hedging", "Raw Market Data"])
        with tab1:
            st.subheader("Historical Asset Performance (Base 100)")
            # Use the newly parsed history_df instead of tester.closePrices
            normalised_prices = (history_df/history_df.iloc[0])*100
            historyFigure = go.Figure()
            for asset in normalised_prices.columns:
                historyFigure.add_trace(go.Scatter(x=normalised_prices.index, y=normalised_prices[asset], name=asset, mode='lines', hovertemplate='%{y:.2f} (Index)'))
            historyFigure.update_layout(template='plotly', xaxis_title='Timeline', yaxis_title="Relative Growth (Base 100)", hovermode='x unified', height=600, legend=dict(orientation='h', yanchor='bottom', y=1.02, xanchor='right', x=1), margin=dict(l=0, r=0, t=0, b=0))
            st.plotly_chart(historyFigure, width='stretch')

        with tab2:
            st.subheader("Forecast Engine")
            failure_rate = np.mean(general[-1, :] < base) * 100
            crash_failure_rate = np.mean(crash[-1, :] < base) * 100
            col_f1, col_f2 = st.columns(2)
            col_f1.metric("Probability of Loss (Normal)", f"{failure_rate:.1f}%")
            col_f2.metric("Probability of Loss (Stress)", f"{crash_failure_rate:.1f}%", delta=f"{crash_failure_rate - failure_rate:.1f}%", delta_color="inverse")
            
            time_steps = np.arange(general.shape[0])
            fig = go.Figure()
            fig.add_shape(type="line", line=dict(color="white", width=2, dash="dot"), x0=0, x1=dayHorizon, y0=base, y1=base, name="Initial Principal")
            
            gen_median = np.median(general, axis=1)
            fig.add_trace(go.Scatter(x=time_steps, y=np.percentile(general, 95, axis=1), mode='lines', line=dict(color='rgba(0,0,255,0.2)', width=1), showlegend=False, hoverinfo='skip'))
            fig.add_trace(go.Scatter(x=time_steps, y=np.percentile(general, 5, axis=1), fill='tonexty', fillcolor='rgba(0,0,255,0.1)', name='Normal 90% Confidence'))
            fig.add_trace(go.Scatter(x=time_steps, y=gen_median, line=dict(color='blue', width=3), name='General Median Outcome'))
            
            fig.add_trace(go.Scatter(x=time_steps, y=np.median(crash, axis=1), line=dict(color='red', width=3, dash='dash'), name='Stress Median Outcome'))
            fig.add_trace(go.Scatter(x=time_steps, y=np.percentile(crash, 5, axis=1), fill='tonexty', fillcolor='rgba(255,0,0,0.2)', name='Extreme Tail Risk (5th Pct)'))
            fig.update_layout(title="Simulated Portfolio Value Paths", xaxis_title="Days into Future", yaxis_title="Portfolio Value ($)", template="plotly_white", hovermode="x unified", height=600, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
            st.plotly_chart(fig, width='stretch')

        with tab3:
            st.subheader("Risk Concentration & Diversification")
            m1, m2 = st.columns(2)
            m1.metric("Diversification Ratio", f"{div_ratio:.2f}")
            if div_ratio < 1.1: m2.error("Low Diversification: Assets are highly correlated.")
            elif div_ratio < 1.5: m2.warning("Moderate Diversification.")
            else: m2.success("Strong Diversification.")
            c1, c2 = st.columns([2, 1])
            with c1:
                comparison_df = pd.DataFrame({"Asset Allocation (%)": [w * 100 for w in weights], "Risk Contribution (%)": [r * 100 for r in risk_contrib]}, index=tickers)
                st.bar_chart(comparison_df)
            with c2:
                max_risk_asset = risk_contrib.idxmax()
                st.info(f"**{max_risk_asset}** is your primary risk driver, accounting for **{risk_contrib[max_risk_asset]*100:.1f}%** of total volatility.")
            st.divider()
            st.subheader("Asset Correlation Matrix")
            fig_corr, ax_corr = plt.subplots(figsize=(10, 6))
            # Use the parsed correlation matrix
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, center=0, ax=ax_corr, fmt=".2f", linewidths=.5)
            st.pyplot(fig_corr)

        with tab4:
            st.subheader("Optimal Hedge Suggestion")
            risk_gap = stressvar95 - var95
            hedge_req = (risk_gap / base) * 100
            st.warning(f"To neutralise the unnecessary risk, consider reallocating **{hedge_req:.2f}%** to defensive/non-correlated assets.")

        with tab5:
            st.subheader("Raw Portfolio Data")
            st.dataframe(history_df, width='stretch')
            csv = history_df.to_csv().encode('utf-8')
            st.download_button(label="Download CSV", data=csv, file_name=f"portfolio_data_{datetime.date.today()}.csv", mime="text/csv")
            
else:
    st.warning("Select appropriate assets in the configuration bar to begin.")