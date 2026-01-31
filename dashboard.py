import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import seaborn as sns
from InvestmentPortfolioStressTester import PortfolioStressTester

st.set_page_config(page_title="Dashboard", layout="wide")
st.title("Investment Portfolio Stress Tester")
st.markdown("""
This dashboard using **Multivariate Monte Carlo Simulations** to assess portfolio risk.
""")

# Sidebar
st.sidebar.header("Portfolio setup")
ticker = st.sidebar.text_input("Tickers (comma separated)", "GOOG, NVDA, AVGO, BTC-USD")
base = st.sidebar.number_input("Initial investment (USD$)", value=23270)

# Weight sliders
st.sidebar.header("Asset weights")
tickers = [t.strip() for t in ticker.split(",")]
weights = []
for t in tickers:
    w = st.sidebar.slider(f"Weights sum to {sum(weights)*100:.0f}%")
    weights = [w/sum(weights) for w in weights]


