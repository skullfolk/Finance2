pip install streamlit
import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")

# Input layout
st.header("DCF Monte Carlo Valuation")
col1, col2, col3, col4 = st.columns(4)
with col1:
    initial_fcf = st.number_input("Initial Free Cash Flow ($M)", value=425.0, step=1.0, help="2025 FCF estimate")
    market_price = st.number_input("Market Price ($)", value=540.73, step=1.0, help="Current share price")
with col2:
    revenue = st.number_input("Total Revenue ($B)", value=2.7, step=0.1, help="2025 projected revenue")
    de_ratio = st.number_input("Debt-to-Equity Ratio", value=0.0182, step=0.0001, format="%f", help="AXON’s D/E ratio")
with col3:
    shares = st.number_input("Shares Outstanding (M)", value=76.67, step=0.1, help="Shares outstanding")
    rd = st.number_input("Cost of Debt (rd)", value=0.04, step=0.001, format="%f", help="Cost of debt")
with col4:
    net_cash = st.number_input("Net Cash ($M)", value=255.22, step=1.0, help="Cash minus debt")
    tc = st.number_input("Tax Rate (tc)", value=0.15, step=0.01, format="%f", help="Effective tax rate")

years = st.slider("Projection Years", 1, 20, 5, help="Explicit forecast period")
baseline_margin = st.number_input("Baseline EBIT Margin", value=0.08, step=0.01, format="%f", help="2025 EBIT margin")

# Validate inputs
if shares <= 0 or revenue <= 0:
    st.error("Shares and revenue must be positive.")
    st.stop()

# Sliders for assumptions
st.subheader("Assumptions")
col5, col6 = st.columns(2)
with col5:
    growth = st.slider("Mean FCF Growth Rate", 0.05, 0.4, 0.20, 0.01, help="Annual FCF growth")
    rf = st.slider("Risk-Free Rate", 0.02, 0.06, 0.04178, 0.001, help="10-year Treasury yield")
    erp = st.slider("Equity Risk Premium", 0.02, 0.06, 0.0406, 0.001, help="Market risk premium")
with col6:
    unlevered_beta = st.slider("Unlevered Beta", 0.6, 1.5, 1.0, 0.01, help="Industry beta")
    term_growth = st.slider("Terminal Growth Rate", 0.01, 0.06, 0.04, 0.001, help="Perpetual growth")
    margin_min = st.slider("Min EBIT Margin", 0.05, 0.15, 0.08, 0.01, help="Minimum projected margin")
    margin_max = st.slider("Max EBIT Margin", 0.10, 0.20, 0.12, 0.01, help="Maximum projected margin")

# Monte Carlo Simulation
simulations = 10000
values = []
wacc_values = []
beta_values = []

for i in range(simulations):
    g = np.random.normal(growth, 0.06)
    m = np.random.triangular(margin_min, (margin_min + margin_max) / 2, margin_max)
    r_f = np.random.normal(rf, 0.002)
    er_p = np.random.normal(erp, 0.005)
    beta_u = np.random.normal(unlevered_beta, 0.1)
    t_growth = np.random.triangular(0.03, term_growth, 0.045)

    beta_l = beta_u * (1 + (1 - tc) * de_ratio)
    beta_adj = 0.67 * beta_l + 0.33 * 1.0
    beta_values.append(beta_adj)
    debt_ratio = de_ratio / (1 + de_ratio)
    equity_ratio = 1 - debt_ratio
    re_cost = r_f + beta_adj * er_p
    wacc = equity_ratio * re_cost + debt_ratio * rd * (1 - tc)
    wacc_values.append(wacc)

    fcf_projection = initial_fcf
    total = 0
    for y in range(1, years + 1):
        fcf_projection *= (1 + g) * (1 + (m - baseline_margin) / baseline_margin * 0.1)
        total += fcf_projection / ((1 + wacc) ** y)

    spread = max(wacc - t_growth, 0.005)
    terminal_value = fcf_projection * (1 + t_growth) / spread
    terminal_pv = terminal_value / ((1 + wacc) ** years)

    equity_val = total + terminal_pv + net_cash
    intrinsic_val = equity_val / shares
    values.append(intrinsic_val)

# Analysis
mean_value = np.mean(values)
median_value = np.median(values)
ci_95 = np.percentile(values, [2.5, 97.5])
mean_wacc = np.mean(wacc_values)
ci_wacc_95 = np.percentile(wacc_values, [2.5, 97.5])
mean_beta = np.mean(beta_values)
ci_beta_95 = np.percentile(beta_values, [2.5, 97.5])
prob_above_market = np.mean(np.array(values) > market_price) * 100

# Display Results
st.title("💰 DCF Monte Carlo Valuation")
col7, col8 = st.columns(2)
with col7:
    st.metric("Mean Intrinsic Value/Share", f"${mean_value:.2f}")
    st.metric("Median Intrinsic Value/Share", f"${median_value:.2f}")
    st.write(f"95% CI for Value/Share: ${ci_95[0]:.2f} - ${ci_95[1]:.2f}")
with col8:
    st.metric("Market Price", f"${market_price:.2f}")
    st.metric("Probability Value > Market", f"{prob_above_market:.2f}%")
    st.metric("Mean WACC", f"{mean_wacc*100:.2f}%")
    st.write(f"95% CI for WACC: {ci_wacc_95[0]*100:.2f}% - {ci_wacc_95[1]*100:.2f}%")

# Visualizations
st.subheader("Valuation Distributions")
col9, col10 = st.columns(2)
with col9:
    fig, ax = plt.subplots()
    ax.hist(values, bins=50, color='skyblue', edgecolor='black')
    ax.axvline(market_price, color='red', linestyle='--', label=f'Market Price (${market_price:.2f})')
    ax.set_title('Intrinsic Value/Share Distribution')
    ax.set_xlabel('Intrinsic Value/Share ($)')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)

with col10:
    fig, ax = plt.subplots()
    ax.hist(wacc_values, bins=50, color='salmon', edgecolor='black')
    ax.axvline(0.0866, color='red', linestyle='--', label='Base WACC (8.66%)')
    ax.set_title('WACC Distribution')
    ax.set_xlabel('WACC')
    ax.set_ylabel('Frequency')
    ax.legend()
    st.pyplot(fig)

# Beta Visualization
st.subheader("Beta Distribution")
fig, ax = plt.subplots()
ax.hist(beta_values, bins=50, color='lightblue', edgecolor='black')
ax.axvline(1.13, color='red', linestyle='--', label='Base Beta (1.13)')
ax.set_title('Beta Distribution')
ax.set_xlabel('Beta')
ax.set_ylabel('Frequency')
ax.legend()
st.pyplot(fig)
