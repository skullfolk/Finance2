import streamlit as st
import numpy as np
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")  # Use wide layout

# Dropdown for company type
company_type = st.selectbox("Select Company Type", ["Profitable", "Non-profitable"])

# Input layout: 4 inputs per row
col1, col2, col3, col4 = st.columns(4)
with col1:
    initial_fcf = st.number_input("Initial Free Cash Flow ($M)", value=329.53, step=1.0)
    market_price = st.number_input("Market Price ($)", value=540.73, step=1.0)
with col2:
    revenue = st.number_input("Total Revenue ($B)", value=2.7, step=0.1)
    de_ratio = st.number_input("Debt-to-Equity Ratio", value=0.00152, step=0.0001, format="%f")
with col3:
    shares = st.number_input("Shares Outstanding (in millions)", value=76.67, step=0.1)
    rd = st.number_input("Cost of Debt (rd)", value=0.04, step=0.001, format="%f")
with col4:
    net_cash = st.number_input("Net Cash ($M)", value=255.22, step=1.0)
    tc = st.number_input("Tax Rate (tc)", value=0.1842, step=0.01, format="%f")

years = st.slider("Projection Years", 1, 20, 5)
revenue_per_share = revenue / shares

# Sliders for assumptions
growth = st.slider("Mean Revenue Growth Rate", 0.05, 0.4, 0.2, 0.01)
margin = st.slider("Mean Profit Margin", 0.05, 0.2, 0.10, 0.005)
rf = st.slider("Risk-Free Rate", 0.02, 0.06, 0.04178, 0.001)
erp = st.slider("Equity Risk Premium", 0.02, 0.06, 0.0406, 0.001)
unlevered_beta = st.slider("Unlevered Beta", 0.6, 1.5, 1.0, 0.01)
term_growth = st.slider("Terminal Growth Rate", 0.01, 0.06, 0.03, 0.001)
reinvestment_rate = st.slider("Reinvestment Rate (for Non-profitable)", 0.1, 1.0, 0.4, 0.05)

# Monte Carlo Simulation
simulations = 10000
values = []

for _ in range(simulations):
    g = np.random.normal(growth, 0.07)
    m = np.random.normal(margin, 0.01)
    r_f = np.random.normal(rf, 0.002)
    er_p = np.random.normal(erp, 0.005)
    beta_u = np.random.normal(unlevered_beta, 0.1)
    t_growth = np.random.normal(term_growth, 0.005)

    beta_l = beta_u * (1 + (1 - tc) * de_ratio)
    beta_adj = 0.67 * beta_l + 0.33 * 1.0
    debt_ratio = de_ratio / (1 + de_ratio)
    equity_ratio = 1 - debt_ratio
    re_cost = r_f + beta_adj * er_p
    wacc = equity_ratio * re_cost + debt_ratio * rd * (1 - tc)

    rev_proj = revenue
    total = 0
    for y in range(1, int(years) + 1):
        rev_proj *= (1 + g)
        if company_type == "Profitable":
            fcf = rev_proj * m  # Assume Operating CF - CapEx approximation
        else:
            op_income = rev_proj * m
            reinvestment = (rev_proj - revenue) * reinvestment_rate
            fcf = op_income * (1 - tc) - reinvestment
        total += fcf / ((1 + wacc) ** y)

    terminal_fcf = rev_proj * m if company_type == "Profitable" else (rev_proj * m) * (1 - tc) - (rev_proj - revenue) * reinvestment_rate
    spread = max(wacc - t_growth, 0.005)
    terminal_value = terminal_fcf * (1 + t_growth) / spread
    terminal_pv = terminal_value / ((1 + wacc) ** years)

    equity_val = total + terminal_pv + net_cash
    intrinsic_val = equity_val / shares
    values.append(intrinsic_val)

# Display Results
mean_value = np.mean(values)
ci_95 = np.percentile(values, [2.5, 97.5])
prob_above_market = np.mean(np.array(values) > market_price) * 100

st.title("💰 DCF Monte Carlo Interactive Valuation")
st.metric(label="Mean Intrinsic Value per Share", value=f"${mean_value:.2f}")
st.write(f"95% Confidence Interval: ${ci_95[0]:.2f} - ${ci_95[1]:.2f}")
st.write(f"Market Price: ${market_price:.2f}")
st.write(f"Probability Value > Market: {prob_above_market:.2f}%")

# Histogram
fig, ax = plt.subplots()
ax.hist(values, bins=50, color='skyblue', edgecolor='black')
ax.axvline(market_price, color='red', linestyle='--', label='Market Price')
ax.set_title('Distribution of Intrinsic Values')
ax.set_xlabel('Intrinsic Value per Share ($)')
ax.set_ylabel('Frequency')
ax.legend()
st.pyplot(fig)
