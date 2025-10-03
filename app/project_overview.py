import streamlit as st


st.header(" 🔎 Project Overview")
st.markdown("#### Data-driven strategies to maximize holiday & event sales performance")
st.write("""
This project analyzes **holiday and event-driven sales patterns** across Walmart store clusters 
(Flagship, Regional, Local, Rural) to identify risks (sales dips 📉) and opportunities (uplift 📈).
By simulating different **promotion strategies** and evaluate their 💰 **profitability & ROI**.
""")


# ---- Interactive Highlights ----
st.subheader("📊 Key Event Focus")
event = st.selectbox(
    "Choose an event to explore:",
    ["Thanksgiving & Black Friday (Uplift Opportunity)","Christmas + New Year (Dip Risk)", "Superbowl (Low Performance Uplift)"]
)

if event == "Thanksgiving & Black Friday (Uplift Opportunity)":
    st.success("👉 Strong uplift across most clusters, with **Flagship stores leading the surge.**")
    st.markdown("**Strategies:** tiered discounts, extended promos, VIP early access,loyalty programme etc.")
elif event == "Superbowl (Low Performance Uplift)":
    st.info("👉 Positive uplift in **all clusters**, but **Flagship stores show negative ROI.**")
    st.markdown("**Strategies:** snack/beverage tie-ins, “Superbowl Family Pack” for sensitive clusters, avoid flagship campaigns.")
elif event == "Christmas + New Year (Dip Risk)":
    st.warning("👉 Sales dip observed, especially in **Rural stores (-14.4%) during Christmas**. Risk mitigation is key.")
    st.markdown("**Strategies:** targeted digital campaigns, clearance bundles, gift-card pushes.")


st.subheader("💡 Try a Promo ROI Simulation")

baseline_sales = st.number_input("Baseline Sales ($)", value=650000, step=10000)
uplift = st.number_input("Predicted Uplift ($)", value=50000, step=5000)
promo_cost_pct = st.slider("Promo Cost (% of Baseline)", 1, 15, 3)    
    
    
promo_cost = baseline_sales * (promo_cost_pct / 100)
net_gain = uplift - promo_cost
roi = net_gain / promo_cost if promo_cost > 0 else 0 
    
col1, col2, col3 = st.columns(3)
col1.metric("📈 Predicted Uplift", f"${uplift:,.0f}")
col2.metric("💸 Promo Cost", f"${promo_cost:,.0f}")
col3.metric("🔑 ROI", f"{roi:.2f}x", delta=f"{net_gain:,.0f} Net Gain")




st.markdown("---")
st.write("This interactive dashboard helps decision-makers **compare strategies, simulate outcomes, and align promos with store sensitivity**.")
st.caption("Built with Streamlit · Revenue Optimization · Data-driven Consulting Demo")

