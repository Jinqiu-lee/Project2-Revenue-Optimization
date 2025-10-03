import streamlit as st  
import numpy as np  
import pandas as pd    
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import auc
import seaborn as sns
import pickle


st.header(" 🤖 Modeling & Evaluation Dashboard")
    

df = pd.read_pickle("data/processed_data.pkl")
impact_df = pd.read_pickle("data/impact_data.pkl")
# load models 
with open("model/blackfriday_thanksgiving_control.pkl","rb") as f:
    bf_control = pickle.load(f)
with open("model/blackfriday_thanksgiving_treat.pkl","rb") as f:
    bf_treat = pickle.load(f)
with open("model/superbowl_control.pkl","rb") as f:    
    spb_control =pickle.load(f)
with open("model/superbowl_treat.pkl","rb") as f:
    spb_treat = pickle.load(f)  
    

# create one-hot encoding 
holiday_dummies = pd.get_dummies(df['Holiday_Name'],prefix='Holiday')
df =  pd.concat([df,holiday_dummies],axis=1)
# create general holiday flag
df['Is_Holiday_Week'] = df['Holiday_Name'].notna().astype(int)
holiday_cols = [col for col in df.columns if 'Holiday_' in col]

if st.checkbox("🟡 Show processed dataframe"):
    st.dataframe(df)

st.markdown("""
🔘 This section evaluates holiday sales performance across clusters.
- **Black Friday + Thanksgiving** → Growth focus  
- **Christmas + New Year** → Risk mitigation  
- **Super Bowl** → Niche targeting  
""")

tabs = st.tabs([
    "🛍️ Black Friday + Thanksgiving",
    "🎄 Christmas + New Year",
    "🏈 Super Bowl",
    "📊 Cross-Holiday Comparison"
]) 

with tabs[0]:
    st.subheader("Black Friday + Thanksgiving (Growth Focus)")
    
    # Treatment = 1 if Black Friday OR Thanksgiving, else 0
    df["treatment_flag"] = (
        df["Holiday_Black Friday"].astype(int) | df["Holiday_Thanksgiving"].astype(int)
    )

    # Define features (X) and target (y)
    df = df.sort_values(['Store','Date'])
    df['sales_lag_1'] = df.groupby('Store')['Weekly_Sales'].shift(1)
    df['sales_lag_2'] = df.groupby('Store')['Weekly_Sales'].shift(2)
    df['sales_moving_avg_4'] = (
        df.groupby('Store')['Weekly_Sales']
        .rolling(window=4)
        .mean()
        .reset_index(level=0,drop=True))
    
    # Step 1: Define Features & Treatment
    features = [
        "Store", "Temperature", "Fuel_Price", "CPI", "Unemployment", 
        "Clusters", "sales_lag_1", "sales_lag_2", "sales_moving_avg_4"
    ]

    X = df[features]
    y = df['Weekly_Sales']
    t = df['treatment_flag'] # treatment assignment
    
    # Step 2: Split data
    X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X, y, t, test_size=0.3, random_state=42
    )
   
    # Step 4: Predict outcomes & compute uplift
    y_pred_control = bf_control.predict(X_test)
    y_pred_treat = bf_treat.predict(X_test)

    #Predcit all dataset
    y_pred_ctrl = bf_control.predict(X)
    y_pred_te = bf_treat.predict(X)

    # uplift = difference
    uplift = y_pred_treat - y_pred_control
    uplift_all = y_pred_te - y_pred_ctrl
    
    df_evaluation = X_test.copy()
    df_evaluation['Weekly_Sales'] = y_test.values
    df_evaluation['treatment_flag'] = t_test.values
    df_evaluation['uplift'] = uplift
    

    # Step 5: Qini Score Function

    def qini_score(df,uplift_col='uplift',treat_col='treatment_flag',outcome_col='Weekly_Sales'):
        df_sorted = df.sort_values(uplift_col, ascending=False).reset_index(drop=True)
        df_sorted["actual_te"] = np.where(df_sorted[treat_col]==1,
                                        df_sorted[outcome_col],
                                        0)
        
        cum_gain = df_sorted["actual_te"].cumsum()
        total_gain = cum_gain.iloc[-1]
        
        random_baseline = np.linspace(0, total_gain, len(df_sorted))
        cum_gain_oracle = df_sorted["actual_te"].sort_values(ascending=False).cumsum()
        x = np.arange(len(df_sorted)) / len(df_sorted)
        
        model_area = auc(x, cum_gain)
        random_area = auc(x, random_baseline)
        oracle_area = auc(x, cum_gain_oracle)

        qini = model_area - random_area
        qini_norm = qini / (oracle_area - random_area)
        return qini, qini_norm, x, cum_gain, random_baseline, cum_gain_oracle

    # Step 6 : Manual uplift Evaluation
    qini, qini_norm, x, cum_gain, random_baseline, oracle = qini_score(df_evaluation)

    st.write(f"Qini Score: {qini:,.0f}")
    st.write(f"Normalized Qini Score: {qini_norm:.3f}")
    
    # Step 7: Plot Qini Curve
    fig, ax = plt.subplots(figsize=(8,6))
    ax.plot(x, cum_gain, label="Model Uplift", color='blue')
    ax.plot(x, random_baseline, label="Random targeting (baseline)", linestyle="--", color="red")
    ax.plot(x, oracle, label="Oracle (perfect targeting)", linestyle=":", color="green")
    ax.set_xlabel("Proportion of population targeted")
    ax.set_ylabel("Cumulative incremental sales")
    ax.set_title("Qini Curve")
    ax.legend()
    st.pyplot(fig)
    
    with st.expander("Business Interpretation from qini curve:"):
        st.markdown("""
                    - Strong early uplift: When targeting the **top 20–30% of stores** ranked by predicted uplift, the blue curve rises much faster than the random baseline. The model effectively identifies the **most holiday-sensitive stores** that generate disproportionately higher incremental sales when promotions are applied.
                    - Diminishing Returns Beyond ~ 70%: After around 70% of stores targeted, the uplift curve flattens and approaches the random baseline. This indicates that beyond this point, promotions add little extra incremental revenue. In practice, lower-ranked stores is less cost-effective.
                    - 🧐 **Where to act** ? We should **target 🔝 30 high-uplift stores**
                    """)
    
    df['uplift']= uplift_all
    store_uplift = df.groupby('Store')['uplift'].mean().reset_index()
    store_uplift = store_uplift.sort_values('uplift',ascending=False)
    top30_stores = store_uplift.head(30)

    # cluster-level uplift
    cluster_uplift= df.groupby('Cluster_Category')['uplift'].mean().reset_index().sort_values('uplift',ascending=False)
    st.markdown("**🔝30 high-uplift stores:**")
    st.write(top30_stores)
    st.markdown("**Cluster uplift:**")
    st.write(cluster_uplift)
    with st.expander("⬆️ **Promotion Target:**"):
        st.markdown("""
                1. Focused Promotions:
                - Concentrate holiday campaigns on the top 20–40% uplift-ranked stores. These top 20-40 stores deliver the highest ROI. 
                - Avoid blanket promos as low-ranked stores since they don't provide significant incremental gains, it will dilutes profitability
                2. Pilot Test: 
                - Use the top-ranked stores to pilot new holiday campaigns
                - Their response can be used to refine future promotional strategies.
                3. Resource Allocation: Allocate more budget, stock, and marketing spend toward high-uplift clusters
                """)
    
    # Profitability estimation function
    def calculate_promo_profit(stores,promo_cost_per_store,margin_rate=0.3):
        df = stores.copy()
        # Gross profit = uplift * margin
        df['gross_profit'] = df['uplift']*margin_rate
        # Net profit = gross profit - promo cost
        df['net_profit']=df['gross_profit']-promo_cost_per_store
        
        # Create summary (aggregate)
        summary = {
            'Promo_Cost_Per_Store':promo_cost_per_store,
            "AVG_uplift": df["uplift"].sum(),
            "Total_Gross_Profit": df["gross_profit"].sum(),
            "Total_Net_Profit": df["net_profit"].sum(),
            "Total_Cost": len(df) * promo_cost_per_store,
            }
        
        return df, summary
    
    def compare_promo_scenarios(store_uplift,promo_cost_levels=[10000,20000,30000,40000,50000],margin_rate=0.3):
        results=[]
    
        for cost in promo_cost_levels:
            best_net_profit = -float('inf')
            best_n = None
            best_summary = None
        
            # Loop over possible top-n stores (1 to all stores)
            for n in range(1,len(store_uplift)+1):
                topN_stores = store_uplift.head(n)
                _,summary = calculate_promo_profit(topN_stores,promo_cost_per_store=cost,margin_rate=margin_rate)
                net_profit = summary['Total_Net_Profit']
                
                if net_profit > best_net_profit:
                    best_net_profit = net_profit
                    best_n = n
                    best_summary = summary
                    
                # Append best scenario for this promo cost
                # ROI = net profit / cost
            best_summary["Optimal_Stores"] = best_n
            best_summary["Promo_Cost_Per_Store"] = cost
            best_summary["ROI_%"] = (best_summary["Total_Net_Profit"] / best_summary["Total_Cost"]) * 100
            results.append(best_summary)
        
        scenario_df = pd.DataFrame(results)
        # Plot results
        fig,axes = plt.subplots(1, 2, figsize=(12,5))
        
        # --- Plot 1: Total Net Profit vs Promo Cost ---
        axes[0].plot(scenario_df["Promo_Cost_Per_Store"], 
             scenario_df["Total_Net_Profit"], 
             marker="o", label="Net Profit")
        axes[0].set_xlabel("Promo Cost per Store ($)")
        axes[0].set_ylabel("Total Net Profit ($)")
        axes[0].set_title("Net Profit vs Promo Cost per Store")
        axes[0].legend()
        
        # --- Plot 2: Optimal Stores vs Promo Cost ---
        axes[1].plot(scenario_df["Promo_Cost_Per_Store"], 
             scenario_df["Optimal_Stores"], 
             marker="s", color="orange", label="Optimal Stores")
        axes[1].set_xlabel("Promo Cost per Store ($)")
        axes[1].set_ylabel("Optimal Number of Stores")
        axes[1].set_title("Optimal Stores vs Promo Cost per Store")
        axes[1].legend()
        
        plt.tight_layout()
        st.pyplot(fig)
        
        return scenario_df
   
    scenarios = compare_promo_scenarios(store_uplift,promo_cost_levels=[10000,20000,30000,40000,50000],margin_rate=0.3)
    st.dataframe(scenarios)
    with st.expander("Insights from Optimal stores and ROI "):
        st.markdown("""
                    - Net profit is highest at a promo cost at 10K/store, generating $ 412K net profit across 45 stores.
                    - At lower promo costs ($ 10K–20K/store), the model selects more stores (39–45) as optimal.
                    - ROI is extremely high at $10K/store (~917%), but falls sharply as cost rises: This indicates that incremental spending per store delivers diminishing returns
                    - Lower-cost promotions ($ 10K–20K/store) deliver the best ROI and sustain higher net profits. At higher costs, even though gross profit remains stable, additional promo spending erodes margins
                    - Campaigns should focus on **🔝35–45 high-uplift stores**.
                    """)
    st.success("🎁 🆙 **Final Strategy for Thanksgiving & Black Friday** 👉")
    st.markdown("""
                1. 💰 **Run leaner promotions**: 
                - Prioritize ($ 10K–20K) per store budgets,focused on 🔝40 high-uplift stores.
                - VIP early access,loyalty programme for Flagstores 
                - Extended promos after holiday week
                - Tiered discounts for high profit margins and high purchase frequency
                2. 💵 **Focus investment**: 
                - Target the top 40 uplift stores, If budget allows, scale to $20K/store
                - Avoid exceeding $30K/store unless strategic branding goals outweigh ROI
                3. 🪬 **Ongoing Optimization**: 
                - Reassess uplift data post-campaign to validate model assumptions and refine store selection ➕ budget allocation.
                """)
    
with tabs[1]:
    st.subheader("Christmas + New Year (Risk Mitigation)")
    st.markdown("**📉 Observed Dips and Net Gain after Mitigated Sales by Cluster**")
    
    def evaluate_NY_Christmas(
        baseline_sales,
        observed_dips,   # negative_uplift_value (e.g., -0.144 for -14.4%)
        mitigation_levels = [0.03,0.06,0.10]   #  3%, 6%, 10%
    ):
        
        results =[]
        
        for cluster,base in baseline_sales.items():
            dip = observed_dips.get(cluster,0)
            dipped_sales = base * (1+dip)    # dip after the holiday (假期销售额下降)
            
            for mitigation in mitigation_levels:
                mitigation_sales = base * (1+dip+mitigation)  # 缓解销售（mitigation strategy）
                net_gain = mitigation_sales - dipped_sales  #Measures how much the mitigation strategy recovered compared to doing nothing.
                
                results.append({
                    "Cluster":cluster,
                    "Baseline Sales":base,
                    "Observed Dip":f"{dip*100:.1f}%",
                    "Mitigation Level":f"{mitigation*100:.0f}%",
                    "Sales After Dip":round(dipped_sales,2),
                    "Mitigated Sales":round(mitigation_sales,2),
                    "Net Gain":round(net_gain,2)
                })
                
        NY_Chris_summary_df = pd.DataFrame(results)
        return NY_Chris_summary_df
           
    baseline_sales = (
        df.groupby("Cluster_Category")["Weekly_Sales"]
        .mean()      # or .median() if that's your baseline definition
        .to_dict()
    )

    observed_dips = {
        "Rural store": -0.144, # keep the worst dip (–14.4%) because Rural is most vulnerable
        "Local store": -0.031,   # exaggerated dip to –7% (vs –3.1%) for conservative planning.
        "Regional store": -0.077,  # smoothed down to –5% (from –7.7%).
        "Flagstore": -0.086  # choose 5%(instead of 8.6%)since flagships usually can absorb losses better.
    }

    
    NY_Christmas_summary = evaluate_NY_Christmas(baseline_sales,observed_dips)
    # joblib.dump(NY_Christmas_summary,"model/mitigation_christmas_newyear.pkl")

    
    st.dataframe(NY_Christmas_summary)
    st.markdown("""
                    **Baseline dip: Without action, sales decline in**
                    - Flagstore 📉8.6%
                    - Regional store 📉7.7%
                    - Local store decline 📉3.1%
                    - Rural store 📉14.4%
                    """)
    st.warning("🔺🔄 **Mitigation Strategy Options for Christmas & New Year** 👉")
    st.markdown("""
                ⓵ 🔷 **Light (3–6%)**: modest uplift, low cost, such as:
                - loyalty boost, gift-card pushes, prioritize promotions before holidays
                - Targeted ads, digital campaigns such as Photo Booth Corner with Christmas/New Year props (drives social media buzz)
                - Holiday events, indoor demos such as Mini Workshops (e.g., cookie decorating, DIY champagne cocktails)
                - Clearance bundles,limited holiday bundle editions: showcase products in action → directly upsell bundled kits
                
                🔹 Use **3% in Local store**(-3.1%).
                
                🔹 Use **6% in flagstore(-8.6%) and Regional store(-7.7%)**.
                
                ⓶ 🔶 **Medium-to-High(10%)**: stronger uplift, medium cost, such as :
                - Discounts, holiday bundles
                - Delivery promos,online campaigns such as partner with local miro-influencers
                - Gift-Centric Marketing ➡️ Medium-budget FB/Instagram carousel ads showing "Top 10 Christmas Gifts Under $50".
                
                🔸 Use heavy mitigation **10% only in critical clusters (Rural stores only Christmas)**. 
                
                🔹 Use 3% for Rural store New year
                """)
  
  
with tabs[2]:
     st.subheader("Super Bowl (Niche Targeting)")
     st.markdown("**🏈 Cluster Sensitivity to Super Bowl**")
     
     # Uplift Model 2 : Super Bowl Promo uplift 
     # Treatment = 1 if Super Bowl, else 0
     risk_df = df.copy()
     risk_df["treatment_flag"] = risk_df["Holiday_Super Bowl"].astype(int)

     # Define features (X) and target (y)
     risk_df = risk_df.sort_values(['Store','Date'])
     risk_df['sales_lag_1'] = risk_df.groupby('Store')['Weekly_Sales'].shift(1)
     risk_df['sales_lag_2'] = risk_df.groupby('Store')['Weekly_Sales'].shift(2)
     risk_df['sales_moving_avg_4'] = (
        risk_df.groupby('Store')['Weekly_Sales']
        .rolling(window=4)
        .mean()
        .reset_index(level=0,drop=True))
     risk_df['dynamic_baseline']= risk_df['sales_moving_avg_4'].fillna(risk_df['Weekly_Sales'])


    # Step 1: Define Features & Treatment
     features = [
        "Store", "Temperature", "Fuel_Price", "CPI", "Unemployment", 
        "Clusters", "sales_lag_1", "sales_lag_2", "sales_moving_avg_4"
    ]

     X = risk_df[features]
     y = risk_df['Weekly_Sales']
     t = risk_df['treatment_flag'] # treatment assignment

    # Step 2: Split data
     X_train, X_test, y_train, y_test, t_train, t_test = train_test_split(
        X, y, t, test_size=0.2, random_state=42
    )
   
    # Step 4: Predict outcomes & compute uplift
     y_pred_ctrl = spb_control.predict(X_test)
     y_pred_te = spb_treat.predict(X_test)

    #Predcit all dataset
     y_pred_ctrl_alldata = spb_control.predict(X)
     y_pred_te_alldata = spb_treat.predict(X)

    # uplift for superbowl
     uplift_superbowl = y_pred_te - y_pred_ctrl
     uplift_all_superbowl = y_pred_te_alldata - y_pred_ctrl_alldata

     evaluate_superbowl = X_test.copy()
     evaluate_superbowl['Weekly_Sales'] = y_test.values
     evaluate_superbowl['treatment_flag'] = t_test.values
     evaluate_superbowl['uplift_superbowl'] = uplift_superbowl
     
     

    # Step 5: Qini Score Function
     def qini_score(risk_df,uplift_col='uplift_superbowl',treat_col='treatment_flag',outcome_col='Weekly_Sales'):
         df_risk = risk_df.sort_values(uplift_col, ascending=False).reset_index(drop=True)
         df_risk["actual_te"] = np.where(df_risk[treat_col]==1,
                                        df_risk[outcome_col],
                                        0)
        
         cum_gain_spb = df_risk["actual_te"].cumsum()
         total_gain_spb = cum_gain_spb.iloc[-1]
        
         random_baseline_spb = np.linspace(0, total_gain_spb, len(df_risk))
         cum_gain_oracle = df_risk["actual_te"].sort_values(ascending=False).cumsum()
         x_spb = np.arange(len(df_risk)) / len(df_risk)
        
         model_area_spb = auc(x_spb, cum_gain_spb)
         random_area_spb = auc(x_spb, random_baseline_spb)
         oracle_area_spb = auc(x_spb, cum_gain_oracle)

         qini_spb = model_area_spb - random_area_spb
         qini_norm_spb = qini_spb / (oracle_area_spb - random_area_spb)
        
         return qini_spb, qini_norm_spb, x_spb, cum_gain_spb, random_baseline_spb, cum_gain_oracle

    # Step 6 : Manual uplift Evaluation
     qini_spb, qini_norm_spb, x_spb, cum_gain_spb, random_baseline_spb, oracle_spb = qini_score(evaluate_superbowl)

     st.write(f"Qini Score for Super Bowl: {qini_spb:,.0f}")
     st.write(f"Normalized Qini Score for Super bowl: {qini_norm_spb:.3f}")

     fig,ax = plt.subplots(figsize=(8,6))
     ax.plot(x_spb, cum_gain_spb,label="Super Bowl Model Uplift", color='purple')
     ax.plot(x_spb, random_baseline_spb, label="Random targeting (baseline)", linestyle="--", color="red")
     ax.plot(x_spb, oracle_spb, label="Oracle (perfect targeting)", linestyle=":", color="green")
     ax.set_xlabel("Proportion of population targeted")
     ax.set_ylabel("Cumulative incremental sales for superbowl")
     ax.set_title("Qini Curve Super Bowl")
     ax.legend()
    
     st.pyplot(fig)

    # Target top 15-20 most sensitive Super bowl stores 
     risk_df['uplift_spb']= uplift_all_superbowl
     store_uplift_spb = risk_df.groupby('Store')['uplift_spb'].mean().reset_index().sort_values('uplift_spb',ascending=False)
     top_stores_spb = store_uplift.head(15)

    # cluster-level uplift
     cluster_uplift_spb= risk_df.groupby('Cluster_Category')['uplift_spb'].mean().reset_index().sort_values('uplift_spb',ascending=False)
     st.write("**Top15 high-uplift stores for Superbowl:**")
     st.dataframe(top_stores_spb)
     st.write("**Cluster uplift for super bowl:**")
     st.dataframe(cluster_uplift_spb)
     
     with st.expander("Business Interpretation from plot"):
         st.markdown("""
                     Based on the qini curve, **target 🔝15 stores**, the blue curve rises much faster than the random baseline.
                     1. Estimates uplift sales (per cluster).
                     2. Apply different promotion cost levels (% of baseline sales or flat costs).
                     3. Calculate Net Gain and ROI vs. Doing Nothing
                     4. Identify which cluster gets the highest ROI at which promo level.
                     """)
    # Profitbality Estimation Function
     def superbowl_profitability(cluster_uplift,baseline_sales,promo_cost_levels=[0.02,0.03,0.04,0.05]):
         results = []
        
         for cluster,uplift in cluster_uplift.items():
             base_sales = baseline_sales.get(cluster,0)
            
             for promo_pct in promo_cost_levels:
                 promo_cost = base_sales * promo_pct   # investment cost
                 net_gain_spb = uplift - promo_cost
                 roi = (net_gain_spb / promo_cost) if promo_cost > 0 else 0
                
                 results.append({
                    "Cluster":cluster,
                    "Baseline Sales":base_sales,
                    "Predicted Uplift":uplift,
                    "Promo Cost %":f"{int(promo_pct*100)}%",
                    "Net Gain vs Do Nothing":net_gain_spb,
                    "ROI":roi
                })
                
         df_superbowl = pd.DataFrame(results)
         return df_superbowl

     cluster_uplift_spb['uplift_spb'] = pd.to_numeric(cluster_uplift_spb['uplift_spb'],errors='coerce')
     cluster_uplift = dict(zip(cluster_uplift_spb['Cluster_Category'],cluster_uplift_spb['uplift_spb']))


     df_superbowl = superbowl_profitability(cluster_uplift,baseline_sales)


     def plot_superbowl_roi(df_superbowl):
        clusters = df_superbowl['Cluster'].unique()
        fig, ax = plt.subplots(figsize=(10,6))
        
        for cluster in clusters:
            cluster_df = df_superbowl[df_superbowl['Cluster']==cluster]
            
            # x = promo cost % (as numbers instead of strings for plotting)
            x = cluster_df["Promo Cost %"].str.replace("%","").astype(int)
            y = cluster_df['ROI']
            
            ax.plot(x,y,marker="o",label=cluster)
            
            # Annotate best ROI point
            best_idx = y.idxmax()
            ax.text(x.loc[best_idx],y.loc[best_idx],
                    f"Best ROI:{y.loc[best_idx]:.2f}",fontsize=9,color='black')
            
        ax.axhline(0,color='grey',linestyle="--")
        ax.set_title("SuperBowl ROI by cluster and promo level")
        ax.set_xlabel("Promo Cost Level (%)")
        ax.set_ylabel("ROI (Net gain/cost)")
        ax.legend(title="Cluster")
        ax.grid(True,alpha=0.3)
        plt.tight_layout()
        
        st.pyplot(fig)
            
     plot_superbowl_roi(df_superbowl)
     st.markdown("**🏈 Net Gain & ROI with different promo cost per cluster**")
     st.dataframe(df_superbowl)
     st.info("**🏈🆙 Final Strategy for Super Bowl** 👉")
     st.markdown("""
                 🎯 Focus on **🔝15 Superbowl sensitive stores** to apply for strategies. Some specific stores (13, 10, 14, 4, 27) show very high absolute uplift (>$640k each), they're tactical targets for heavy campaigns. 
                 
                 - 🎉 Party Bundles: snack + beverage packs, “Superbowl Family Pack” at attractive price points.
                 - 💰 Micro-promotions: run short-term buy 2, get 1 free for high-consumption SKUs (chips, soda, beer).
                 - 🔀 Cross-sell Essentials: promote grilling meat, condiments, drinks → align with Superbowl gatherings.
                 - 📺 Localized Campaigns for Local store: regional TV/IG/FB ads or targeted geo-fenced digital ads before the game.
                 - ⛽️ Fuel/Travel tie-in for Rural store: partner discounts (gas coupons with Superbowl bundles).
                 - 🪬 Click & Collect: promote “order online, pick up before game day.” Convenience + assurance of stock
                 - 🏬 In-store displays: heavy Superbowl branding, cross-merchandising aisles.
                 
                 ①🔴 **5% promo_cost** for (13, 10, 14, 4, 27) stores.
                 
                 ②🔵 **3% promo_cost** for other Local store(+43k uplift) ➡️ Event-driven, looking for convenience and party-ready packs.
                 
                 ③🟣 **3% promo_cost** for Rural store(+32k uplift) ➡️ Rural shoppers may be more price sensitive, travel less, and buy in bulk.
                 
                 ④🟠 **2% promo_cost** for Regional store(+39k uplift) ➡️ medium-density areas where promotions will work but at a more moderate scale.
     
                 ⑤❎ No promo: Flagstore: –25k (negative uplift) → promotions here could actually cannibalize sales or are ineffective. Avoid promos or use experimential marketing instead 
                 """)
     
     
with tabs[3]:
    
    st.subheader("📊 Cross-Holiday Impact Summary")
    impact_df =pd.read_pickle("data/impact_data.pkl")
    
    summary_data = {
    "Holiday": ["Black Friday", "Thanksgiving", "Christmas", "New Year", "Super Bowl"],
    "Avg Uplift %": [37.6, 37.6, -8.3, -4.9, 3.3],
    "Avg Dip %": [0, 0, -8.3, -4.9, 0],
    "Best Cluster": ["Local store", "Local store", "Local store", "Rural store", "Flagstore"]
}
    summary_df = pd.DataFrame(summary_data)
    st.subheader("📌 Key Metrics Comparison")
    st.dataframe(summary_df)
    
    col1, col2 = st.columns(2)
    
    # Bar chart
    with col1:
        fig,ax = plt.subplots(figsize=(6,4))
        sns.barplot(data=summary_df,x="Holiday",y="Avg Uplift %", ax=ax, palette="viridis")
        ax.set_title("Average Uplift % by Holiday")
        ax.set_ylabel("Uplift (%)")
        plt.xticks(rotation=30,ha="right")
        st.pyplot(fig)

    with col2:
        pivot_data = summary_df.pivot(index="Holiday", values="Avg Uplift %", columns="Best Cluster").fillna(0)
        fig2, ax2 = plt.subplots(figsize=(6,4))
        sns.heatmap(pivot_data, annot=True, fmt=".1f", cmap="RdBu_r", center=0, cbar_kws={"label": "Impact %"}, ax=ax2)
        ax2.set_title("Cluster × Holiday Impact (%)")
        st.pyplot(fig2)
        
    # --- Strategic Insights ---
    st.subheader("💡 Strategic Insights")
    st.markdown("""
    1. **🛍️Black Friday & Thanksgiving**  
    - Highest uplift → prioritize **growth modeling**.  
    - Focus: Maximize promos in **Flagship & Regional stores**.  

    2. **🎄Christmas & New Year**  
    - Negative dips observed → prioritize **mitigation modeling**.  
    - Focus: counter-strategies like bundles, shifting promo budget earlier.  

    3. **🏉Super Bowl**  
    - Niche uplift → **targeted campaigns only** in **Regional clusters**.  
    - Focus: localized sports-driven promotions.  
    """)


    # --- Transition to Modeling ---
    st.success("""
    ✅ Next Step: We have built  **holiday-specific models** —  
    - Uplift models for **growth holidays (Black Friday & Thanksgiving)**  
    - Mitigation models for **risk holidays (Christmas & New Year)**  
    - Targeted segmentation for **Super Bowl**
    """)     
    


    
    