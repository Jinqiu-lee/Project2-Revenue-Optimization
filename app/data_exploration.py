import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
from xgboost import XGBRegressor
from sklearn.model_selection import train_test_split
from xgboost import plot_importance
from fredapi import Fred
from prophet.serialize import model_from_json

# load model and forecast dataframe
with open("model/prophet_sales.json", "r") as f:
    prophet_model = model_from_json(f.read())
forecast = pd.read_csv("model/prophet_forecast.csv")
df_prophet = pd.read_csv("model/df_prophet.csv")
future = pd.read_csv("model/df_future.csv")
model_xgb = XGBRegressor()
model_xgb.load_model("model/model_xgb.json")

#fred = Fred(api_key="2991e539c6491edffa341a08ad95a396")

@st.cache_data
def load_data():
    df = pd.read_csv("./data/Walmart.csv")
    df['Date'] = pd.to_datetime(df['Date'], format="%d-%m-%Y")
    df_holiday = pd.read_csv("./data/Walmart Holiday Weekly Sales.csv")
    df_holiday['Date'] = pd.to_datetime(df_holiday['Date'], format="%d-%m-%Y")
    df_non_holiday = pd.read_csv("./data/Walmart Non Holiday Weekly Sales.csv")
    df_non_holiday['Date'] = pd.to_datetime(df_non_holiday['Date'], format="%d-%m-%Y")
    macro_data = pd.read_pickle("./data/macro_data.pkl")
    
    return df, df_holiday,df_non_holiday,macro_data

df,df_holiday, df_non_holiday,macro_data = load_data()


st.subheader(" 🎬 Data Exploring ")
if st.checkbox("Show Walmart Weekly Sales data"):
    st.write("##### Data Shape:",df.shape)
    st.dataframe(df)

    
with st.expander("🧠 What did you find from Walmart dataframe ?"):
    st.markdown("""
                What factors may affect sales ?
                - Holiday_flag: Often cause spikes in sales
                - Tempertature: affect seasonal products
                - Fule_Price: impact customers' disposal income & shopping trips
                - CPI & Unemployment : CPI-customers price index, with unemployment,indicate the economic enviornment ,affecting spending 
                """)
    
# Data exploration 1 - Seasonal patterns 

fig,ax = plt.subplots(figsize=(20,8))
sns.lineplot(data=df,x='Date',y='Weekly_Sales')
ax.set_title("Weekly Sales Over Time")
st.write(" ##### 1️⃣ ♻️ Compare Seasonal Patterns with Weekly Sales")
st.pyplot(fig)
with st.expander("Insights from **Weekly Sales Over Time** plot"):
    st.markdown("""
                - **Overal Trend and baseline** : Overall weekly sales remain relatively stable between Jan-Nov from 2010-2012, fluctuating around 1.0–1.2 million.
                - **Seasonal patterns** : Sharp peaks occur at the end of each year(around end November to December), indicating strong holiday-driven demand, and sales decreased below the average baseline immediately after holiday period. 
                - **Yearly repetition** : These spikes and dips follow a repeated annual pattern, confirming strong seasonality in sales behavior.
                """)
   
   
# Data exploration 2 - Compare Holiday and Non-Holiday Sales
st.write(" ##### 2️⃣ 🔂 Compare Holiday and Non-Holiday Sales") 
fig, ax = plt.subplots(figsize=(12,5))
sns.boxenplot(data=df,x='Holiday_Flag',y='Weekly_Sales')
ax.set_title("Holiday and Non-Holiday Sales")
st.pyplot(fig)
with st.expander("Insights from **Holiday vs Non-Holiday Sales** plot"):
    st.markdown("""
                Contrary to the common assumption that holidays drive massive sales spikes, the data reveals that the difference in weekly sales between holiday and non-holiday periods is surprisingly modest. 
                **What might be the reason** ?
                - Holidays are not a guaranteed massive revenue driver on their own 
                - Holiday in weekend or not ? 
                - Was the holiday supported by a major makerting compaign ?
                - Super Bowl week compare with other holiday?
                - Overall economic climate ?
              """)

# Data exploration 3 - Focus on Weekly Sales Correlation
st.write(" ##### 3️⃣ 🗓️ Weekly Sales Correlations ") 

sales_corr = df.corr(numeric_only=True)["Weekly_Sales"].sort_values(ascending=True)
sales_corr_holiday = df_holiday.corr(numeric_only=True)["Weekly_Sales"].sort_values(ascending=True)
sales_corr_non_holiday = df_non_holiday.corr(numeric_only=True)["Weekly_Sales"].sort_values(ascending=True)

c = st.container()
col1, col2, col3 = c.columns([2,2,2])
with col1:
    st.write("Sales Correlation of all dates:")
    st.dataframe(sales_corr)
with col2:
    st.write("Sales Correlation Holiday: ")
    st.dataframe(sales_corr_holiday)
with col3:
    st.write("Sales Correlation Non-Holiday:")
    st.dataframe(sales_corr_non_holiday)

with st.expander("Insights from **Sales Correlation** table "):
    st.markdown("""
                1. **Store** has the strongest negative correlation with sales, which means the correlation suggests that higher-numbered stores have systematically lower sales than lower-numbered stores(Store location, size, or management),needs to do store segmentation
                2. **macro-economic factors** (Unemployment and CPI) is the second most important factor effecting sales, we can't do much about it)
                """)
# Data exploration 4 - Sales by Store
st.write(" ##### 4️⃣ 🏪 Stores By Sales ") 
sales_by_store = df.groupby("Store")["Weekly_Sales"].mean().sort_values()
fig, ax = plt.subplots(figsize=(12,5))
sales_by_store.plot(kind="bar")
ax.set_title("Stores by Sales")
st.pyplot(fig)

st.write(" ##### 5️⃣ 🏪 Store Segmentation ") 

# Aggregate Store-Level Metrics
store_summary = df.groupby("Store").agg(
    mean_sales =("Weekly_Sales","mean"),
    median_sales =("Weekly_Sales","median"),
    sales_var =("Weekly_Sales","var"),
    sales_std = ("Weekly_Sales","std"),
    holiday_sales =("Weekly_Sales", lambda x: df.loc[x.index, "Holiday_Flag"].mul(x).sum()/df.loc[x.index, "Holiday_Flag"].sum()),
    nonholiday_sales=("Weekly_Sales", lambda x: df.loc[x.index, "Weekly_Sales"][df.loc[x.index,"Holiday_Flag"]==0].mean())
).reset_index()

# % increase in sales during holidays, and which store benifits the most from holiday
store_summary["holiday_uplift"]=(store_summary['holiday_sales'] - store_summary['nonholiday_sales'])/store_summary['nonholiday_sales']
store_summary = store_summary.drop(columns=["holiday_sales", "nonholiday_sales"])
store_summary.columns = ["Store","mean_sales","sales_std","median_sales","sales_var","holiday_uplift"]

# sSelect features for clustering 

X = store_summary[["mean_sales","sales_std","holiday_uplift"]]
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

kmeans = KMeans(n_clusters=4,random_state=42)
store_summary["Clusters"] = kmeans.fit_predict(X_scaled)

# Define clusters assignment logic
#  Profile the clusters
cluster_profile = store_summary.groupby("Clusters").agg({
    "mean_sales":"median",
    "holiday_uplift":"median"
}).reset_index()

#  Define thresholds for high/low
sales_median = cluster_profile["mean_sales"].median()
uplift_median = cluster_profile["holiday_uplift"].median()

# Categorize clusters using your rules
def assign_category(row):
    if row['mean_sales'] > sales_median and row["holiday_uplift"] <= uplift_median:
        return "Flagstore"
    elif row['mean_sales'] >= sales_median and row['holiday_uplift'] >= uplift_median:
        return "Regional store"
    elif row['mean_sales'] <= sales_median and row['holiday_uplift'] > uplift_median:
        return "Local store"
    else:
        return "Rural store"
    
cluster_profile["Category"] = cluster_profile.apply(assign_category, axis=1)

# create mapping dict
cluster_map = dict(zip(cluster_profile['Clusters'],cluster_profile['Category']))

# add back to full dataset
store_summary["Cluster_Category"] = store_summary["Clusters"].map(cluster_map)

# merge cluster/category info back into the original df
df = df.merge(
    store_summary[['Store','Clusters','Cluster_Category']],
    on="Store",
    how="left"
)

st.dataframe(cluster_profile)
fig, ax = plt.subplots(figsize=(10,6))

scatter = ax.scatter(
    store_summary["mean_sales"],
    store_summary["holiday_uplift"],
    c=store_summary["Clusters"],
    cmap="viridis",
    s=100,
    alpha=0.7
)
ax.set_xlabel("Mean Weekly Sales")
ax.set_ylabel("Holiday Uplift (%) ")
ax.set_title("Store segmentation: Volume vs Holiday Sensitivity")

handles = []
colors = scatter.cmap(scatter.norm(store_summary["Clusters"].unique()))
for cluster, color in zip(store_summary["Clusters"].unique(), colors):
    label = cluster_map.get(cluster, f"Cluster {cluster}")
    handles.append(mpatches.Patch(color=color, label=label))

ax.legend(handles=handles, title="Cluster Category")
st.pyplot(fig)

with st.expander("Store Segmentation info"):
    st.markdown("""
                - Flagstore --- High Sales, Median Holiday Uplift
                - Regional Store  -- Median-to-High Sales, Median Holiday Uplift
                - Local store  -- Low-to-Median Sales, Median-to-High Holiday Uplift
                - Rural store -- Low Sales, Negative to Low Holiday Uplift
                """)



st.write(" ##### 6️⃣ ➡️ Sales Forecast For Clusters ") 


# Train a Prophet model per cluster category. Forecast future sales per cluster
cluster_sales = (
     df.groupby(['Cluster_Category','Date'])['Weekly_Sales'].sum().reset_index()
)
# Merge with original dataframe to get regressors
df_with_features = df.groupby(['Cluster_Category','Date']).agg({
     'Weekly_Sales':'sum',
     'Holiday_Flag':'max',
     'Unemployment':'mean',
     'CPI':'mean',
     'Fuel_Price':'mean'
}).reset_index()

df_with_features['Month'] = pd.to_datetime(df_with_features['Date']).dt.month

# forcaste per cluster category
forecast_results = {}

for cluster in df_with_features['Cluster_Category'].unique():
     df_prophet['ds'] = pd.to_datetime(df_prophet['ds'])
     future['ds'] = pd.to_datetime(future['ds']) 
     forecast['ds'] = pd.to_datetime(forecast['ds']) 
     forecast_results[cluster]=(prophet_model,forecast,df_prophet)
     fig = prophet_model.plot(forecast)
     plt.title(f"Sales Forecast for {cluster} with Regressors",fontsize=14)
     st.pyplot(fig)


# Check and validate the sales trending down with macro market(retail index, unemployment and inflation etc)

st.write("##### Sales results VS Macro economics")

macro_df = macro_data.resample('ME').mean()  # ensure monthly

df['Date'] = pd.to_datetime(df['Date'])
cluster_sales = df.groupby(['Date', 'Cluster_Category'])['Weekly_Sales'].mean().reset_index()
cluster_monthly = (cluster_sales.groupby(
    ['Cluster_Category',pd.Grouper(key='Date',freq='ME')])['Weekly_Sales']
                   .mean().reset_index())
# Pivot so each cluster is its own column
cluster_pivot = cluster_monthly.pivot(index='Date',columns='Cluster_Category',values='Weekly_Sales')

combine = cluster_pivot.merge(macro_df,left_index=True,right_index=True,how='inner')

fig,ax1= plt.subplots(figsize=(16,8))

# plot sales per cluster
for col in cluster_pivot.columns:
    ax1.plot(combine.index,combine[col],label=f"{col}  sales")
ax1.set_ylabel('Cluster Monthly Sales')
ax1.set_xlabel("Date")
ax1.legend(loc='upper left')

# Plot macro indicators (right y-axis)
ax2 = ax1.twinx()
ax2.plot(combine.index,combine['RetailSales'],color='purple',linestyle="--",label='National RetaiL Sales')
ax2.plot(combine.index,combine['Unemployment'],color='brown',label='Unemployment')
ax2.plot(combine.index,combine['CPI'],color='black',linestyle="--",label='CPI')
ax2.set_ylabel("Macro Indicators")

# Merge legends 
lines1, labels1 = ax1.get_legend_handles_labels()
lines2,labels2 =ax2.get_legend_handles_labels()
ax1.legend(lines1+lines2,labels1+labels2,loc='upper right')
plt.title("Cluster Sales vs. Macro Indicators (CPI, Unemployment, RetailSales)")
st.pyplot(fig)

def diagnose_prophet_trend_issue(model, forecast):
    """Check why Prophet trend is so pessimistic"""
    
    # Extract components
    forecast_components = model.predict(future)
    
    st.write("=== PROPHET COMPONENT ANALYSIS ===")
    
    # Check trend contribution
    trend_start = forecast_components['trend'].iloc[0]
    trend_end = forecast_components['trend'].iloc[-52]  # Last year of historical
    trend_change = ((trend_end - trend_start) / trend_start) * 100
    
    st.write(f"Prophet trend change: {trend_change:.1f}%")
    
    # Check if other components are compensating
    if 'seasonal' in forecast_components.columns:
        seasonal_impact = forecast_components['seasonal'].std()
        st.write(f"Seasonal volatility: {seasonal_impact:.0f}")
    
    # Check regressor impact
    for regressor in ['CPI', 'Unemployment']:
        if regressor in forecast_components.columns:
            regressor_impact = forecast_components[regressor].iloc[-1] - forecast_components[regressor].iloc[0]
            st.write(f"{regressor} total impact: {regressor_impact:.0f}")
    
    # The key insight:
    total_forecast_change = forecast_components['yhat'].iloc[-1] - forecast_components['yhat'].iloc[0]
    st.write(f"Total forecast change: {(total_forecast_change / forecast_components['yhat'].iloc[0] * 100):.1f}%")
    

diagnose_prophet_trend_issue(prophet_model,forecast)

with st.expander("Insights from **Sales Forecasting** Analysis"):
    st.markdown("""  
                1. CPI and Unemployment both impact are 0, macro economics are influencing the forecast, the sales driven by other factors.
                2. Model detected an inherent downward trend in the business ,National RetailSales on the contrary are are trending up.The downward trend exists independently of macroeconomic condition.
                3. seasonality_impact = (-10.9) - (-5.3) , about -5.6%, so seasonality and other factors contributed additional decline
                """)

# 🔎 Sales Driver Analysis with XGBoost
st.write(" ##### 7️⃣ ✇ Sales Driver analysis with XGBoost") 

# Encode cluster as categorical if you want to model all together
df["Clusters"] = df["Clusters"].astype("category").cat.codes 

X = df[['Holiday_Flag','CPI','Unemployment','Fuel_Price','Clusters','Temperature']]
y = df['Weekly_Sales']

X_train, X_test, y_train, y_test = train_test_split(X,y,test_size=0.2,shuffle=False)

y_pred_xgb= model_xgb.predict(X_test)

fig, ax = plt.subplots(figsize=(8,6))
plot_importance(model_xgb,ax=ax,importance_type="gain")  # "weight", "gain", or "cover"
plt.title("Feature Importance(XGBoost)")
st.pyplot(fig)

with st.expander("Insights from Feature Importance :"):
    st.markdown("""
                1. Store(Clusters) Format 📝 Dominance: 94%+ of Sales Predictive Power
                2. Combine with the Store Segmentation and Seasonal Patterns(Holiday Uplift)
                - Only **Local Store** has median to high holiday uplift
                - Flagstore and Regional store both underperformed during Holidays 
                - Rural store has the negative holiday uplift impact
                3. Need to find out which stores perform well and poorly on which holidays。 
                """)


st.write(" ##### 8️⃣ 🎁 Holiday Impact per Clusters/Stores") 
holiday_type = pd.concat([
    pd.DataFrame({
        'Holiday_Name': 'Thanksgiving',
        'Date': pd.to_datetime(['2010-11-25','2011-11-24']),
    }),
    pd.DataFrame({
        'Holiday_Name': 'Christmas',
        'Date': pd.to_datetime(['2010-12-25','2011-12-25']),
    }),
    pd.DataFrame({
        'Holiday_Name': 'New Year',
        'Date': pd.to_datetime(['2011-01-01','2012-01-01',]),
    }),
    pd.DataFrame({
        'Holiday_Name': 'Black Friday',
        'Date': pd.to_datetime(['2010-11-26','2011-11-25']),
    }),
    pd.DataFrame({
        'Holiday_Name': 'Super Bowl',
        'Date': pd.to_datetime(['2010-02-07','2011-02-06','2012-02-05']),
    })
])

holiday_type['Week_end_date'] = holiday_type['Date'] + pd.to_timedelta((4-holiday_type['Date'].dt.weekday) % 7,unit='D')
#holiday_type[['Date','Holiday_Name','Week_end_date']]
df = pd.merge(df,holiday_type[['Week_end_date','Holiday_Name']].rename(columns={'Week_end_date':'Date'}),on='Date',how='left')

st.write(f"\nNumber of holiday records found : {df['Holiday_Name'].notna().sum()}")
st.write("Holidays found:")
st.write(df['Holiday_Name'].value_counts())

df.to_pickle("data/processed_data.pkl")

holidays = holiday_type.copy()
# holidays = holidays.rename(columns={'Date':'ds','Holiday_Name':'holiday'})
# holidays['ds'] = pd.to_datetime(holidays['ds'])
holiday_impact_per_cluster = []

for cluster in df['Cluster_Category'].unique():
    cluster_data = df[df['Cluster_Category'] == cluster]
    cluster_baseline = cluster_data[cluster_data['Holiday_Name'].isna()]['Weekly_Sales'].mean()


    for holiday in ['Thanksgiving', 'Christmas', 'New Year', 'Super Bowl', 'Black Friday']:
        holiday_mask = (cluster_data['Holiday_Name'] == holiday)
        holiday_sales = cluster_data.loc[holiday_mask,'Weekly_Sales']
        
        if len(holiday_sales) > 0:
            avg_holiday_sales = holiday_sales.mean()
            impact = avg_holiday_sales - cluster_baseline
            pct_impact = (impact / cluster_baseline) * 100
        else:
            avg_holiday_sales = impact = pct_impact = np.nan
            
        holiday_impact_per_cluster.append({
            'Cluster':cluster,
            'Holiday':holiday,
            'Avg_Holiday_Sales':avg_holiday_sales,
            'Cluster_Baseline':cluster_baseline,
            'Impact':impact,
            'Percentage of impact':pct_impact,
            'N_Observations':len(holiday_sales)
        })
        
impact_df = pd.DataFrame(holiday_impact_per_cluster)
#impact_df[['Cluster', 'Holiday', 'Impact', 'Percentage of impact', 'N_Observations']]
impact_df.to_pickle("data/impact_data.pkl")


col1,col2 = st.columns(2)
with col1:
    fig1,ax1 = plt.subplots(figsize=(8,6))
    pivot_df = impact_df.pivot(index='Holiday',columns='Cluster',values='Impact')
    sns.heatmap(pivot_df,annot=True,fmt=',.0f',cmap='RdBu_r',center=0,
            annot_kws={'size':10},cbar_kws={'label':'Sales Impact($)'})
    ax1.set_title('Holiday Sales Impact by Cluster($)')
    st.pyplot(fig1)

with col2:
    fig2,ax2=plt.subplots(figsize=(8,6))
    pivot_pct = impact_df.pivot(index='Holiday',columns='Cluster',values='Percentage of impact')
    sns.heatmap(pivot_pct,annot=True, fmt=',.1f', cmap='RdBu_r', center=0,
            annot_kws={'size': 10}, cbar_kws={'label': 'Percentage of impact'})
    ax2.set_title('Holiday Sales impact by cluster(%)')
    st.pyplot(fig2)

with st.expander("Insights of Holiday Impact:"):
    st.markdown("""
                1. **Black Friday & Thanksgiving** (High positive uplift across clusters).
                - Consistently strong uplift (>40% in Flagstore, Local, Regional)
                - Even Rural stores benefit moderately (~10%).
                - These holidays are proven revenue drivers and should remain central to campaign planning.
                2. **Christmas & New Year** (Negative impact in most clusters). 
                - Sales actually dip during these periods (especially Rural stores: -14.4%).
                - This suggests potential cannibalization (customers may shift spend to other times) or weaker holiday relevance.
                3. **Super Bowl** (Small positive but weak uplift).
                - Slight improvements (2–5%), not strong across clusters.
                - Could be leveraged only if cost-efficient promotions are possible.
                """)


st.subheader("📊 Strategic Next Steps: Holiday Sales Impact")

# --- Black Friday + Thanksgiving (Growth) ---
with st.expander("🛍️ Black Friday + Thanksgiving (Growth Focus)"):
    st.markdown(
        """
        :green[**Opportunity:** Strong positive uplift, main growth driver.]  

        **Actions**
        - Build uplift model to capture holiday-driven upside.
        - Identify customer/store segments with strongest responses (e.g., Flagship & Regional).
        - Use as *primary driver* for forecasting + promo ROI analysis.  

        **Next Step** → Launch growth-focused uplift model and benchmark ROI.
        """
    )

# --- Christmas + New Year (Risk Mitigation) ---
with st.expander("🎄 Christmas + 🎆 New Year (Risk Mitigation)"):
    st.markdown(
        """
        :red[**Risk:** Sales dip observed (e.g., Rural –14.4%).]  

        **Actions**
        - Develop mitigation model to protect baseline sales.
        - Detect underperforming clusters and customers at risk.  

        **Counter-Strategies**
        - Early-bird bundles  
        - Shift promo budget toward stronger holidays  
        - Off-peak campaigns to smooth demand  

        **Next Step** → Test counter-strategies in simulation and evaluate net gain.
        """
    )

# --- Super Bowl (Niche Targeting) ---
with st.expander("🏈 Super Bowl (Niche Targeting)"):
    st.markdown(
        """
        :blue[**Niche Opportunity:** Localized positive uplift in sports-driven regions.]  

        **Actions**
        - Build smaller uplift model focused on niche segments.
        - Target stores/clusters with above-average Super Bowl demand.  

        **Next Step** → Run localized promo pilots, evaluate ROI before wider rollout.
        """
    )


