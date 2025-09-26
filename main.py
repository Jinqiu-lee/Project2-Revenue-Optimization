import streamlit as st 
import app.data_exploration
import app.model_and_evaluation
import app.project_overview


st.set_page_config(page_title="Revenue Optimization & Sales Forecasting",
                   page_icon="📊",
                   layout="wide")
st.title("💹  Revenue Optimization Dashboard")
st.markdown("This app demonstrates **Sales Forecasting and Revenue Optimizaton** for Walmart weekly sales data.")

overview_page = st.Page("app/project_overview.py",title="Project Overview",icon="🔍")
data_page = st.Page("app/data_exploration.py",title="Data Exploration and Visulization",icon="👁️‍🗨️")
model_page = st.Page("app/model_and_evaluation.py",title="Model and Evaluation",icon="🤖")

pg = st.navigation([overview_page,data_page,model_page])
pg.run()
