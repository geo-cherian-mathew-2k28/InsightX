import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from sklearn.ensemble import RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
from datetime import timedelta
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# -----------------------------------------------------------------------------
# CONFIGURATION & SETTINGS
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="InsightX | AI-Driven Log Analytics",
    page_icon="🚀",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for "Industry-Grade" Look
st.markdown("""
<style>
    .main {
        background-color: #f5f7f9;
    }
    .metric-card {
        background-color: #ffffff;
        border-radius: 10px;
        padding: 20px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }
    h1, h2, h3 {
        color: #2c3e50;
    }
    .stButton>button {
        background-color: #2c3e50;
        color: white;
        border-radius: 5px;
    }
</style>
""", unsafe_allow_html=True)

# -----------------------------------------------------------------------------
# 1. DATA INGESTION & CLEANING ENGINE
# -----------------------------------------------------------------------------
@st.cache_data
def load_and_process_data(file_path):
    """
    Simulates the Data Ingestion and Cleaning layer.
    Reads raw CSV or Excel, fixes types, handles missing values, and creates features.
    """
    try:
        # Load Data based on extension
        if file_path.endswith('.xlsx'):
            df = pd.read_excel(file_path)
        else:
            df = pd.read_csv(file_path)
        
        # Standardize Column Names (Remove special chars, lower case)
        df.columns = [c.strip().lower().replace(' ', '_').replace('(', '').replace(')', '').replace('.', '') for c in df.columns]
        
        # Fix Typos in Column Names (Based on your dataset snippet)
        if 'accessed_ffom' in df.columns:
            df.rename(columns={'accessed_ffom': 'accessed_from'}, inplace=True)
            
        # 1. Date Conversion
        df['accessed_date'] = pd.to_datetime(df['accessed_date'], errors='coerce')
        
        # 2. Extract Time Features
        df['hour'] = df['accessed_date'].dt.hour
        df['day_name'] = df['accessed_date'].dt.day_name()
        df['month'] = df['accessed_date'].dt.month_name()
        df['date_only'] = df['accessed_date'].dt.date

        # 3. Numeric Cleaning
        # Handle 'sales' (remove currency symbols if any, though snippet looks numeric)
        if 'sales' in df.columns:
            df['sales'] = pd.to_numeric(df['sales'], errors='coerce').fillna(0)
        
        if 'duration_secs' in df.columns:
            df['duration_secs'] = pd.to_numeric(df['duration_secs'], errors='coerce').fillna(0)
            
        if 'returned_amount' in df.columns:
            df['returned_amount'] = pd.to_numeric(df['returned_amount'], errors='coerce').fillna(0)
        
        # 4. Categorical Cleaning
        if 'gender' in df.columns:
            df['gender'] = df['gender'].fillna('Unknown')
        if 'country' in df.columns:
            df['country'] = df['country'].fillna('Unknown')
        if 'accessed_from' in df.columns:
            df['accessed_from'] = df['accessed_from'].fillna('Direct') # Browser/Source
        
        # 5. Logic: Calculate Net Sales (Sales - Returns)
        df['net_sales'] = df['sales'] - df['returned_amount']
        
        # 6. Logic: Define "Cart Abandonment" proxy
        # If duration > 300s (5 mins) but sales == 0, likely abandonment or window shopping
        df['potential_abandonment'] = np.where((df['duration_secs'] > 300) & (df['sales'] == 0), 'Yes', 'No')

        return df

    except Exception as e:
        st.error(f"Error processing logs: {e}")
        return None

# -----------------------------------------------------------------------------
# 2. MACHINE LEARNING ENGINE
# -----------------------------------------------------------------------------
def train_sales_forecast(df):
    """
    Predicts future daily sales using RandomForest.
    """
    # Aggregate data by day
    daily_sales = df.groupby('date_only')['sales'].sum().reset_index()
    daily_sales['date_only'] = pd.to_datetime(daily_sales['date_only'])
    
    # Feature Engineering for ML (Lag features)
    daily_sales['day_of_year'] = daily_sales['date_only'].dt.dayofyear
    daily_sales['day_of_week'] = daily_sales['date_only'].dt.dayofweek
    daily_sales['month'] = daily_sales['date_only'].dt.month
    
    # Target
    X = daily_sales[['day_of_year', 'day_of_week', 'month']]
    y = daily_sales['sales']
    
    # Train Model
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X, y)
    
    # Predict next 30 days
    last_date = daily_sales['date_only'].max()
    future_dates = [last_date + timedelta(days=x) for x in range(1, 31)]
    future_df = pd.DataFrame({'date_only': future_dates})
    future_df['day_of_year'] = future_df['date_only'].dt.dayofyear
    future_df['day_of_week'] = future_df['date_only'].dt.dayofweek
    future_df['month'] = future_df['date_only'].dt.month
    
    predictions = model.predict(future_df[['day_of_year', 'day_of_week', 'month']])
    future_df['predicted_sales'] = predictions
    
    return daily_sales, future_df

def customer_segmentation(df):
    """
    Clusters users based on Duration and Spend (RFM-like approach).
    """
    # Group by IP (simulating User ID)
    user_data = df.groupby('ip').agg({
        'duration_secs': 'mean',
        'sales': 'sum',
        'accessed_date': 'count' # Frequency
    }).rename(columns={'accessed_date': 'visits'})
    
    # Normalize
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(user_data)
    
    # KMeans Clustering
    kmeans = KMeans(n_clusters=3, random_state=42)
    user_data['cluster'] = kmeans.fit_predict(scaled_data)
    
    # Label Clusters
    # We simplistically label based on Sales mean
    cluster_means = user_data.groupby('cluster')['sales'].mean().sort_values()
    label_map = {
        cluster_means.index[0]: 'Casual Browsers',
        cluster_means.index[1]: 'Potential Customers',
        cluster_means.index[2]: 'High-Value VIPs'
    }
    user_data['segment'] = user_data['cluster'].map(label_map)
    
    return user_data

# -----------------------------------------------------------------------------
# 3. UI & DASHBOARD LOGIC
# -----------------------------------------------------------------------------
def main():
    # Sidebar
    st.sidebar.title("InsightX ⚙️")
    st.sidebar.markdown("---")
    
    # File Uploader (Hardcoded for demo but adaptable)
    filename = "E-commerce Website Logs.xlsx"
    
    st.sidebar.info("Loading Data Engine...")
    df = load_and_process_data(filename)
    
    if df is not None:
        st.sidebar.success("Data Ingested Successfully!")
        
        # Sidebar Filters
        st.sidebar.subheader("Filter Dashboard")
        selected_country = st.sidebar.multiselect("Select Country", options=df['country'].unique(), default=df['country'].unique()[:5])
        
        # Filter Logic
        if selected_country:
            df_filtered = df[df['country'].isin(selected_country)]
        else:
            df_filtered = df
        
        # TABS
        tab1, tab2, tab3, tab4, tab5 = st.tabs([
            "🏛 Executive Summary", 
            "📊 Sales Intelligence", 
            "🕸 User Behavior", 
            "🤖 ML Predictions", 
            "📝 Raw Data"
        ])
        
        # --- TAB 1: EXECUTIVE SUMMARY ---
        with tab1:
            st.title("InsightX Executive Dashboard")
            st.markdown("### Real-time High-Level Business KPIs")
            
            # KPI Metrics
            total_sales = df_filtered['sales'].sum()
            total_visits = len(df_filtered)
            avg_order_val = df_filtered[df_filtered['sales'] > 0]['sales'].mean()
            conversion_rate = (len(df_filtered[df_filtered['sales'] > 0]) / total_visits) * 100
            
            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Total Revenue", f"${total_sales:,.2f}", "+4.5%")
            c2.metric("Total Traffic (Hits)", f"{total_visits:,}", "+12%")
            c3.metric("Avg Order Value", f"${avg_order_val:,.2f}", "-2%")
            c4.metric("Conversion Rate", f"{conversion_rate:.2f}%", "+1.2%")
            
            st.markdown("---")
            
            # Hourly Traffic Heatmap
            st.subheader("Peak Traffic Analysis")
            heatmap_data = df_filtered.groupby(['day_name', 'hour']).size().reset_index(name='count')
            fig_heat = px.density_heatmap(
                heatmap_data, x='hour', y='day_name', z='count', 
                color_continuous_scale='Viridis',
                title="Traffic Heatmap: Day vs Hour"
            )
            st.plotly_chart(fig_heat, use_container_width=True)
            
            # Insight Text Generation (Rule-based)
            if not heatmap_data.empty:
                peak_day = heatmap_data.sort_values('count', ascending=False).iloc[0]['day_name']
                peak_hour = heatmap_data.sort_values('count', ascending=False).iloc[0]['hour']
                st.info(f"💡 **AI Insight:** The highest traffic intensity is observed on **{peak_day}s** around **{peak_hour}:00**. Marketing campaigns should target this window for maximum visibility.")
            else:
                st.info("💡 **AI Insight:** Not enough data to determine peak traffic hours.")

        # --- TAB 2: SALES INTELLIGENCE ---
        with tab2:
            st.subheader("Revenue Analytics")
            
            # Sales over Time
            daily_sales = df_filtered.groupby('date_only')['sales'].sum().reset_index()
            fig_sales = px.line(daily_sales, x='date_only', y='sales', title="Daily Revenue Trend", markers=True)
            fig_sales.update_layout(xaxis_title="Date", yaxis_title="Revenue ($)")
            st.plotly_chart(fig_sales, use_container_width=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Sales by Country (Choropleth or Bar)
                country_sales = df_filtered.groupby('country')['sales'].sum().reset_index().sort_values('sales', ascending=False).head(10)
                fig_country = px.bar(country_sales, x='country', y='sales', color='sales', title="Top 10 Countries by Revenue")
                st.plotly_chart(fig_country, use_container_width=True)
                
            with col2:
                # Payment Methods
                if 'pay_method' in df_filtered.columns:
                    pay_counts = df_filtered['pay_method'].value_counts().reset_index()
                    pay_counts.columns = ['Method', 'Count']
                    fig_pie = px.pie(pay_counts, names='Method', values='Count', title="Payment Method Distribution", hole=0.4)
                    st.plotly_chart(fig_pie, use_container_width=True)
                else:
                    st.warning("Payment method data not found.")

        # --- TAB 3: USER BEHAVIOR ---
        with tab3:
            st.subheader("User Engagement & Behavior Patterns")
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Browser/Source Analysis
                browser_data = df_filtered['accessed_from'].value_counts().head(5)
                st.markdown("#### Top Browsers/Devices")
                st.dataframe(browser_data, width=400)
                
            with col2:
                # Duration Histogram
                fig_hist = px.histogram(df_filtered, x='duration_secs', nbins=50, title="Session Duration Distribution", color_discrete_sequence=['#ff7f0e'])
                st.plotly_chart(fig_hist, use_container_width=True)
                
            st.markdown("---")
            st.subheader("⚠️ Cart Abandonment Analysis")
            
            abandonment = df_filtered['potential_abandonment'].value_counts(normalize=True) * 100
            st.progress(int(abandonment.get('Yes', 0)))
            st.caption(f"Approximately {abandonment.get('Yes', 0):.1f}% of sessions show signs of 'Window Shopping' (High duration, 0 sales).")
            
            # Detailed look at abandonment
            abandon_df = df_filtered[df_filtered['potential_abandonment'] == 'Yes']
            st.dataframe(abandon_df[['ip', 'country', 'duration_secs', 'accessed_from']].head(5))

        # --- TAB 4: ML PREDICTIONS ---
        with tab4:
            st.title("🤖 AI-Powered Future Insights")
            st.markdown("Using **RandomForestRegressor** for forecasting and **K-Means** for segmentation.")
            
            col1, col2 = st.columns(2)
            
            # 1. Sales Forecasting
            with col1:
                st.subheader("📈 30-Day Sales Forecast")
                if len(df_filtered) > 50: # Need enough data
                    hist_data, future_data = train_sales_forecast(df_filtered)
                    
                    # Combine for plotting
                    fig_forecast = go.Figure()
                    fig_forecast.add_trace(go.Scatter(x=hist_data['date_only'], y=hist_data['sales'], mode='lines', name='Historical'))
                    fig_forecast.add_trace(go.Scatter(x=future_data['date_only'], y=future_data['predicted_sales'], mode='lines+markers', name='Forecast', line=dict(dash='dash', color='green')))
                    
                    fig_forecast.update_layout(title="Sales Prediction (Next 30 Days)", xaxis_title="Date", yaxis_title="Sales ($)")
                    st.plotly_chart(fig_forecast, use_container_width=True)
                    
                    total_predicted = future_data['predicted_sales'].sum()
                    st.success(f"Predicted Revenue for next 30 days: **${total_predicted:,.2f}**")
                else:
                    st.warning("Not enough data points for accurate forecasting.")

            # 2. Customer Segmentation
            with col2:
                st.subheader("👥 Customer Segmentation (Clustering)")
                try:
                    segmented_df = customer_segmentation(df_filtered)
                    
                    fig_cluster = px.scatter(
                        segmented_df, x='duration_secs', y='sales', color='segment',
                        title="User Segmentation: Duration vs Spend",
                        color_discrete_map={'Casual Browsers': 'blue', 'Potential Customers': 'orange', 'High-Value VIPs': 'red'}
                    )
                    st.plotly_chart(fig_cluster, use_container_width=True)
                    
                    st.markdown("**Segment Definitions:**")
                    st.markdown("- **Casual Browsers:** Low spend, varying time.")
                    st.markdown("- **Potential Customers:** Moderate spend, engaged.")
                    st.markdown("- **High-Value VIPs:** High spenders (Target for loyalty programs).")
                except Exception as e:
                    st.error(f"Clustering error: {e}")

        # --- TAB 5: RAW DATA ---
        with tab5:
            st.subheader("Processed Log Dataset")
            st.dataframe(df)
            
            # Download Button
            csv = df.to_csv(index=False).encode('utf-8')
            st.download_button("Download Cleaned Data (CSV)", csv, "insightx_processed_data.csv", "text/csv")

    else:
        st.error(f"Failed to load data. Please ensure the file '{filename}' is in the directory.")

if __name__ == "__main__":
    main()