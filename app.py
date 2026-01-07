import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import os

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LinearRegression
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score

DATA_FILE = "pakistan_finance_data.csv"
st.set_page_config(page_title="Budget Planning System", layout="wide")

st.markdown("""
    <style>
    .main { background-color: #f5f5f5; }
    .stButton>button { width: 100%; }
    .metric-card { background-color: white; padding: 20px; border-radius: 10px; box-shadow: 2px 2px 10px rgba(0,0,0,0.1); }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data(uploaded_file):
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
    elif os.path.exists(DATA_FILE):
        df = pd.read_csv(DATA_FILE)
    else:
        return pd.DataFrame()
    
    if 'Date' in df.columns:
        df['Date'] = pd.to_datetime(df['Date'])
    return df

@st.cache_resource
class IntelligentBudgetSystem:
    def __init__(self):
        self.vectorizer = TfidfVectorizer()
        self.classifier = MultinomialNB()
        self.regressor = LinearRegression()
        self.kmeans = KMeans(n_clusters=3, random_state=42)
        self.accuracy = 0.0

    def train_models(self, df):
        if df.empty:
            return pd.DataFrame()

        X = df["Description"]
        y = df["Category"]
        
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        X_train_vec = self.vectorizer.fit_transform(X_train)
        self.classifier.fit(X_train_vec, y_train)
        
        X_test_vec = self.vectorizer.transform(X_test)
        preds = self.classifier.predict(X_test_vec)
        self.accuracy = accuracy_score(y_test, preds)
        
        X_full_vec = self.vectorizer.fit_transform(X)
        self.classifier.fit(X_full_vec, y)
        
        self.kmeans.fit(df[["Amount"]])
        
        df["Month"] = df["Date"].dt.month
        monthly_data = df.groupby("Month")["Amount"].sum().reset_index()
        self.regressor.fit(monthly_data[["Month"]], monthly_data["Amount"])
        
        return monthly_data

    def predict_category(self, text):
        try:
            vec = self.vectorizer.transform([text])
            return self.classifier.predict(vec)[0]
        except:
            return "Unknown"

    def predict_next_month(self, next_month_idx):
        try:
            return self.regressor.predict([[next_month_idx]])[0]
        except:
            return 0

st.sidebar.title("Smart Budget PK")
uploaded_file = st.sidebar.file_uploader("Upload Transaction CSV", type=["csv"])

df = load_data(uploaded_file)

if not df.empty:
    if 'ml_system' not in st.session_state:
        st.session_state.ml_system = IntelligentBudgetSystem()
        st.session_state.monthly_data = st.session_state.ml_system.train_models(df)
    
    system = st.session_state.ml_system
    
    unique_categories = list(set(df["Category"].unique()))
    st.sidebar.info(f"Tracking {len(unique_categories)} Unique Categories")
    st.sidebar.markdown(f"**Model Accuracy:** `{system.accuracy * 100:.2f}%`")
    
    menu = st.sidebar.radio("Navigation", ["Dashboard", "AI Categorizer", "Budget Forecast"])
    
    if menu == "Dashboard":
        st.title("Intelligent Budget Dashboard")
        
        total_spent = df["Amount"].sum()
        avg_txn = df["Amount"].mean()
        
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Spending (Year)", f"PKR {total_spent:,.0f}")
        c2.metric("Avg Transaction Size", f"PKR {avg_txn:,.0f}")
        c3.metric("Total Transactions", len(df))
        
        col_left, col_right = st.columns(2)
        with col_left:
            st.subheader("Category Distribution")
            fig_pie = px.pie(df, names='Category', values='Amount', hole=0.4)
            st.plotly_chart(fig_pie, use_container_width=True)
            
        with col_right:
            st.subheader("Monthly Spending Trend")
            monthly = st.session_state.monthly_data
            if not monthly.empty:
                fig_bar = px.bar(monthly, x="Month", y="Amount", title="Monthly Expenses")
                st.plotly_chart(fig_bar, use_container_width=True)
            else:
                st.info("Not enough data for monthly trends.")

    elif menu == "AI Categorizer":
        st.title("AI Transaction Categorizer")
        st.info("The system learned from your dataset. Test the classification below.")
        
        user_text = st.text_input("Enter Transaction Description", "")
        
        if st.button("Categorize"):
            if user_text:
                cat = system.predict_category(user_text)
                st.success(f"Predicted Category: {cat}")
            else:
                st.warning("Please enter a description.")
        
        st.write("---")
        st.subheader("Raw Dataset View")
        st.dataframe(df.head(20), use_container_width=True)

    elif menu == "Budget Forecast":
        st.title("Budget Prediction (Regression Analysis)")
        
        income = st.number_input("Monthly Income (PKR)", value=150000, step=5000)
        
        next_month_idx = 13
        predicted_amt = system.predict_next_month(next_month_idx)
        
        savings = income - predicted_amt
        if income > 0:
            adherence_rate = (savings / income) * 100
        else:
            adherence_rate = 0
        
        c1, c2 = st.columns(2)
        c1.metric("Predicted Expense (Next Month)", f"PKR {predicted_amt:,.0f}")
        c2.metric("Budget Adherence Rate", f"{adherence_rate:.1f}%", delta_color="normal")
        
        if savings < 0:
            st.error("Warning: Your predicted spending exceeds your income.")
        else:
            st.success("Status: You are projected to stay within budget.")

else:
    st.warning("Please upload a CSV file or run the generator script to create 'pakistan_finance_data.csv'.")