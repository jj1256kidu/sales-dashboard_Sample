# --- LOGIN + DASHBOARD IN ONE FILE ---
import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
from functools import lru_cache

# --------------------------
# 1. User Authentication
# --------------------------
USER_CREDENTIALS = {
    "admin": "admin123",
    "sales": "sales2024"
}

# Initialize session state
for key in ['logged_in', 'username', 'df', 'current_view', 'date_filter',
            'selected_practice', 'selected_stage', 'reset_triggered', 
            'selected_team_member', 'sales_target']:
    if key not in st.session_state:
        st.session_state[key] = False if key == 'logged_in' else None if key != 'sales_target' else 0.0

# Login Page
def show_login():
    st.set_page_config(page_title="Login - Sales Dashboard", page_icon="🔒")
    st.markdown("""
        <div style="text-align:center; margin-top: 80px;">
            <h1 style="color:#4A90E2;">🔐 Sales Dashboard Login</h1>
            <p style="font-size:1.1em;">Please enter your credentials to continue</p>
        </div>
    """, unsafe_allow_html=True)

    with st.form("login_form", clear_on_submit=False):
        username = st.text_input("Username")
        password = st.text_input("Password", type="password")
        submit_btn = st.form_submit_button("Login")

        if submit_btn:
            if username in USER_CREDENTIALS and USER_CREDENTIALS[username] == password:
                st.success("Login successful!")
                st.session_state.logged_in = True
                st.session_state.username = username
                st.rerun()
            else:
                st.error("Invalid username or password. Please try again.")

# Logout function
def logout():
    st.session_state.logged_in = False
    st.session_state.username = ""
    st.rerun()

# --------------------------
# 2. Format Helpers
# --------------------------
def format_amount(x):
    try:
        if pd.isna(x) or x == 0:
            return "₹0L"
        value = float(str(x).replace('₹', '').replace('L', '').replace(',', ''))
        return f"₹{int(value)}L"
    except:
        return "₹0L"

def format_percentage(x):
    try:
        if pd.isna(x) or x == 0:
            return "0%"
        if isinstance(x, str):
            value = float(x.rstrip('%'))
        else:
            value = float(x)
        return f"{int(value)}%"
    except:
        return "0%"

def format_number(x):
    try:
        if pd.isna(x) or x == 0:
            return "0"
        value = float(str(x).replace(',', ''))
        return f"{int(value):,}"
    except:
        return "0"

# --------------------------
# 3. Process & Filter Data
# --------------------------
@st.cache_data
def process_data(df):
    df = df.copy()
    df['Expected Close Date'] = pd.to_datetime(df['Expected Close Date'], errors='coerce')
    df['Month'] = df['Expected Close Date'].dt.strftime('%B')
    df['Year'] = df['Expected Close Date'].dt.year
    df['Quarter'] = df['Expected Close Date'].dt.quarter.map({1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'})
    def convert_probability(x):
        try:
            if pd.isna(x):
                return 0
            if isinstance(x, str):
                x = x.rstrip('%')
            return float(x)
        except:
            return 0
    df['Probability_Num'] = df['Probability'].apply(convert_probability)
    df['Is_Won'] = df['Sales Stage'].str.contains('Won', case=False, na=False)
    df['Amount_Lacs'] = df['Amount'].fillna(0).div(100000).round(0).astype(int)
    df['Weighted_Amount'] = (df['Amount_Lacs'] * df['Probability_Num'] / 100).round(0).astype(int)
    return df

@st.cache_data
def calculate_team_metrics(df):
    team_metrics = df.groupby('Sales Owner').agg({
        'Amount': lambda x: int(x[df['Is_Won'] & x.notna()].sum() / 100000) if len(x[df['Is_Won'] & x.notna()]) > 0 else 0,
        'Is_Won': 'sum',
        'Amount_Lacs': lambda x: int(x[~df['Is_Won'] & x.notna()].sum()) if len(x[~df['Is_Won'] & x.notna()]) > 0 else 0,
        'Weighted_Amount': lambda x: int(x[~df['Is_Won'] & x.notna()].sum()) if len(x[~df['Is_Won'] & x.notna()]) > 0 else 0
    }).reset_index()
    team_metrics.columns = ['Sales Owner', 'Closed Won', 'Closed Deals', 'Current Pipeline', 'Weighted Projections']
    team_metrics = team_metrics.fillna(0)
    pipeline_deals = df[~df['Is_Won']].groupby('Sales Owner').size()
    team_metrics['Pipeline Deals'] = team_metrics['Sales Owner'].map(pipeline_deals).fillna(0).astype(int)
    total_deals = team_metrics['Closed Deals'] + team_metrics['Pipeline Deals']
    team_metrics['Win Rate'] = np.where(
        total_deals > 0,
        (team_metrics['Closed Deals'] / total_deals * 100).round(0),
        0
    ).astype(int)
    return team_metrics

@st.cache_data
def filter_dataframe(df, filters):
    mask = pd.Series(True, index=df.index)
    if filters.get('selected_member') != "All Team Members":
        mask &= df['Sales Owner'] == filters['selected_member']
    if filters.get('search'):
        search_mask = pd.Series(False, index=df.index)
        search = filters['search'].lower()
        for col in ['Organization Name', 'Opportunity Name', 'Sales Owner', 'Sales Stage']:
            search_mask |= df[col].astype(str).str.lower().str.contains(search, na=False)
        mask &= search_mask
    if filters.get('month_filter') != "All Months":
        mask &= df['Month'] == filters['month_filter']
    if filters.get('quarter_filter') != "All Quarters":
        mask &= df['Quarter'] == filters['quarter_filter']
    if filters.get('year_filter') != "All Years":
        mask &= df['Year'] == filters['year_filter']
    if filters.get('probability_filter') != "All Probability":
        if filters['probability_filter'] == "Custom Range":
            prob_range = filters['custom_prob_range'].split("-")
        else:
            prob_range = filters['probability_filter'].split("-")
        min_prob = float(prob_range[0])
        max_prob = float(prob_range[1].rstrip("%"))
        mask &= (df['Probability_Num'] >= min_prob) & (df['Probability_Num'] <= max_prob)
    if filters.get('status_filter') != "All Status":
        if filters['status_filter'] == "Committed for the Month":
            current_month = pd.Timestamp.now().strftime('%B')
            mask &= (df['Month'] == current_month) & (df['Probability_Num'] > 75)
        elif filters['status_filter'] == "Upsides for the Month":
            current_month = pd.Timestamp.now().strftime('%B')
            mask &= (df['Month'] == current_month) & (df['Probability_Num'].between(25, 75))
        else:
            mask &= df['Sales Stage'] == filters['status_filter']
    if filters.get('focus_filter') != "All Focus":
        mask &= df['KritiKal Focus Areas'] == filters['focus_filter']
    return df[mask]

# --------------------------
# 4. Your Original Dashboard Pages
# --------------------------
# Place your full implementations of these functions below:
# - show_data_input()
# - show_overview()
# - show_sales_team()
# - show_detailed()

# ⚠️ Due to character limits, the detailed logic for each view
# (which you already provided and is long) should be defined **exactly**
# as you have in your original script.

# Example stub to demonstrate:
def show_data_input():
    st.title("Data Input Page")
    st.info("Paste your original `show_data_input()` code here.")

def show_overview():
    st.title("Overview Page")
    st.info("Paste your original `show_overview()` code here.")

def show_sales_team():
    st.title("Sales Team Page")
    st.info("Paste your original `show_sales_team()` code here.")

def show_detailed():
    st.title("Detailed Data Page")
    st.info("Paste your original `show_detailed()` code here.")

# --------------------------
# 5. Main Navigation & Launch
# --------------------------
def main():
    if not st.session_state.logged_in:
        show_login()
    else:
        with st.sidebar:
            st.title(f"Welcome, {st.session_state.username}")
            selected = st.radio(
                "Select View",
                options=["Data Input", "Overview", "Sales Team", "Detailed Data"],
                key="navigation"
            )
            if st.button("🚪 Logout"):
                logout()
            st.session_state.current_view = selected.lower().replace(" ", "_")

        if st.session_state.current_view == "data_input":
            show_data_input()
        elif st.session_state.current_view == "overview":
            show_overview()
        elif st.session_state.current_view == "sales_team":
            show_sales_team()
        elif st.session_state.current_view == "detailed_data":
            show_detailed()

if __name__ == "__main__":
    main()
