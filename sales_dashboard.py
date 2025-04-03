import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import numpy as np
import io
from functools import lru_cache

# Format helper functions
def format_amount(x):
    try:
        if pd.isna(x) or x == 0:
            return "₹0L"
        # Convert to float first to handle string inputs, then to int
        value = float(str(x).replace('₹', '').replace('L', '').replace(',', ''))
        return f"₹{int(value)}L"
    except:
        return "₹0L"

def format_percentage(x):
    try:
        if pd.isna(x) or x == 0:
            return "0%"
        # Handle string percentage inputs
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
        # Convert to float first to handle string inputs, then to int
        value = float(str(x).replace(',', ''))
        return f"{int(value):,}"
    except:
        return "0"

# Set page config
st.set_page_config(
    page_title="Sales Dashboard",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Initialize session state
if 'df' not in st.session_state:
    st.session_state.df = None
if 'current_view' not in st.session_state:
    st.session_state.current_view = 'data_input'
if 'date_filter' not in st.session_state:
    st.session_state.date_filter = None
if 'selected_practice' not in st.session_state:
    st.session_state.selected_practice = 'All'
if 'selected_stage' not in st.session_state:
    st.session_state.selected_stage = 'All'
if 'reset_triggered' not in st.session_state:
    st.session_state.reset_triggered = False
if 'selected_team_member' not in st.session_state:
    st.session_state.selected_team_member = None

# Custom CSS for modern styling
st.markdown("""
<style>
    /* Modern theme colors */
    :root {
        --primary-color: #4A90E2;
        --background-color: #1E1E1E;
        --secondary-background-color: #252526;
        --text-color: #FFFFFF;
        --font-family: 'Segoe UI', sans-serif;
    }

    /* Main container styling */
    .main {
        background-color: var(--background-color);
        color: var(--text-color);
        font-family: var(--font-family);
    }

    /* Card styling */
    .stCard {
        background-color: var(--secondary-background-color);
        border-radius: 10px;
        padding: 15px;
        margin: 30px 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
    }

    /* Number formatting */
    .big-number {
        font-size: 2.8em;
        font-weight: 700;
        color: #2ecc71;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
        letter-spacing: -1px;
    }

    .metric-value {
        font-size: 2em;
        font-weight: 600;
        color: #4A90E2;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }

    .metric-label {
        font-size: 1.2em;
        color: #333;
        margin-bottom: 5px;
        font-weight: 500;
    }

    /* Section headers */
    .section-header {
        font-size: 1.8em;
        font-weight: 700;
        color: #2c3e50;
        margin: 30px 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }

    /* Chart text styling */
    .js-plotly-plot .plotly .main-svg {
        font-size: 14px;
        font-weight: 500;
    }

    /* Table styling */
    .dataframe {
        font-size: 1.2em;
        background-color: white;
        border-radius: 8px;
        padding: 15px;
    }

    .dataframe th {
        background-color: #4A90E2;
        color: white;
        font-weight: 700;
        padding: 15px;
        font-size: 1.1em;
    }

    .dataframe td {
        padding: 12px;
        border-bottom: 1px solid #eee;
        font-weight: 500;
    }

    /* Upload container styling */
    .upload-container {
        background-color: rgba(74, 144, 226, 0.1);
        border-radius: 10px;
        padding: 30px;
        margin: 20px 0;
        border: 2px dashed rgba(74, 144, 226, 0.3);
        text-align: center;
    }

    /* Button styling */
    .stButton>button {
        background-color: var(--primary-color);
        color: white;
        border-radius: 5px;
        padding: 10px 20px;
        border: none;
        transition: all 0.3s ease;
    }

    .stButton>button:hover {
        background-color: #357ABD;
        transform: translateY(-2px);
    }

    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}

    /* Custom header */
    .custom-header {
        background: linear-gradient(90deg, #4A90E2 0%, #357ABD 100%);
        padding: 20px;
        border-radius: 10px;
        margin-bottom: 20px;
        color: white;
        text-align: center;
    }

    /* Info box */
    .info-box {
        background-color: rgba(74, 144, 226, 0.1);
        border-left: 4px solid #4A90E2;
        padding: 15px;
        border-radius: 4px;
        margin: 10px 0;
    }

    /* Container styling */
    .container {
        margin: 30px 0;
        padding: 15px;
    }

    /* Graph container */
    .graph-container {
        margin: 30px 0;
        padding: 15px;
        background: white;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
    }

    /* Metric container */
    .metric-container {
        margin: 30px 0;
        padding: 15px;
        background: #f8f9fa;
        border-radius: 10px;
    }

    /* Section divider */
    .section-divider {
        margin: 30px 0;
        border-top: 1px solid #eee;
    }

    /* Custom styling for number input */
    [data-testid="stNumberInput"] {
        position: relative;
        background: transparent !important;
    }
    [data-testid="stNumberInput"] > div > div > input {
        color: white !important;
        font-size: 1.8em !important;
        font-weight: 800 !important;
        text-align: center !important;
        background: transparent !important;
        border: none !important;
        padding: 0 !important;
    }
    /* Hide the increment/decrement buttons */
    [data-testid="stNumberInput"] > div > div > div {
        display: none !important;
    }
    /* Container styling */
    div[data-testid="column"] > div > div > div > div > div {
        background: linear-gradient(135deg, #FF6B6B 0%, #FF8E8E 100%);
        padding: 20px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
        text-align: center;
    }

    /* Hide increment buttons */
    [data-testid="stNumberInput"] input[type="number"] {
        -moz-appearance: textfield;
    }
    [data-testid="stNumberInput"] input[type="number"]::-webkit-outer-spin-button,
    [data-testid="stNumberInput"] input[type="number"]::-webkit-inner-spin-button {
        -webkit-appearance: none;
        margin: 0;
    }
    
    /* Style the input field */
    [data-testid="stNumberInput"] {
        background: transparent;
    }
    
    /* Style the display value */
    .target-value {
        font-family: 'Segoe UI', sans-serif;
        font-size: 2.5em;
        font-weight: 800;
        color: #FF6B6B;
        text-align: center;
        padding: 20px;
        margin: 10px 0;
        text-shadow: 1px 1px 2px rgba(0,0,0,0.1);
    }
</style>
""", unsafe_allow_html=True)

# Cache data processing functions
@st.cache_data
def process_data(df):
    """Process and prepare data for the dashboard"""
    df = df.copy()
    
    # Convert dates and calculate time-based columns at once
    df['Expected Close Date'] = pd.to_datetime(df['Expected Close Date'], errors='coerce')
    df['Month'] = df['Expected Close Date'].dt.strftime('%B')
    df['Year'] = df['Expected Close Date'].dt.year
    df['Quarter'] = df['Expected Close Date'].dt.quarter.map({1: 'Q1', 2: 'Q2', 3: 'Q3', 4: 'Q4'})
    
    # Convert probability and calculate numeric values at once with safe null handling
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
    
    # Pre-calculate common flags and metrics with safe null handling
    df['Is_Won'] = df['Sales Stage'].str.contains('Won', case=False, na=False)
    df['Amount_Lacs'] = df['Amount'].fillna(0).div(100000).round(0).astype(int)
    df['Weighted_Amount'] = (df['Amount_Lacs'] * df['Probability_Num'] / 100).round(0).astype(int)
    
    return df

@st.cache_data
def calculate_team_metrics(df):
    """Calculate all team-related metrics at once"""
    # Calculate base metrics with safe null handling
    team_metrics = df.groupby('Sales Owner').agg({
        'Amount': lambda x: int(x[df['Is_Won'] & x.notna()].sum() / 100000) if len(x[df['Is_Won'] & x.notna()]) > 0 else 0,
        'Is_Won': 'sum',
        'Amount_Lacs': lambda x: int(x[~df['Is_Won'] & x.notna()].sum()) if len(x[~df['Is_Won'] & x.notna()]) > 0 else 0,
        'Weighted_Amount': lambda x: int(x[~df['Is_Won'] & x.notna()].sum()) if len(x[~df['Is_Won'] & x.notna()]) > 0 else 0
    }).reset_index()
    
    team_metrics.columns = ['Sales Owner', 'Closed Won', 'Closed Deals', 'Current Pipeline', 'Weighted Projections']
    
    # Fill NaN values with 0
    team_metrics = team_metrics.fillna(0)
    
    # Calculate Pipeline Deals with safe null handling
    pipeline_deals = df[~df['Is_Won']].groupby('Sales Owner').size()
    team_metrics['Pipeline Deals'] = team_metrics['Sales Owner'].map(pipeline_deals).fillna(0).astype(int)
    
    # Safe calculation of Win Rate to avoid division by zero
    total_deals = team_metrics['Closed Deals'] + team_metrics['Pipeline Deals']
    team_metrics['Win Rate'] = np.where(
        total_deals > 0,
        (team_metrics['Closed Deals'] / total_deals * 100).round(0),
        0
    ).astype(int)
    
    return team_metrics

@st.cache_data
def filter_dataframe(df, filters):
    """Apply filters to dataframe efficiently"""
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
            min_prob = float(prob_range[0])
            max_prob = float(prob_range[1].rstrip("%"))
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

def show_data_input():
    # Custom header
    st.markdown("""
        <div class="custom-header">
            <h1>Sales Performance Dashboard</h1>
            <p style="font-size: 1.2em; margin: 0;">Upload your sales data to begin analysis</p>
        </div>
    """, unsafe_allow_html=True)

    # Main upload section
    col1, col2 = st.columns([2, 1])
    
    with col1:
        st.markdown('<div class="upload-container">', unsafe_allow_html=True)
        uploaded_file = st.file_uploader(
            "Upload Sales Data",
            type=['xlsx', 'csv'],
            help="Upload your sales data file in Excel or CSV format"
        )
        st.markdown("</div>", unsafe_allow_html=True)
        
        if uploaded_file:
            try:
                if uploaded_file.name.endswith('.xlsx'):
                    # Handle Excel files
                    excel_file = pd.ExcelFile(uploaded_file)
                    sheet_name = st.selectbox("Select Worksheet", excel_file.sheet_names)
                    df = pd.read_excel(uploaded_file, sheet_name=sheet_name)
                else:
                    # Handle CSV files
                    df = pd.read_csv(uploaded_file)
                
                st.session_state.df = df
                st.success(f"Successfully loaded {len(df):,} records")
                
                # Preview the data
                st.subheader("Data Preview")
                st.dataframe(df.head(), use_container_width=True)
                
            except Exception as e:
                st.error(f"Error: {str(e)}")
    
    with col2:
        st.markdown("""
        <div class="info-box">
            <h4>Required Data Fields</h4>
            <ul>
                <li>Amount</li>
                <li>Sales Stage</li>
                <li>Expected Close Date</li>
                <li>Practice/Region</li>
            </ul>
            <h4>File Formats</h4>
            <ul>
                <li>Excel (.xlsx)</li>
                <li>CSV (.csv)</li>
            </ul>
        </div>
        """, unsafe_allow_html=True)

def show_overview():
    if st.session_state.df is None:
        st.warning("Please upload your sales data to view the dashboard")
        return
    
    st.title("Sales Performance Overview")
    
    df = st.session_state.df.copy()
    won_deals = df[df['Sales Stage'].str.contains('Won', case=False, na=False)]
    won_amount = won_deals['Amount'].sum() / 100000

    if 'Sales Stage' in df.columns and 'Amount' in df.columns:
        # Removed: Target vs Closed Won section

        # II. Practice
        st.markdown("""
            <div style='background: linear-gradient(90deg, #4A90E2 0%, #357ABD 100%); padding: 15px; border-radius: 10px; margin-bottom: 30px;'>
                <h3 style='color: white; margin: 0; text-align: center; font-size: 1.8em; font-weight: 600;'>Practice</h3>
            </div>
        """, unsafe_allow_html=True)
        
        if 'Practice' in df.columns:
            # Add practice filter
            practices = ['All'] + sorted(df['Practice'].dropna().unique().tolist())
            selected_practice = st.selectbox(
                "Select Practice",
                options=practices,
                key="practice_filter"
            )
            
            # Filter data based on selected practice
            if selected_practice != 'All':
                df = df[df['Practice'] == selected_practice]
            
            # Calculate practice metrics
            practice_metrics = df.groupby('Practice').agg({
                'Amount': lambda x: x[df['Sales Stage'].str.contains('Won', case=False, na=False)].sum() / 100000,
                'Sales Stage': lambda x: x[df['Sales Stage'].str.contains('Won', case=False, na=False)].count()
            }).reset_index()
            
            practice_metrics.columns = ['Practice', 'Closed Amount', 'Closed Deals']
            
            # Calculate total pipeline amount by practice (excluding closed won)
            pipeline_df = df[~df['Sales Stage'].str.contains('Won', case=False, na=False)]
            total_pipeline = pipeline_df.groupby('Practice')['Amount'].sum() / 100000
            practice_metrics['Total Pipeline'] = practice_metrics['Practice'].map(total_pipeline)
            
            # Calculate total deals by practice (excluding closed won)
            total_deals = pipeline_df.groupby('Practice').size()
            practice_metrics['Pipeline Deals'] = practice_metrics['Practice'].map(total_deals)
            
            # Sort practice metrics by Total Pipeline in descending order
            practice_metrics = practice_metrics.sort_values('Total Pipeline', ascending=False)
            
            # Create a comprehensive view
            col1, col2 = st.columns(2)
            
            with col1:
                # Practice-wise Pipeline Amount
                fig_pipeline = go.Figure()
                
                fig_pipeline.add_trace(go.Bar(
                    x=practice_metrics['Practice'],
                    y=practice_metrics['Total Pipeline'],
                    name='Pipeline',
                    text=practice_metrics['Total Pipeline'].apply(lambda x: f"₹{int(x)}L"),
                    textposition='outside',
                    textfont=dict(size=16, color='#4A90E2', family='Segoe UI', weight='bold'),
                    marker_color='#4A90E2',
                    marker_line=dict(color='#357ABD', width=2),
                    opacity=0.9
                ))
                
                fig_pipeline.add_trace(go.Bar(
                    x=practice_metrics['Practice'],
                    y=practice_metrics['Closed Amount'],
                    name='Closed Won',
                    text=practice_metrics['Closed Amount'].apply(lambda x: f"₹{int(x)}L"),
                    textposition='outside',
                    textfont=dict(size=16, color='#2ecc71', family='Segoe UI', weight='bold'),
                    marker_color='#2ecc71',
                    marker_line=dict(color='#27ae60', width=2),
                    opacity=0.9
                ))
                
                fig_pipeline.update_layout(
                    title=dict(
                        text="Practice-wise Pipeline vs Closed Won",
                        font=dict(size=22, family='Segoe UI', color='#2c3e50', weight='bold'),
                        x=0.5,
                        y=0.95,
                        xanchor='center',
                        yanchor='top'
                    ),
                    height=500,
                    barmode='group',
                    bargap=0.15,
                    bargroupgap=0.1,
                    xaxis_title=dict(
                        text="Practice",
                        font=dict(size=16, family='Segoe UI', color='#2c3e50', weight='bold'),
                        standoff=15
                    ),
                    yaxis_title=dict(
                        text="Amount (Lakhs)",
                        font=dict(size=16, family='Segoe UI', color='#2c3e50', weight='bold'),
                        standoff=15
                    ),
                    showlegend=True,
                    legend=dict(
                        font=dict(size=14, family='Segoe UI', color='#2c3e50'),
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99,
                        bgcolor='rgba(255, 255, 255, 0.8)',
                        bordercolor='rgba(0, 0, 0, 0.2)',
                        borderwidth=1
                    ),
                    font=dict(size=14, family='Segoe UI'),
                    xaxis=dict(
                        tickfont=dict(size=12, family='Segoe UI', color='#2c3e50'),
                        gridcolor='rgba(0, 0, 0, 0.1)'
                    ),
                    yaxis=dict(
                        tickfont=dict(size=12, family='Segoe UI', color='#2c3e50'),
                        gridcolor='rgba(0, 0, 0, 0.1)'
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(t=80, b=40, l=40, r=40)
                )
                
                st.plotly_chart(fig_pipeline, use_container_width=True)
            
            with col2:
                # Practice-wise Deal Count
                fig_deals = go.Figure()
                
                fig_deals.add_trace(go.Bar(
                    x=practice_metrics['Practice'],
                    y=practice_metrics['Pipeline Deals'],
                    name='Pipeline Deals',
                    text=practice_metrics['Pipeline Deals'],
                    textposition='outside',
                    textfont=dict(size=16, color='#4A90E2', family='Segoe UI', weight='bold'),
                    marker_color='#4A90E2',
                    marker_line=dict(color='#357ABD', width=2),
                    opacity=0.9
                ))
                
                fig_deals.add_trace(go.Bar(
                    x=practice_metrics['Practice'],
                    y=practice_metrics['Closed Deals'],
                    name='Closed Deals',
                    text=practice_metrics['Closed Deals'],
                    textposition='outside',
                    textfont=dict(size=16, color='#2ecc71', family='Segoe UI', weight='bold'),
                    marker_color='#2ecc71',
                    marker_line=dict(color='#27ae60', width=2),
                    opacity=0.9
                ))
                
                fig_deals.update_layout(
                    title=dict(
                        text="Practice-wise Pipeline vs Closed Deals",
                        font=dict(size=22, family='Segoe UI', color='#2c3e50', weight='bold'),
                        x=0.5,
                        y=0.95,
                        xanchor='center',
                        yanchor='top'
                    ),
                    height=500,
                    barmode='group',
                    bargap=0.15,
                    bargroupgap=0.1,
                    xaxis_title=dict(
                        text="Practice",
                        font=dict(size=16, family='Segoe UI', color='#2c3e50', weight='bold'),
                        standoff=15
                    ),
                    yaxis_title=dict(
                        text="Number of Deals",
                        font=dict(size=16, family='Segoe UI', color='#2c3e50', weight='bold'),
                        standoff=15
                    ),
                    showlegend=True,
                    legend=dict(
                        font=dict(size=14, family='Segoe UI', color='#2c3e50'),
                        yanchor="top",
                        y=0.99,
                        xanchor="right",
                        x=0.99,
                        bgcolor='rgba(255, 255, 255, 0.8)',
                        bordercolor='rgba(0, 0, 0, 0.2)',
                        borderwidth=1
                    ),
                    font=dict(size=14, family='Segoe UI'),
                    xaxis=dict(
                        tickfont=dict(size=12, family='Segoe UI', color='#2c3e50'),
                        gridcolor='rgba(0, 0, 0, 0.1)'
                    ),
                    yaxis=dict(
                        tickfont=dict(size=12, family='Segoe UI', color='#2c3e50'),
                        gridcolor='rgba(0, 0, 0, 0.1)'
                    ),
                    plot_bgcolor='white',
                    paper_bgcolor='white',
                    margin=dict(t=80, b=40, l=40, r=40)
                )
                
                st.plotly_chart(fig_deals, use_container_width=True)
            
            # Add practice summary metrics
            st.markdown("### Practice Summary")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                total_pipeline = practice_metrics['Total Pipeline'].sum()
                st.markdown(f"""
                    <div style='text-align: center; padding: 15px; background: #f8f9fa; border-radius: 10px;'>
                        <div class='metric-label'>Total Pipeline</div>
                        <div class='metric-value'>₹{int(total_pipeline)}L</div>
                        <div style='color: #666; font-size: 0.9em;'>Active pipeline value</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col2:
                total_deals = practice_metrics['Pipeline Deals'].sum()
                st.markdown(f"""
                    <div style='text-align: center; padding: 15px; background: #f8f9fa; border-radius: 10px;'>
                        <div class='metric-label'>Pipeline Deals</div>
                        <div class='metric-value'>{int(total_deals)}</div>
                        <div style='color: #666; font-size: 0.9em;'>Active opportunities</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col3:
                total_won = practice_metrics['Closed Deals'].sum()
                win_rate = (total_won / (total_won + total_deals) * 100) if (total_won + total_deals) > 0 else 0
                st.markdown(f"""
                    <div style='text-align: center; padding: 15px; background: #f8f9fa; border-radius: 10px;'>
                        <div class='metric-label'>Win Rate</div>
                        <div class='metric-value'>{int(win_rate)}%</div>
                        <div style='color: #666; font-size: 0.9em;'>{int(total_won)} won</div>
                    </div>
                """, unsafe_allow_html=True)
            
            with col4:
                avg_deal_size = practice_metrics['Closed Amount'].sum() / total_won if total_won > 0 else 0
                st.markdown(f"""
                    <div style='text-align: center; padding: 15px; background: #f8f9fa; border-radius: 10px;'>
                        <div class='metric-label'>Avg Deal Size</div>
                        <div class='metric-value'>₹{int(avg_deal_size)}L</div>
                        <div style='color: #666; font-size: 0.9em;'>Per won deal</div>
                    </div>
                """, unsafe_allow_html=True)
            
            # Add practice-wise summary table
            st.markdown("### Practice-wise Details")
            summary_data = practice_metrics.copy()
            summary_data['Win Rate'] = (summary_data['Closed Deals'] / (summary_data['Closed Deals'] + summary_data['Pipeline Deals']) * 100).round(1)
            
            # Format the summary table
            summary_data['Closed Amount'] = summary_data['Closed Amount'].apply(lambda x: f"₹{int(x)}L")
            summary_data['Total Pipeline'] = summary_data['Total Pipeline'].apply(lambda x: f"₹{int(x)}L")
            summary_data['Win Rate'] = summary_data['Win Rate'].apply(lambda x: f"{int(x)}%")
            
            st.dataframe(
                summary_data[['Practice', 'Closed Amount', 'Total Pipeline', 'Closed Deals', 'Pipeline Deals', 'Win Rate']],
                use_container_width=True
            )
        else:
            st.error("Practice column not found in the dataset")
    
    else:
        st.error("Required data fields (Sales Stage, Amount) not found in the dataset")

    # V. KritiKal Focus Areas
    st.markdown("""
        <div style='background: linear-gradient(90deg, #9b59b6 0%, #8e44ad 100%); padding: 15px; border-radius: 10px; margin-bottom: 30px;'>
            <h3 style='color: white; margin: 0; text-align: center; font-size: 1.8em; font-weight: 600;'>KritiKal Focus Areas</h3>
        </div>
    """, unsafe_allow_html=True)
    
    if 'KritiKal Focus Areas' in df.columns:
        # Calculate metrics by Focus Area
        focus_metrics = df.groupby('KritiKal Focus Areas').agg({
            'Amount': 'sum',
            'Sales Stage': lambda x: x[df['Sales Stage'].str.contains('Won', case=False, na=False)].count()
        }).reset_index()
        
        # Handle NaN values
        focus_metrics['KritiKal Focus Areas'] = focus_metrics['KritiKal Focus Areas'].fillna('Uncategorized')
        
        focus_metrics.columns = ['Focus Area', 'Total Amount', 'Closed Deals']
        focus_metrics['Total Amount'] = focus_metrics['Total Amount'] / 100000  # Convert to Lakhs
        
        # Calculate total deals and percentage share
        total_deals = df.groupby('KritiKal Focus Areas').size().reset_index()
        total_deals.columns = ['Focus Area', 'Total Deals']
        focus_metrics = focus_metrics.merge(total_deals, on='Focus Area', how='left')
        
        # Calculate percentage share
        total_amount = focus_metrics['Total Amount'].sum()
        focus_metrics['Share %'] = (focus_metrics['Total Amount'] / total_amount * 100).round(1)
        
        # Sort by Total Amount in descending order
        focus_metrics = focus_metrics.sort_values('Total Amount', ascending=False)
        
        # First show the summary table
        st.markdown("### Focus Areas Summary")
        summary_data = focus_metrics.copy()
        summary_data['Total Amount'] = summary_data['Total Amount'].apply(lambda x: f"₹{int(x)}L")
        summary_data['Share %'] = summary_data['Share %'].apply(lambda x: f"{int(x)}%")
        
        # Reset index to start from 1 and make it visible
        summary_data = summary_data.reset_index(drop=True)
        summary_data.index = summary_data.index + 1  # Start from 1 instead of 0
        
        st.dataframe(
            summary_data[['Focus Area', 'Total Amount', 'Share %', 'Total Deals', 'Closed Deals']],
            use_container_width=True
        )
        
        # Then show the donut chart
        st.markdown("### Focus Areas Distribution")
        fig_focus = go.Figure(data=[go.Pie(
            labels=focus_metrics['Focus Area'],
            values=focus_metrics['Total Amount'],
            hole=.4,
            textinfo='label+percent+value',
            texttemplate='%{label}<br>%{percent}<br>' + format_amount('%{value}'),
            textfont=dict(size=14, family='Segoe UI', weight='bold')
        )])
        
        fig_focus.update_layout(
            title=dict(
                text="Focus Areas Distribution",
                font=dict(size=22, family='Segoe UI', color='#2c3e50', weight='bold'),
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top'
            ),
            height=500,
            showlegend=True,
            legend=dict(
                font=dict(size=14, family='Segoe UI', color='#2c3e50'),
                yanchor="top",
                y=0.99,
                xanchor="right",
                x=0.99,
                bgcolor='rgba(255, 255, 255, 0.8)',
                bordercolor='rgba(0, 0, 0, 0.2)',
                borderwidth=1
            ),
            annotations=[dict(
                text=f"Total: ₹{int(total_amount)}L",
                font=dict(size=16, family='Segoe UI', weight='bold'),
                showarrow=False,
                x=0.5,
                y=0.5
            )]
        )
        
        st.plotly_chart(fig_focus, use_container_width=True)
    else:
        st.info("KritiKal Focus Areas column not found in the dataset")

    # V. Monthly Pipeline Trend
    st.markdown("""
        <div style='background: linear-gradient(90deg, #00b4db 0%, #0083b0 100%); padding: 15px; border-radius: 10px; margin-bottom: 30px;'>
            <h3 style='color: white; margin: 0; text-align: center; font-size: 1.8em; font-weight: 600;'>Monthly Pipeline Trend</h3>
        </div>
    """, unsafe_allow_html=True)
    
    if 'Expected Close Date' in df.columns and 'Amount' in df.columns and 'Sales Stage' in df.columns:
        # Convert Expected Close Date to datetime
        df['Expected Close Date'] = pd.to_datetime(df['Expected Close Date'], errors='coerce')
        
        # Create a selectbox for deal type
        deal_type = st.selectbox(
            "Select Deal Type",
            ["🌊 Pipeline", "🟢 Closed Won", "📦 All Deals"],
            index=0
        )
        
        # Filter data based on selection
        if deal_type == "🌊 Pipeline":
            filtered_df = df[~df['Sales Stage'].str.contains('Won', case=False, na=False)]
            color = '#00b4db'
        elif deal_type == "🟢 Closed Won":
            filtered_df = df[df['Sales Stage'].str.contains('Won', case=False, na=False)]
            color = '#2ecc71'
        else:  # All Deals
            filtered_df = df
            color = '#9b59b6'
        
        # Group by month and calculate metrics
        monthly_data = filtered_df.groupby(filtered_df['Expected Close Date'].dt.to_period('M')).agg({
            'Amount': 'sum',
            'Sales Stage': 'count'
        }).reset_index()
        
        monthly_data['Expected Close Date'] = monthly_data['Expected Close Date'].astype(str)
        monthly_data['Amount'] = monthly_data['Amount'] / 100000  # Convert to Lakhs
        
        # Create line chart
        fig_trend = go.Figure()
        
        fig_trend.add_trace(go.Scatter(
            x=monthly_data['Expected Close Date'],
            y=monthly_data['Amount'],
            mode='lines+markers',
            name=deal_type,
            line=dict(width=3, color=color),
            marker=dict(size=8, color=color),
            text=monthly_data['Amount'].apply(lambda x: f"₹{int(x)}L"),
            textposition='top center',
            textfont=dict(size=12, family='Segoe UI', weight='bold')
        ))
        
        fig_trend.update_layout(
            title=dict(
                text=f"{deal_type} Trend",
                font=dict(size=22, family='Segoe UI', color='#2c3e50', weight='bold'),
                x=0.5,
                y=0.95,
                xanchor='center',
                yanchor='top'
            ),
            height=500,
            showlegend=False,
            xaxis_title=dict(
                text="Month",
                font=dict(size=16, family='Segoe UI', color='#2c3e50', weight='bold'),
                standoff=15
            ),
            yaxis_title=dict(
                text="Amount (Lakhs)",
                font=dict(size=16, family='Segoe UI', color='#2c3e50', weight='bold'),
                standoff=15
            ),
            font=dict(size=14, family='Segoe UI'),
            xaxis=dict(
                tickfont=dict(size=12, family='Segoe UI', color='#2c3e50'),
                gridcolor='rgba(0, 0, 0, 0.1)'
            ),
            yaxis=dict(
                tickfont=dict(size=12, family='Segoe UI', color='#2c3e50'),
                gridcolor='rgba(0, 0, 0, 0.1)'
            ),
            plot_bgcolor='white',
            paper_bgcolor='white',
            margin=dict(t=80, b=40, l=40, r=40)
        )
        
        st.plotly_chart(fig_trend, use_container_width=True)
        
        # Add summary metrics
        col1, col2, col3 = st.columns(3)
        
        with col1:
            total_value = monthly_data['Amount'].sum()
            st.markdown(f"""
                <div style='text-align: center; padding: 15px; background: #f8f9fa; border-radius: 10px;'>
                    <div class='metric-label'>Total Value</div>
                    <div class='metric-value'>₹{int(total_value)}L</div>
                    <div style='color: #666; font-size: 0.9em;'>Overall</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col2:
            avg_monthly = monthly_data['Amount'].mean()
            st.markdown(f"""
                <div style='text-align: center; padding: 15px; background: #f8f9fa; border-radius: 10px;'>
                    <div class='metric-label'>Monthly Average</div>
                    <div class='metric-value'>₹{int(avg_monthly)}L</div>
                    <div style='color: #666; font-size: 0.9em;'>Per month</div>
                </div>
            """, unsafe_allow_html=True)
        
        with col3:
            total_deals = monthly_data['Sales Stage'].sum()
            st.markdown(f"""
                <div style='text-align: center; padding: 15px; background: #f8f9fa; border-radius: 10px;'>
                    <div class='metric-label'>Total Deals</div>
                    <div class='metric-value'>{int(total_deals)}</div>
                    <div style='color: #666; font-size: 0.9em;'>Number of deals</div>
                </div>
            """, unsafe_allow_html=True)
    else:
        st.info("Required columns (Expected Close Date, Amount, Sales Stage) not found in the dataset")

def show_sales_team():
    if st.session_state.df is None:
        st.warning("Please upload your sales data to view team information")
        return
    
    # Process data once with caching
    df = process_data(st.session_state.df)
    
    # Get unique team members
    team_members = sorted(df['Sales Owner'].dropna().unique().tolist())
    
    # Main content area with enhanced styling
    st.markdown("""
        <div style='
            background: linear-gradient(135deg, #1e3c72 0%, #2a5298 100%);
            padding: 25px;
            border-radius: 15px;
            margin-bottom: 25px;
            box-shadow: 0 4px 8px rgba(0,0,0,0.1);
        '>
            <h2 style='
                color: white;
                margin: 0;
                text-align: center;
                font-size: 2em;
                font-weight: 600;
                letter-spacing: 0.5px;
                text-shadow: 1px 1px 2px rgba(0,0,0,0.2);
            '>Sales Team Data</h2>
        </div>
    """, unsafe_allow_html=True)

    # Calculate metrics once
    metrics = calculate_team_metrics(df)
    
    # Display metrics with consistent styling
    col1, col2, col3, col4 = st.columns(4)
    
    metric_style = """
        text-align: center;
        padding: 20px;
        background: linear-gradient(135deg, {gradient});
        border-radius: 15px;
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
        margin: 10px 5px;
    """
    
    metric_text_style = """
        color: #FFFFFF;
        font-family: 'Segoe UI', sans-serif;
        font-size: 2.6em;
        font-weight: 800;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.4);
        margin: 15px 0;
        letter-spacing: 0.5px;
        -webkit-font-smoothing: antialiased;
    """
    
    label_style = """
        color: #FFFFFF;
        font-family: 'Segoe UI', sans-serif;
        font-size: 1.5em;
        font-weight: 800;
        margin-bottom: 12px;
        text-transform: uppercase;
        letter-spacing: 1px;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
        -webkit-font-smoothing: antialiased;
    """
    
    sublabel_style = """
        color: #FFFFFF;
        font-family: 'Segoe UI', sans-serif;
        font-size: 1.2em;
        font-weight: 700;
        margin-top: 8px;
        letter-spacing: 0.5px;
        text-shadow: 1px 1px 3px rgba(0,0,0,0.3);
        -webkit-font-smoothing: antialiased;
    """
    
    # Calculate total metrics
    total_pipeline = metrics['Current Pipeline'].sum()
    total_closed = metrics['Closed Won'].sum()
    total_closed_deals = metrics['Closed Deals'].sum()
    total_pipeline_deals = metrics['Pipeline Deals'].sum()
    
    with col1:
        st.markdown(f"""
            <div style='{metric_style.format(gradient="#2193b0 0%, #6dd5ed 100%")}'>
                <div style='{label_style}'>Pipeline Value</div>
                <div style='{metric_text_style}'>₹{int(total_pipeline)}L</div>
                <div style='{sublabel_style}'>Active opportunities</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown(f"""
            <div style='{metric_style.format(gradient="#11998e 0%, #38ef7d 100%")}'>
                <div style='{label_style}'>Closed Won</div>
                <div style='{metric_text_style}'>₹{int(total_closed)}L</div>
                <div style='{sublabel_style}'>Won opportunities</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col3:
        win_rate = round((total_closed_deals / (total_closed_deals + total_pipeline_deals) * 100), 1) if (total_closed_deals + total_pipeline_deals) > 0 else 0
        st.markdown(f"""
            <div style='{metric_style.format(gradient="#4e54c8 0%, #8f94fb 100%")}'>
                <div style='{label_style}'>Win Rate</div>
                <div style='{metric_text_style}'>{int(win_rate)}%</div>
                <div style='{sublabel_style}'>{int(total_closed_deals)} won</div>
            </div>
        """, unsafe_allow_html=True)
    
    with col4:
        avg_deal_size = round(total_closed / total_closed_deals, 1) if total_closed_deals > 0 else 0
        st.markdown(f"""
            <div style='{metric_style.format(gradient="#f12711 0%, #f5af19 100%")}'>
                <div style='{label_style}'>Avg Deal Size</div>
                <div style='{metric_text_style}'>₹{int(avg_deal_size)}L</div>
                <div style='{sublabel_style}'>Per won deal</div>
            </div>
        """, unsafe_allow_html=True)

    # Filters section with compact layout
    st.markdown("""
        <div style='padding: 15px; background: linear-gradient(to right, #f8f9fa, #e9ecef); border-radius: 10px; margin: 15px 0;'>
            <h4 style='color: #2a5298; margin: 0; font-size: 1.1em; font-weight: 600;'>🔍 Filters</h4>
        </div>
    """, unsafe_allow_html=True)

    # All filters in one row
    col1, col2, col3, col4, col5, col6, col7, col8 = st.columns(8)
    
    with col1:
        filters = {
            'selected_member': st.selectbox(
                "👤 Sales Owner",
                options=["All Team Members"] + team_members,
                key="team_member_filter"
            )
        }
    with col2:
        filters['search'] = st.text_input("🔍 Search", placeholder="Search...")
    with col3:
        # Define Indian fiscal year order
        fiscal_order = ['April', 'May', 'June', 'July', 'August', 'September', 
                       'October', 'November', 'December', 'January', 'February', 'March']
        
        # Get available months and sort them according to fiscal order
        available_months = df['Month'].dropna().unique().tolist()
        available_months.sort(key=lambda x: fiscal_order.index(x) if x in fiscal_order else len(fiscal_order))
        filters['month_filter'] = st.selectbox("📅 Month", options=["All Months"] + available_months)
    with col4:
        filters['quarter_filter'] = st.selectbox("📊 Quarter", options=["All Quarters", "Q1", "Q2", "Q3", "Q4"])
    with col5:
        filters['year_filter'] = st.selectbox("📅 Year", options=["All Years"] + sorted(df['Expected Close Date'].dt.year.unique().tolist()))
    with col6:
        # Probability filter with custom range option
        probability_options = ["All Probability", "0-25%", "26-50%", "51-75%", "76-100%", "Custom Range"]
        filters['probability_filter'] = st.selectbox("📈 Probability", options=probability_options)
        
        # Show custom range inputs when "Custom Range" is selected
        if filters['probability_filter'] == "Custom Range":
            col6a, col6b = st.columns(2)
            with col6a:
                min_prob = st.number_input("Min %", min_value=0, max_value=100, value=0, step=1)
            with col6b:
                max_prob = st.number_input("Max %", min_value=0, max_value=100, value=100, step=1)
            filters['custom_prob_range'] = f"{min_prob}-{max_prob}%"
    with col7:
        # Get unique sales stages and add custom options
        status_options = ["All Status", "Committed for the Month", "Upsides for the Month"]
        filters['status_filter'] = st.selectbox("🎯 Status", options=status_options)
        
        if filters['status_filter'] == "Committed for the Month":
            current_month = pd.Timestamp.now().strftime('%B')
            mask = (df['Month'] == current_month) & (df['Probability_Num'] > 75)
            filtered_df = df[mask]
        elif filters['status_filter'] == "Upsides for the Month":
            current_month = pd.Timestamp.now().strftime('%B')
            mask = (df['Month'] == current_month) & (df['Probability_Num'].between(25, 75))
            filtered_df = df[mask]
    with col8:
        filters['focus_filter'] = st.selectbox("🎯 Focus", options=["All Focus"] + sorted(df['KritiKal Focus Areas'].dropna().unique().tolist()))

    # Apply all filters at once
    filtered_df = filter_dataframe(df, filters)
    
    # Performance Metrics Row
    st.markdown("""
        <div style='margin-bottom: 20px;'>
            <h3 style='color: #2a5298; margin: 0; font-size: 1.4em; font-weight: 600;'>Performance Metrics</h3>
        </div>
    """, unsafe_allow_html=True)

    # Create columns for metrics
    m1, m2, m3 = st.columns(3)

    # Get the metrics values
    current_pipeline = filtered_df[~filtered_df['Is_Won']]['Amount_Lacs'].sum()
    weighted_projections = filtered_df[~filtered_df['Is_Won']]['Weighted_Amount'].sum()
    closed_won = filtered_df[filtered_df['Is_Won']]['Amount_Lacs'].sum()

    with m1:
        st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, #4A90E2 0%, #357ABD 100%);
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                text-align: center;
                height: 100%;
            '>
                <div style='color: white; font-size: 1.1em; font-weight: 600; margin-bottom: 8px;'>
                    🌊 Current Pipeline
                </div>
                <div style='color: white; font-size: 1.8em; font-weight: 800;'>
                    ₹{int(current_pipeline)}L
                </div>
            </div>
        """, unsafe_allow_html=True)

    with m2:
        st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, #6B5B95 0%, #846EA9 100%);
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                text-align: center;
            '>
                <div style='color: white; font-size: 1.1em; font-weight: 600; margin-bottom: 8px;'>
                    ⚖️ Weighted Projections
                </div>
                <div style='color: white; font-size: 1.8em; font-weight: 800;'>
                    ₹{int(weighted_projections)}L
                </div>
            </div>
        """, unsafe_allow_html=True)

    with m3:
        st.markdown(f"""
            <div style='
                background: linear-gradient(135deg, #2ECC71 0%, #27AE60 100%);
                padding: 20px;
                border-radius: 10px;
                box-shadow: 0 4px 6px rgba(0,0,0,0.1);
                text-align: center;
            '>
                <div style='color: white; font-size: 1.1em; font-weight: 600; margin-bottom: 8px;'>
                    💰 Closed Won
                </div>
                <div style='color: white; font-size: 1.8em; font-weight: 800;'>
                    ₹{int(closed_won)}L
                </div>
            </div>
        """, unsafe_allow_html=True)

    # Add some spacing after the metrics
    st.markdown("<div style='margin: 25px 0;'></div>", unsafe_allow_html=True)

    # Opportunities section with consistent styling
    st.markdown(f"""
        <div style='
            background: linear-gradient(to right, #f8f9fa, #e9ecef);
            padding: 20px;
            border-radius: 15px;
            margin: 25px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        '>
            <h3 style='
                color: #2a5298;
                margin: 0;
                font-size: 1.4em;
                font-weight: 600;
                font-family: "Segoe UI", sans-serif;
            '>Team Member Performance</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Reset index to create proper serial numbers
    filtered_df = filtered_df.reset_index(drop=True)
    filtered_df.index = filtered_df.index + 1  # Start from 1 instead of 0
    
    # Create a copy of the filtered dataframe with only required columns
    display_df = filtered_df[['Organization Name', 'Opportunity Name', 'Geography', 
                            'Expected Close Date', 'Probability', 'Amount', 
                            'Sales Owner', 'Pre-sales Technical Lead', 'Business Owner', 
                            'Type', 'KritiKal Focus Areas']].copy()
    
    # Rename columns for display
    display_df = display_df.rename(columns={
        'Amount': 'Amount (In Lacs)',
        'Pre-sales Technical Lead': 'Tech Owner',
        'Type': 'Hunting /farming'
    })
    
    # Convert Amount to Lacs and create numeric column for sorting
    display_df['Amount (In Lacs)'] = display_df['Amount (In Lacs)'].apply(lambda x: int(x/100000) if pd.notnull(x) else 0)
    display_df['Probability'] = display_df['Probability'].apply(format_percentage)
    display_df['Weighted Revenue (In Lacs)'] = display_df.apply(
        lambda row: int((row['Amount (In Lacs)']) * float(str(row['Probability']).rstrip('%'))/100) if pd.notnull(row['Amount (In Lacs)']) else 0, 
        axis=1
    )
    
    # Format dates
    display_df['Expected Close Date'] = pd.to_datetime(display_df['Expected Close Date']).dt.strftime('%d-%b-%Y')
    
    # Sort by Amount in descending order
    display_df = display_df.sort_values('Amount (In Lacs)', ascending=False)
    
    # Add S.No column after sorting
    display_df.index = range(1, len(display_df) + 1)
    display_df.index.name = 'S.No'
    
    # Format the display
    st.dataframe(
        display_df,
        column_config={
            'Amount (In Lacs)': st.column_config.NumberColumn(
                'Amount (In Lacs)',
                format="₹%d L",
                help="Amount in Lakhs"
            ),
            'Weighted Revenue (In Lacs)': st.column_config.NumberColumn(
                'Weighted Revenue (In Lacs)',
                format="₹%d L",
                help="Weighted Revenue in Lakhs"
            ),
            'Probability': st.column_config.TextColumn(
                'Probability',
                help="Probability of winning the deal"
            ),
            'Expected Close Date': st.column_config.TextColumn(
                'Expected Close Date',
                help="Expected closing date"
            )
        }
    )
    
    st.markdown("<div style='height: 25px;'></div>", unsafe_allow_html=True)  # Consistent spacing
    
    # Team performance table with consistent styling
    st.markdown("""
        <div style='
            background: linear-gradient(to right, #f8f9fa, #e9ecef);
            padding: 20px;
            border-radius: 15px;
            margin: 25px 0;
            box-shadow: 0 4px 8px rgba(0,0,0,0.05);
        '>
            <h3 style='
                color: #2a5298;
                margin: 0;
                font-size: 1.4em;
                font-weight: 600;
                font-family: "Segoe UI", sans-serif;
            '>Team Member Performance</h3>
        </div>
    """, unsafe_allow_html=True)
    
    # Calculate and display team metrics with rounded numbers
    team_metrics = df.groupby('Sales Owner').agg({
        'Amount': lambda x: round(x[df['Sales Stage'].str.contains('Won', case=False, na=False)].sum() / 100000, 1),
        'Sales Stage': lambda x: x[df['Sales Stage'].str.contains('Won', case=False, na=False)].count()
    }).reset_index()
    
    team_metrics.columns = ['Sales Owner', 'Closed Won', 'Closed Deals']
    
    # Calculate Current Pipeline
    pipeline_df = df[~df['Sales Stage'].str.contains('Won', case=False, na=False)]
    total_pipeline = round(pipeline_df.groupby('Sales Owner')['Amount'].sum() / 100000, 1)
    team_metrics['Current Pipeline'] = team_metrics['Sales Owner'].map(total_pipeline)
    
    # Calculate Weighted Projections
    def calculate_weighted_projection(owner):
        owner_pipeline = pipeline_df[pipeline_df['Sales Owner'] == owner]
        weighted_sum = sum((amount * prob / 100) 
                         for amount, prob in zip(owner_pipeline['Amount'], 
                                              owner_pipeline['Probability_Num']))
        return round(weighted_sum / 100000, 1)  # Convert to Lacs
    
    team_metrics['Weighted Projections'] = team_metrics['Sales Owner'].apply(calculate_weighted_projection)
    
    # Calculate Win Rate
    total_deals = pipeline_df.groupby('Sales Owner').size()
    team_metrics['Pipeline Deals'] = team_metrics['Sales Owner'].map(total_deals)
    team_metrics['Win Rate'] = round((team_metrics['Closed Deals'] / (team_metrics['Closed Deals'] + team_metrics['Pipeline Deals']) * 100), 1)
    team_metrics = team_metrics.sort_values('Current Pipeline', ascending=False)
    
    # Format the display data
    summary_data = team_metrics.copy()
    summary_data['Current Pipeline'] = summary_data['Current Pipeline'].apply(lambda x: f"₹{x:,}L")
    summary_data['Weighted Projections'] = summary_data['Weighted Projections'].apply(lambda x: f"₹{x:,}L")
    summary_data['Closed Won'] = summary_data['Closed Won'].apply(lambda x: f"₹{x:,}L")
    summary_data['Win Rate'] = summary_data['Win Rate'].apply(lambda x: f"{x}%")
    
    # Display the enhanced team performance table
    st.dataframe(
        summary_data[[
            'Sales Owner',
            'Current Pipeline',
            'Weighted Projections',
            'Closed Won',
            'Pipeline Deals',
            'Closed Deals',
            'Win Rate'
        ]],
        use_container_width=True
    )

def show_detailed():
    if st.session_state.df is None:
        st.warning("Please upload your sales data to view detailed information")
        return
    
    st.title("Detailed Sales Data")
    
    df = st.session_state.df
    
    # Search and filters
    search = st.text_input("Search", placeholder="Search in any field...")
    
    # Filter the dataframe based on search
    if search:
        mask = np.column_stack([df[col].astype(str).str.contains(search, case=False, na=False) 
                              for col in df.columns])
        df = df[mask.any(axis=1)]
    
    # Display the filtered dataframe
    st.dataframe(df, use_container_width=True)

def main():
    # Sidebar navigation
    with st.sidebar:
        st.title("Navigation")
        
        selected = st.radio(
            "Select View",
            options=["Data Input", "Overview", "Sales Team", "Detailed Data"],
            key="navigation"
        )
        
        st.session_state.current_view = selected.lower().replace(" ", "_")
    
    # Display the selected view
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
