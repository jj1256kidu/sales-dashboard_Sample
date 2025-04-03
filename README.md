# Sales Dashboard

An interactive dashboard built with Streamlit to visualize and analyze sales data.

## Features

- Data input from Excel file or Google Sheet
- Interactive filters for Practice, Quarter, and Hunting/Farming
- Key Performance Indicators (KPIs) display
- Practice-wise sales summary visualization
- Detailed deals table with export functionality
- All monetary values displayed in Lakhs (₹10^5 units)

## Local Setup

1. Clone the repository:
```bash
git clone https://github.com/yourusername/sales-dashboard.git
cd sales-dashboard
```

2. Create and activate a virtual environment (optional but recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install the required dependencies:
```bash
pip install -r requirements.txt
```

4. Run the dashboard locally:
```bash
streamlit run sales_dashboard.py
```

## Deployment to Streamlit Cloud

1. Push your code to GitHub:
```bash
git init
git add .
git commit -m "Initial commit"
git branch -M main
git remote add origin https://github.com/yourusername/sales-dashboard.git
git push -u origin main
```

2. Deploy to Streamlit Cloud:
   - Go to [share.streamlit.io](https://share.streamlit.io/)
   - Sign in with your GitHub account
   - Click "New app"
   - Select your repository, branch, and the main file (sales_dashboard.py)
   - Click "Deploy"

## Data Format Requirements

The input data should be in an Excel file or Google Sheet with a sheet named 'Raw_Data' containing the following columns:

- Organization Name
- Opportunity Name
- Geography
- Expected Close Date
- Probability
- Amount (in Lakhs)
- Sales Stage
- Practice
- Quarter
- Hunting/Farming
- Sales Owner
- Tech Owner

## Usage

1. Choose your data input method (Excel file or Google Sheet URL)
2. Enter your sales target in Lakhs
3. Use the filters to narrow down the data
4. View KPIs, practice-wise summary, and detailed deals table
5. Export filtered data to CSV if needed

## Project Structure

```
sales-dashboard/
├── .streamlit/
│   └── config.toml
├── sales_dashboard.py
├── requirements.txt
├── README.md
└── .gitignore
```

## Notes

- For Google Sheets, make sure the sheet is publicly accessible
- The sheet must contain a tab named 'Raw_Data'
- All monetary values should be in Lakhs (₹10^5 units)
- The dashboard is configured with a clean, corporate theme
- Maximum file upload size is set to 200MB 