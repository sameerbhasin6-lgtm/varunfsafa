import streamlit as st

try:
    import pdfplumber
except Exception as e:
    st.error(f"PDF dependency error: {e}")
    st.stop()

import pandas as pd
import re
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from transformers import pipeline

# --- PAGE CONFIGURATION ---
st.set_page_config(
    page_title="FinDetect AI | Hidden Liability Scanner",
    page_icon="🛡️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- CSS FOR "PRO" FEEL ---
st.markdown("""
    <style>
    .metric-card {background-color: #f0f2f6; padding: 20px; border-radius: 10px; border-left: 5px solid #4e8cff;}
    .risk-high {color: #ff4b4b; font-weight: bold;}
    .risk-med {color: #ffa421; font-weight: bold;}
    </style>
    """, unsafe_allow_html=True)

# --- 1. AI MODEL & LOGIC ---
@st.cache_resource
def load_model():
    # Using a smaller model to reduce memory usage on Cloud Free Tier
    return pipeline("zero-shot-classification", model="facebook/bart-large-mnli")

def get_yahoo_data(ticker):
    try:
        stock = yf.Ticker(ticker)
        bs = stock.balance_sheet
        debt = bs.loc['Total Debt'][0] if 'Total Debt' in bs.index else 0
        equity = bs.loc['Stockholders Equity'][0] if 'Stockholders Equity' in bs.index else 0
        return debt, equity
    except:
        return 0, 0

def extract_details(text):
    """Extracts Amount AND determines specific Category"""
    # 1. Extract Amount (Crores)
    amount = 0.0
    # Adjusted regex for pypdf which might lose some spacing
    pattern = r"([\d,]+(?:\.\d+)?)\s?(Crore|Cr|Lakh|L|Mn|Million|Billion)"
    match = re.search(pattern, text, re.IGNORECASE)
    
    if match:
        val_str = match.group(1).replace(",", "")
        unit = match.group(2).lower()
        try:
            val = float(val_str)
            if 'lakh' in unit or 'l' == unit: val /= 100
            elif 'mn' in unit or 'million' in unit: val /= 10
            elif 'billion' in unit: val *= 100
            amount = round(val, 2)
        except:
            amount = 0.0

    # 2. Determine Category based on keywords
    category = "Other Liability"
    text_lower = text.lower()
    if any(x in text_lower for x in ["tax", "income tax", "dispute", "demand", "excise", "customs"]):
        category = "🏛️ Legal/Tax Dispute"
    elif any(x in text_lower for x in ["guarantee", "letter of comfort", "standby", "indemnity"]):
        category = "🤝 Corporate Guarantee"
    elif any(x in text_lower for x in ["commitment", "capital", "export obligation"]):
        category = "🏗️ Capital Commitment"
    
    return amount, category

# --- 2. MAIN DASHBOARD UI ---
st.title("🛡️ FinDetect AI")
st.markdown("**Enterprise Off-Balance Sheet Analyzer (Ind-AS Compliant)**")

# SIDEBAR
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/2704/2704022.png", width=50)
    st.header("Analysis Parameters")
    ticker = st.text_input("Ticker Symbol", value="VEDL.NS")
    
    st.divider()
    st.subheader("SCAN SETTINGS")
    start_page = st.number_input("Start Page", value=200, step=10, help="Where do the Notes start?")
    pages_to_scan = st.number_input("Pages to Scan", value=50, step=10)
    
    st.divider()
    if st.button("🔄 Fetch Market Data", use_container_width=True):
        with st.spinner("Connecting to NSE..."):
            d, e = get_yahoo_data(ticker)
            st.session_state['rep_debt'] = d / 10000000 
            st.session_state['rep_equity'] = e / 10000000
    
    rep_debt = st.number_input("Reported Debt (₹ Cr)", value=st.session_state.get('rep_debt', 50000.0))
    rep_equity = st.number_input("Reported Equity (₹ Cr)", value=st.session_state.get('rep_equity', 30000.0))

# MAIN AREA
uploaded_file = st.file_uploader("📂 Upload Annual Report (PDF)", type="pdf")

if uploaded_file and st.button("🚀 START FORENSIC ANALYSIS", type="primary"):
    
    # ANALYSIS ENGINE
    classifier = load_model()
    labels = ["High Risk Liability", "Standard Text", "Asset"]
    
    # Expanded Keyword List
    triggers = ["guarantee", "letter of comfort", "undertaking", "unconsolidated", 
                "contingent liability", "disputed tax", "show cause", "claim", 
                "demand notice", "export obligation"]
    
    hits = []
    total_hidden = 0.0
    category_totals = {}

    progress_bar = st.progress(0)
    status_text = st.empty()
    
    # --- CHANGED TO PYPDF LOGIC HERE ---
    try:
        reader = pypdf.PdfReader(uploaded_file)
        max_pages = len(reader.pages)
        end_page = min(start_page + pages_to_scan, max_pages)
        scan_range = range(start_page, end_page)
        
        for i, page_num in enumerate(scan_range):
            progress_bar.progress((i + 1) / len(scan_range))
            status_text.caption(f"Scanning Page {page_num} for Ind-AS 37 Disclosures...")
            
            try:
                page = reader.pages[page_num]
                text = page.extract_text()
                
                if not text: continue
                
                # Cleanup text for better processing
                sentences = text.replace('\n', ' ').split('. ')
                
                for sent in sentences:
                    # HEURISTIC FILTER
                    if any(t in sent.lower() for t in triggers):
                         amt, cat = extract_details(sent)
                         if amt > 0:
                             # AI VERIFICATION
                             res = classifier(sent, labels)
                             if res['labels'][0] == "High Risk Liability" and res['scores'][0] > 0.4:
                                 total_hidden += amt
                                 # Update Category Totals
                                 category_totals[cat] = category_totals.get(cat, 0) + amt
                                 
                                 hits.append({
                                     "Page": page_num,
                                     "Category": cat,
                                     "Risk Context": sent[:120] + "...",
                                     "Amount (Cr)": amt,
                                     "Confidence": res['scores'][0]
                                 })
            except Exception as e:
                # Skip individual page errors
                continue
                
    except Exception as e:
        st.error(f"Error reading PDF: {e}")
            
    progress_bar.empty()
    status_text.empty()

    # --- 3. RESULTS DASHBOARD (TABS LAYOUT) ---
    
    # CALCULATIONS
    new_debt = rep_debt + total_hidden
    old_lev = rep_debt / rep_equity if rep_equity else 0
    new_lev = new_debt / rep_equity if rep_equity else 0
    impact_pct = ((new_lev - old_lev) / old_lev) * 100 if old_lev else 0

    st.success(f"Analysis Complete. Found ₹ {total_hidden:,.0f} Crores in hidden liabilities.")

    # TABS FOR ORGANIZATION
    tab1, tab2, tab3 = st.tabs(["📊 Executive Summary", "🍩 Risk Breakdown", "📝 Forensic Evidence"])

    with tab1:
        # METRICS ROW
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Reported Leverage", f"{old_lev:.2f}x", "Baseline")
        c2.metric("Hidden Debt Found", f"₹ {total_hidden:,.0f} Cr", f"{len(hits)} items detected", delta_color="inverse")
        c3.metric("Adjusted Leverage", f"{new_lev:.2f}x", f"+{impact_pct:.1f}% Increase", delta_color="inverse")
        c4.metric("Risk Rating", "HIGH" if impact_pct > 10 else "MODERATE", "AI Assessment")

        st.markdown("### 📉 Impact Visualization")
        fig_lev = go.Figure(data=[
            go.Bar(name='Reported', x=['Debt/Equity Ratio'], y=[old_lev], marker_color='#2ecc71', width=0.3),
            go.Bar(name='Hidden Risk', x=['Debt/Equity Ratio'], y=[new_lev - old_lev], marker_color='#e74c3c', width=0.3)
        ])
        fig_lev.update_layout(barmode='stack', title="Before vs. After AI Analysis", height=350)
        st.plotly_chart(fig_lev, use_container_width=True)

    with tab2:
        st.markdown("### 🔍 What composes this risk?")
        if category_totals:
            # DONUT CHART
            df_cat = pd.DataFrame(list(category_totals.items()), columns=['Category', 'Amount'])
            fig_pie = px.pie(df_cat, values='Amount', names='Category', hole=0.4, 
                             color_discrete_sequence=px.colors.qualitative.Pastel)
            fig_pie.update_traces(textinfo='percent+label')
            st.plotly_chart(fig_pie, use_container_width=True)
            
            # INSIGHT BOX
            max_cat = max(category_totals, key=category_totals.get)
            st.info(f"💡 **AI Insight:** The largest hidden risk source is **{max_cat}**. This typically indicates { 'aggressive tax positions' if 'Tax' in max_cat else 'high off-balance sheet exposure to subsidiaries'}.")
        else:
            st.write("No categorical data available.")

    with tab3:
        st.markdown("### 📑 Detailed Findings Log")
        if hits:
            df_hits = pd.DataFrame(hits)
            # Highlighting high value items
            st.dataframe(df_hits.style.background_gradient(subset=['Amount (Cr)'], cmap='Reds'), use_container_width=True)
            
            # DOWNLOAD BUTTON
            csv = df_hits.to_csv(index=False).encode('utf-8')
            st.download_button("📥 Download Forensic Report", csv, "risk_report.csv", "text/csv")
        else:
            st.warning("No significant risks found in the scanned page range.")


