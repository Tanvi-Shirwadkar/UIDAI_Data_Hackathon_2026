import streamlit as st
import os
from PIL import Image
import json
import pandas as pd
import plotly.express as px
from src.ai_insights import get_ai_insights

# ==========================================
# CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="UIDAI Identity Health Dashboard",
    page_icon="🇮🇳",
    layout="wide"
)


# Function to load image safely
def load_chart(filename):
    path = os.path.join("output", filename)
    if os.path.exists(path):
        return Image.open(path)
    else:
        return None


# ==========================================
# SIDEBAR
# ==========================================
st.sidebar.image("https://upload.wikimedia.org/wikipedia/en/c/cf/Aadhaar_Logo.svg", width=150)
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to:", ["Executive Summary", "Operational Analysis", "Risk & Security", "AI Interpretation"])

st.sidebar.info("Data Source: UIDAI Anonymised Dataset 2025")

# ==========================================
# MAIN PAGE
# ==========================================


if page == "Executive Summary":
    st.title("Aadhaar Identity Health Engine")
    st.markdown("### National Level Status Report")

    # Metrics Row
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Total Pincodes Analyzed", "32,168", "+1.2%")
    col2.metric("Critical Risk Districts", "14", "High Priority")
    col3.metric("Child Update Gap", "2.1M", "-5% YoY")
    col4.metric("Fraud Spikes Detected", "3", "Last 30 Days")

    st.divider()

    st.subheader("Aadhaar Identity Risk Map (District Level)")

    st.markdown(
        "Hover over a district to view its **Identity Health Index** and **Risk Category**."
    )

    @st.cache_data
    def load_data():
        df = pd.read_csv("output/final_health_index.csv")
        with open("data/india_district.geojson", "r", encoding="utf-8") as f:
            geojson = json.load(f)
        return df, geojson

    df, india_geojson = load_data()

    def clean_name(x):
        if pd.isna(x):
            return x
        return str(x).replace("*", "").strip().title()

    df["district_clean"] = df["district"].apply(clean_name)
    df["state_clean"] = df["state"].apply(clean_name)

    state_fix = {
        "Nct Of Delhi": "Delhi",
        "Jammu And Kashmir": "Jammu and Kashmir",
        "Jammu & Kashmir": "Jammu and Kashmir"
    }
    df["state_clean"] = df["state_clean"].replace(state_fix)

    df["risk_category_clean"] = df["risk_category"]

    zero_activity_mask = (
        (df["enrol_total_vol"] == 0) &
        (df["bio_total_vol"] == 0) &
        (df["demo_total_vol"] == 0)
    )

    df.loc[zero_activity_mask, "risk_category_clean"] = "No Data / Inactive"

    risk_colors = {
        "Critical Risk": "#d7191c",
        "Moderate": "#fdae61",
        "Healthy": "#1a9641",
        "No Data / Inactive": "#bdbdbd" # grey
    }

    fig = px.choropleth(
    df,
    geojson=india_geojson,
    locations="district_clean",
    featureidkey="properties.NAME_2",
    color="risk_category_clean",
    color_discrete_map=risk_colors,
    hover_name="district_clean",
    hover_data={
        "state_clean": True,
        "health_index": ":.1f",
        "risk_category_clean": True,
        "district_clean": False
    },
    labels={
        "state_clean": "State",
        "health_index": "Health Index",
        "risk_category_clean": "Risk Category"
    }
)

    fig.update_geos(fitbounds="locations", visible=False)
    fig.update_layout(margin={"r":0,"t":0,"l":0,"b":0})

    st.plotly_chart(fig, width="stretch")

    with st.expander("How to interpret this map"):
        st.markdown("""
        **What this shows**  
        Each district is coloured by its Aadhaar **Identity Health Risk Category**, derived from a composite health score.  
        ⚪ Grey — No recorded Aadhaar activity or insufficient data during the analysis period.

        **What is the Health Index?**  
        The **Health Index (0–100)** is a composite score that measures how well a district’s Aadhaar ecosystem is functioning.

        It combines:
        • **Enrolment performance** — how much Aadhaar activity the district handles  
        • **Operational stability** — consistency of activity over time  
        • **Biometric compliance** — especially child biometric updates  
        • **Infrastructure proxy** — overall system capacity  

        Higher scores indicate a more reliable, stable, and compliant identity system.

        **What is the Risk Category?**  
        Each district is classified into a risk group based on its Health Index and performance patterns:

        • 🔴 **Critical Risk** — High stress, weak compliance, or unstable operations  
        • 🟡 **Moderate** — Functional but inconsistent performance  
        • 🟢 **Healthy** — Strong volume handling with high stability and compliance  
        • ⚪ **No Data / Inactive** — No measurable Aadhaar activity in the analysis window  

        **How to use it**  
        • Hover over districts to inspect **State**, **Health Index**, and **Risk Category**  
        • Identify regional clusters of systemic risk  
        • Prioritize audits, mobile units, and policy action in high-risk zones
        """)
    # The Hero Chart (Chart 4)
    st.subheader("1. The Solution: Identity Health Clusters (ML)")
    
    st.write(
        "We used Unsupervised Machine Learning (K-Means) to classify every district based on Performance, Compliance, and Stability.")
    
    img = load_chart("4_Advanced_Health_Clusters.png")
    if img:
        st.image(img, width=700)
    else:
        st.error("Chart 4 not found. Please run main.py first.")

    st.info("Insight: High Aadhaar volume does not guarantee system health—compliance and stability matter more.")

    with st.expander("How to interpret this chart"):
        st.markdown("""
        **What this chart shows**  
        Each dot represents a district, clustered using Machine Learning based on Aadhaar volume, operational stability, and compliance quality.

        **How to read the axes**  
        • **X-axis (Total Volume – Log Scale):** Left = low-activity districts, Right = high-activity districts  
        • **Y-axis (Health Index 0–100):** Bottom = unstable or non-compliant systems, Top = stable and reliable systems   

        **Why this matters**  
        • High activity does not automatically imply a healthy identity system  
        • Some large districts process massive volumes but fail to maintain biometric freshness  

        **Key insight**  
        • Most large districts perform better due to stronger infrastructure  
        • The most important signals are large districts below the trend — they have scale but are failing  
        • These are the highest-value targets for audits, compliance checks, and fraud monitoring
        """)


elif page == "Operational Analysis":
    st.title("Operational Efficiency & Inequality")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Intra-District Inequality (Pincode Variance)")
        st.write(
            "**Chart 8:** Distribution of total Aadhaar workload across pincodes within a district, "
            "highlighting operational imbalance."
        )
        img = load_chart("8_Pincode_Variance.png")
        if img: st.image(img)

        st.info("Insight: Districts may appear functional overall, yet hide severe service overload at specific pincodes.")
        with st.expander("How to interpret this chart"):
            st.markdown("""
            **What this shows**  
            This chart visualizes how Aadhaar enrolment and update transactions are distributed across 
            pincodes within a single district.

            **Key observations**  
            • A small number of pincodes handle a disproportionately high workload.  
            • The high Coefficient of Variation (CV = 0.96) indicates severe intra-district imbalance.

            **Why this happens**  
            • Urban and high-density pincodes attract more transactions due to accessibility and awareness.  
            • Peripheral areas face lower demand or limited service access.

            **Why this matters**  
            • Uneven workload creates service pressure in high-demand areas.  
            • Underutilized capacity persists elsewhere.

            **Recommended action**  
            • Reallocate staff and kits at the pincode level.  
            • Deploy mobile enrolment units in high-load zones.  
            • Monitor variance metrics to detect early stress.
            """)

    with col2:
        st.subheader("Enrolment Temporal Consistency")
        st.write("**Chart 1:** Monthly enrolment trends showing operational stability vs. camps.")
        img = load_chart("EDA_1_Enrolment_Trend.png")
        if img: st.image(img)
        
        st.info("Insight: Aadhaar enrolment demand is not uniform—child enrolments are seasonal, adults are structural.")
        with st.expander("How to interpret this chart"):
            st.markdown("""
            **What this shows**  
            This chart compares monthly enrolment volumes for children and adults from March to December 2025.

            **Key observations**  
            • Child enrolments show strong seasonality with sharp peaks around August–September.  
            • Adult enrolments remain consistently low with only minor month-to-month variation.

            **Why this happens**  
            • Child enrolments are driven by academic admission cycles and school calendars.  
            • Adult enrolments are need-based and less dependent on fixed timelines.

            **Why this matters**  
            • Seasonal dependence makes child enrolment vulnerable to short-term disruptions.  
            • Adult enrolments provide a stable baseline demand across the year.

            **Recommended action**  
            • Increase enrolment capacity during school admission periods.  
            • Maintain steady infrastructure for adult enrolments year-round.
            """)


elif page == "Risk & Security":
    st.title("Fraud & Compliance Sentinel")

    st.subheader("Automated Fraud Detection (Security Sentinel)")
    st.write(
        "**Chart 7:** A Z-score–based algorithm that automatically flags abnormal "
        "daily Aadhaar transaction spikes (> 3σ)."
    )

    img = load_chart("7_Anomaly_Sentinel.png")
    if img: st.image(img, width=700)

    st.info("Insight: The Security Sentinel isolates true anomalies without noise, enabling proactive fraud response.")

    with st.expander("How to interpret this chart"):
        st.markdown("""
        **What this shows**  
        This chart tracks daily Aadhaar transaction volumes and highlights days that deviate sharply from historical patterns.

        **Key observation**  
        • A massive spike on **01 March** is flagged as a **>3σ anomaly**  
        • All other days remain within normal statistical bounds  

        **Why this spike is suspicious**  
        • Not gradual or seasonal  
        • Far exceeds operational noise  
        • Extremely unlikely to occur naturally  

        **Why normal days are not flagged**  
        • Routine fluctuations stay within ±1σ to ±2σ  
        • The algorithm is calibrated to avoid false alarms  

        **Why this matters**  
        Early detection enables rapid audits, investigation, and damage containment before systemic impact.

        **Why Z-score**  
        • Explainable and statistically sound  
        • Unsupervised and scalable  
        • Low false-positive rate  
        """)

    st.divider()

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Biometric Splits")

        st.write(
            "**Chart 2:** Distribution of biometric updates between children (5–17) "
            "and adults (18+) across the dataset."
        )

        img = load_chart("EDA_2_Bio_Split.png")
        if img: st.image(img)

        st.info("Insight: Equal biometric updates for children and adults are a warning sign—children should dominate update volume.")

        with st.expander("How to interpret this chart"):
            st.markdown("""
            **What this shows**  
            This chart compares the share of biometric updates performed for children versus adults.

            **Why this matters**  
            • Children’s biometrics change rapidly and require periodic updates  
            • Adults require far fewer updates over time  

            **Key insight**  
            • A near-50 split is not healthy parity  
            • It signals systemic under-updating of child biometrics  

            **What this indicates**  
            • Silent accumulation of outdated child records  
            • Increased risk of future authentication failures  
            • Gaps in awareness, access, or operational prioritization

            This serves as a diagnostic check before deeper district-level risk analysis,
            motivating the identification of Child Identity Risk Zones.
            """)

    with col2:
        st.subheader("Child Identity Risk Zones")

        st.write(
            "**Chart 5:** Districts where children are enrolled in Aadhaar "
            "but fail to complete mandatory biometric updates."
        )

        img = load_chart("5_child_risk_gap.png")
        if img: st.image(img)

        st.info("Insight: Aadhaar enrolment without biometric updates creates silent identity failure among children.")

        with st.expander("How to interpret this chart"):
            st.markdown("""
            **What this shows**  
            This chart highlights districts with large gaps between child enrolments and biometric updates, indicating incomplete or outdated identities.

            **Why this is critical**  
            • Children’s biometrics change as they grow  
            • Without updates, Aadhaar authentication reliability degrades  
            • These identities exist but fail at the point of use  

            **Why certain districts are high-risk**  
            • Strong enrolment drives but weak follow-through  
            • High migration and urban churn  
            • Geographic barriers and limited access to update centers  

            **Why this matters**  
            • Service denial in PDS, DBT, and scholarships  
            • Increased risk of identity misuse or proxy authentication  
            • Overestimation of true identity coverage  

            **Recommended action**  
            • Age-triggered biometric update reminders  
            • Mobile update units in high-gap districts  
            • Shift district KPIs from enrolment counts to update completion
            """)


elif page == "AI Interpretation":
    st.title(" AI-Powered Business Insights for UIDAI")
    
    st.divider()
    
    # Check if API key is configured
    if not os.path.exists(".env"):
        st.error(" **Configuration Required**")
        st.stop()
    
    # Generate insights button
    if st.button(" Generate AI Business Insights", type="primary", use_container_width=True):
        with st.spinner(" AI is analyzing all graphs and generating comprehensive insights for UIDAI..."):
            insights, error = get_ai_insights()
            
            if error:
                st.error(f"**Error:** {error}")
                st.markdown("""
                **Troubleshooting Steps:**
                1. Check your `.env` file exists and contains: `api_key=your_actual_key`
                2. Verify your API key from [Groq Console](https://console.groq.com/keys)
                3. Ensure you have internet connectivity
                4. Check if you have remaining API quota
                """)
            else:
                st.success(" AI insights generated successfully!")
                st.markdown(insights)
                
            
    else:
        # Show preview of what will be generated
        st.info(" Click the button above to generate AI-powered insights")
        
        
        st.divider()
        
        st.markdown("###  Based on Analysis of:")
        
        metrics_cols = st.columns(3)
        
        with metrics_cols[0]:
            st.metric("Graphs Analyzed", "6", "Comprehensive")
            st.caption("All visualization insights")
        
        with metrics_cols[1]:
            st.metric("Districts Covered", "32,168", "Pincodes")
            st.caption("Complete national coverage")
        
        with metrics_cols[2]:
            st.metric("Data Points", "2M+", "Records")
            st.caption("Enrolment + Biometric + Demographic")
        
        st.divider()
        
        with st.expander(" Individual Graph Insights Used"):
            st.markdown("""
            1. **Enrolment Temporal Consistency** - Seasonal patterns and capacity planning
            2. **Biometric Splits** - Child vs adult update distribution analysis
            3. **Identity Health Clusters (ML)** - District-level system health classification
            4. **Child Identity Risk Zones** - Enrolment-update gap analysis
            5. **Anomaly Detection** - Fraud spike identification
            6. **Pincode Variance** - Intra-district resource distribution
            
            The AI combines insights from ALL graphs to provide holistic recommendations.
            """)