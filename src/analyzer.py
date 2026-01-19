import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import os
from sklearn.preprocessing import MinMaxScaler
from sklearn.cluster import KMeans
from src.config import DATA_DIR, OUTPUT_DIR


# =============================================================================
# UTILITY: Load and Clean (Standardized)
# =============================================================================
def load_and_combine_chunks(file_pattern):
    search_path = os.path.join(DATA_DIR, file_pattern)
    files = glob.glob(search_path)
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except:
            pass
    if not dfs: return pd.DataFrame()
    return pd.concat(dfs, ignore_index=True)


def clean_standardize(df, prefix):
    if df.empty: return df
    df.columns = df.columns.str.lower().str.strip()

    # 1. Standardize Column Names (Domain Separation)
    rename_map = {}
    if prefix == 'enrol':
        rename_map = {'age_0_5': 'enrol_infant', 'age_5_17': 'enrol_child', 'age_18_greater': 'enrol_adult'}
    elif prefix == 'bio':
        rename_map = {'bio_age_5_17': 'bio_child', 'bio_age_17_': 'bio_adult'}
    elif prefix == 'demo':
        rename_map = {'demo_age_5_17': 'demo_child', 'demo_age_17_': 'demo_adult'}

    df = df.rename(columns=rename_map)

    # 2. String Normalization (Crucial for Merging)
    df['state'] = df['state'].str.title().str.strip()
    # Garbage Filter: Remove states with numbers (Data Quality Control)
    df = df[~df['state'].str.contains(r'\d', regex=True, na=False)]

    state_map = {'Westbengal': 'West Bengal', 'Orissa': 'Odisha', 'Uttaranchal': 'Uttarakhand', 'Delhi': 'NCT Of Delhi', 'Tamilnadu': 'Tamil Nadu'}
    df['state'] = df['state'].replace(state_map)

    # 3. Temporal Standardization
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')
    df['month'] = df['date'].dt.to_period('M')

    return df


# =============================================================================
# STEP 3: INDEPENDENT EDA (Before Processing)
# =============================================================================
def run_independent_eda(df_enrol, df_bio, df_demo):
    """
    Generates exploratory plots on raw data distributions.
    Reference: Tukey, J. W. (1977). Exploratory Data Analysis.
    """
    print("Running Independent EDA (Step 3)...")

    # A. Enrolment Trend (Time Series Decomposition check)
    if not df_enrol.empty:
        trend = df_enrol.groupby('month')[['enrol_child', 'enrol_adult']].sum()
        plt.figure(figsize=(10, 5))
        trend.plot(kind='line', marker='o')
        plt.title('Enrolment Temporal Consistency')
        plt.xlabel('Month')
        plt.ylabel('Volume')
        plt.grid(True, alpha=0.3)
        plt.savefig(os.path.join(OUTPUT_DIR, 'EDA_1_Enrolment_Trend.png'))
        plt.close()

    # B. Biometric Gap (Child vs Adult Ratio)
    if not df_bio.empty:
        bio_sum = df_bio[['bio_child', 'bio_adult']].sum()
        plt.figure(figsize=(6, 6))
        bio_sum.plot(kind='pie', autopct='%1.1f%%', colors=['#ff9999', '#66b3ff'])
        plt.title('Biometric Update Distribution (Target vs Actual)')
        plt.savefig(os.path.join(OUTPUT_DIR, 'EDA_2_Bio_Split.png'))
        plt.close()


# =============================================================================
# STEP 1 & 4: PINCODE FEATURE ENGINEERING (The "Core Layer")
# =============================================================================
def process_domain_features(df, prefix, val_cols):
    """
    Calculates statistical moments (Sum, Std) at Pincode level.
    This preserves granular signal without memory explosion.
    """
    if df.empty: return pd.DataFrame()

    # Group by Pincode to get spatial features
    # We aggregate TIME (Month) here to get Pincode Stability
    stats = df.groupby(['state', 'district', 'pincode'])[val_cols].agg(['sum', 'std']).reset_index()

    # Flatten MultiIndex columns
    stats.columns = ['state', 'district', 'pincode'] + [f'{c}_{stat}' for c in val_cols for stat in ['sum', 'std']]

    # Calculate Domain Total for this pincode
    sum_cols = [c for c in stats.columns if 'sum' in c]
    stats[f'{prefix}_total_vol'] = stats[sum_cols].sum(axis=1)

    # Calculate Stability (Inverse Coefficient of Variation)
    # CV = sigma / mu. Stability = 1 / CV.
    std_cols = [c for c in stats.columns if 'std' in c]
    total_std = stats[std_cols].sum(axis=1)
    # Add epsilon to avoid div by zero
    stats[f'{prefix}_stability'] = np.where(stats[f'{prefix}_total_vol'] > 0,
                                            1 / ((total_std / (stats[f'{prefix}_total_vol'] + 1)) + 0.1),
                                            0)
    return stats


def load_and_engineer_data():
    print("Building Pincode-Level Feature Set...")

    # 1. Load Raw
    raw_enrol = load_and_combine_chunks("api_data_aadhar_enrolment_*.csv")
    raw_bio = load_and_combine_chunks("api_data_aadhar_biometric_*.csv")
    raw_demo = load_and_combine_chunks("api_data_aadhar_demographic_*.csv")

    # 2. Clean
    df_enrol = clean_standardize(raw_enrol, 'enrol')
    df_bio = clean_standardize(raw_bio, 'bio')
    df_demo = clean_standardize(raw_demo, 'demo')

    # 3. Run EDA
    run_independent_eda(df_enrol, df_bio, df_demo)

    # 4. Feature Engineering (Per Domain)
    # This replaces the "Total Activity" sum. We treat them separately.
    feat_enrol = process_domain_features(df_enrol, 'enrol', ['enrol_child', 'enrol_adult'])
    feat_bio = process_domain_features(df_bio, 'bio', ['bio_child', 'bio_adult'])
    feat_demo = process_domain_features(df_demo, 'demo', ['demo_child', 'demo_adult'])

    # 5. Merge Features (Pincode Level)
    # Outer Join on spatial keys
    keys = ['state', 'district', 'pincode']
    master = feat_enrol

    if master.empty:
        master = feat_bio
    elif not feat_bio.empty:
        master = pd.merge(master, feat_bio, on=keys, how='outer')

    if master.empty:
        master = feat_demo
    elif not feat_demo.empty:
        master = pd.merge(master, feat_demo, on=keys, how='outer')

    master = master.fillna(0)
    return master


# =============================================================================
# STEP 5: REBUILD INDEX & CLUSTERING
# =============================================================================
def generate_health_index(df):
    """
    Constructs the Identity Health Index using Weighted Linear Combination.
    Weights: 0.30 Enrol, 0.35 Bio, 0.25 Demo, 0.10 Infra
    """
    print("Generating Weighted Health Index (Step 5)...")

    # We aggregate to District Level for the Final Index Reporting
    dist_stats = df.groupby(['state', 'district']).agg({
        'enrol_total_vol': 'sum',
        'enrol_stability': 'mean',  # Mean stability of pincodes in district
        'bio_total_vol': 'sum',
        'bio_stability': 'mean',
        'demo_total_vol': 'sum',
        'enrol_child_sum': 'sum',
        'bio_child_sum': 'sum'
    }).reset_index()

    scaler = MinMaxScaler()

    # A. Access Score (Enrolment)
    # Weight: 0.30
    # Note: double brackets [['col']] return a DataFrame, so no reshape needed
    s_enrol_vol = scaler.fit_transform(dist_stats[['enrol_total_vol']]).flatten()
    s_enrol_stab = scaler.fit_transform(dist_stats[['enrol_stability']]).flatten()
    score_enrol = 0.5 * s_enrol_vol + 0.5 * s_enrol_stab

    # B. Compliance Score (Biometric)
    # Weight: 0.35
    # Critical Feature: Child Bio / Child Enrol Ratio
    mbu_ratio = np.where(dist_stats['enrol_child_sum'] > 0,
                         dist_stats['bio_child_sum'] / dist_stats['enrol_child_sum'], 0)
    # mbu_ratio is already a numpy array, so reshape works
    s_mbu = scaler.fit_transform(mbu_ratio.reshape(-1, 1)).flatten()
    score_bio = s_mbu

    # C. Accuracy Score (Demographic)
    # Weight: 0.25
    s_demo_vol = scaler.fit_transform(dist_stats[['demo_total_vol']]).flatten()
    score_demo = s_demo_vol

    # D. Infrastructure Score (Proxy)
    # Weight: 0.10
    total_capacity = dist_stats['enrol_total_vol'] + dist_stats['bio_total_vol'] + dist_stats['demo_total_vol']

    # --- THE FIX IS HERE ---
    # Convert Series to Values (NumPy) before reshaping
    score_infra = scaler.fit_transform(total_capacity.values.reshape(-1, 1)).flatten()
    # -----------------------

    # FINAL FORMULA
    dist_stats['health_index'] = (
                                         0.30 * score_enrol +
                                         0.35 * score_bio +
                                         0.25 * score_demo +
                                         0.10 * score_infra
                                 ) * 100

    # Clustering (K-Means)
    X = np.column_stack((score_enrol, score_bio, score_demo))
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    dist_stats['cluster'] = kmeans.fit_predict(X)

    # Rank Clusters
    cluster_rank = dist_stats.groupby('cluster')['health_index'].mean().sort_values().index
    rank_map = {cluster_rank[0]: 'Critical Risk', cluster_rank[1]: 'Moderate', cluster_rank[2]: 'Healthy'}
    dist_stats['risk_category'] = dist_stats['cluster'].map(rank_map)

    # Visualization
    plt.figure(figsize=(10, 6))
    sns.scatterplot(data=dist_stats, x='enrol_total_vol', y='health_index', hue='risk_category', palette='RdYlGn')
    plt.xscale('log')
    plt.title('Aadhaar Identity Health Index (Multi-Factor)')
    plt.xlabel('Total Volume (Log Scale)')
    plt.ylabel('Health Index (0-100)')
    plt.savefig(os.path.join(OUTPUT_DIR, '4_Advanced_Health_Clusters.png'))
    plt.close()

    return dist_stats.sort_values('health_index')


def analyze_mbu_gap(df):
    """
    Visualizes Child Lifecycle Failure (Enrolment vs Update Gap).
    """
    print("Analyzing Child Risks...")
    if 'enrol_child_sum' not in df.columns: return

    # Aggregate Pincode features to District for plotting
    dist = df.groupby('district')[['enrol_child_sum', 'bio_child_sum']].sum().reset_index()
    dist['gap'] = dist['enrol_child_sum'] - dist['bio_child_sum']
    top_risk = dist.sort_values('gap', ascending=False).head(10)

    plt.figure(figsize=(10, 5))
    sns.barplot(data=top_risk, x='gap', y='district', hue='district', palette='Reds_r', legend=False)
    plt.title('Child Identity Risk Zones (Enrolment vs Update Gap)')
    plt.xlabel('Gap Count (Enrolled but not Updated)')
    plt.savefig(os.path.join(OUTPUT_DIR, '5_child_risk_gap.png'))
    plt.close()


def analyze_fraud_spikes():
    """
    (Chart 7) The 'Fraud Sentinel' - Z-Score Anomaly Detection
    Detects specific dates with impossible spikes (>3 Sigma).
    """
    print("Running Fraud Sentinel (Anomaly Detection)...")

    # 1. Load ONLY Date and Counts (RAM-Safe Mode)
    # We ignore pincodes/districts here to keep it light
    files = glob.glob(os.path.join(DATA_DIR, "api_data_aadhar_demographic_*.csv"))
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f, usecols=['date', 'demo_age_5_17', 'demo_age_17_'])
            dfs.append(df)
        except:
            pass

    if not dfs:
        print("Skipping Sentinel - No Data Found.")
        return

    raw_df = pd.concat(dfs, ignore_index=True)
    raw_df['date'] = pd.to_datetime(raw_df['date'], format='%d-%m-%Y', errors='coerce')

    # 2. Aggregate to National Daily Volume
    daily_ts = raw_df.groupby('date')[['demo_age_5_17', 'demo_age_17_']].sum().sum(axis=1).reset_index()
    daily_ts.columns = ['date', 'total_vol']

    # 3. Calculate Z-Score (The "Anomaly" Math)
    mean_vol = daily_ts['total_vol'].mean()
    std_vol = daily_ts['total_vol'].std()
    daily_ts['z_score'] = (daily_ts['total_vol'] - mean_vol) / std_vol

    # 4. Detect Anomalies (> 3 Standard Deviations)
    anomalies = daily_ts[daily_ts['z_score'] > 3]
    print(f"   [!] DETECTED {len(anomalies)} SUSPICIOUS DAYS (>3 Sigma)")

    # 5. Plot the Sentinel Chart
    plt.figure(figsize=(12, 6))

    # Normal Flow (Grey)
    plt.plot(daily_ts['date'], daily_ts['total_vol'], color='gray', alpha=0.5, label='Normal Daily Flow')

    # Anomalies (Red)
    if not anomalies.empty:
        plt.scatter(anomalies['date'], anomalies['total_vol'], color='red', s=100, label='Anomaly (>3σ)', zorder=5)

        # Label the biggest spike automatically
        max_spike = anomalies.loc[anomalies['total_vol'].idxmax()]
        plt.annotate(f"SUSPICIOUS SPIKE\n{max_spike['date'].strftime('%d-%b')}",
                     (max_spike['date'], max_spike['total_vol']),
                     xytext=(10, 10), textcoords='offset points',
                     arrowprops=dict(arrowstyle="->", color='red', lw=2))

    plt.title('Security Sentinel: Automated Fraud Spike Detection')
    plt.ylabel('Daily Volume')
    plt.xlabel('Timeline')
    plt.legend()
    plt.grid(True, alpha=0.3)

    plt.savefig(os.path.join(OUTPUT_DIR, '7_Anomaly_Sentinel.png'))
    plt.close()


def analyze_pincode_variance(df):
    """
    (Chart 8) Pincode Variance Plot
    NOW calculates Total Load (Enrol + Bio + Demo) for true accuracy.
    """
    print("Generating Pincode Variance Analysis...")

    # 1. Create a Total Load Column (The Fix)
    df['total_load'] = df['enrol_total_vol'] + df['bio_total_vol'] + df['demo_total_vol']

    # 2. Pick the busiest district based on TOTAL activity, not just Enrolment
    top_district = df.groupby(['state', 'district'])['total_load'].sum().idxmax()
    state, dist_name = top_district

    subset = df[(df['state'] == state) & (df['district'] == dist_name)].copy()
    if subset.empty: return

    # 3. Sort by Total Load
    subset = subset.sort_values('total_load', ascending=False)

    plt.figure(figsize=(12, 6))

    # Plot Total Load
    sns.barplot(data=subset.head(30), x='pincode', y='total_load', palette='viridis')

    plt.title(f'Intra-District Inequality: {dist_name}, {state}\n(Total Operational Load)', fontsize=14)
    plt.ylabel('Total Transactions (Enrol + Update)')
    plt.xlabel('Pincode')
    plt.xticks(rotation=45)
    plt.grid(axis='y', alpha=0.3)

    # Annotation
    std_dev = subset['total_load'].std()
    mean_val = subset['total_load'].mean()
    cv = std_dev / mean_val if mean_val > 0 else 0

    plt.figtext(0.15, 0.8, f"High Service Variance (CV = {cv:.2f})\nWorkload is not distributed evenly.",
                bbox={"facecolor": "white", "alpha": 0.8, "pad": 5})

    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, '8_Pincode_Variance.png'))
    plt.close()
# You must add analyze_inequality to the import list in main.py!



