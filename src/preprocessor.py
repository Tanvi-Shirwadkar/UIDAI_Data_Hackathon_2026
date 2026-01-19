import pandas as pd
import glob
import os

# Define where your files are
DATA_DIR = "data/"


def load_and_combine_chunks(file_pattern):
    """
    Step 1: Glues multiple CSV parts into one DataFrame.
    Example: 'enrolment_*.csv' -> One Enrolment DF
    """
    search_path = os.path.join(DATA_DIR, file_pattern)
    files = glob.glob(search_path)

    if not files:
        print(f"WARNING: No files found for {file_pattern}")
        return pd.DataFrame()  # Return empty if missing

    print(f"Combining {len(files)} files for pattern: {file_pattern}...")
    dfs = []
    for f in files:
        try:
            df = pd.read_csv(f)
            dfs.append(df)
        except Exception as e:
            print(f"Error reading {f}: {e}")

    return pd.concat(dfs, ignore_index=True)


def clean_domain_df(df, domain_prefix):
    """
    Step 2: Cleans a specific domain (Demo, Bio, or Enrol).
    """
    if df.empty: return df

    # A. Standardize Column Names
    # (e.g., rename 'Age_0_5' to 'enrol_infant' so it doesn't clash later)
    # You need to adjust these column names based on your actual CSV headers!
    df.columns = df.columns.str.lower().str.strip()

    # Example Renaming Logic (Customize this part!)
    if domain_prefix == 'enrol':
        df = df.rename(columns={'age_0_5': 'enrol_infant', 'age_5_17': 'enrol_child', 'age_18_greater': 'enrol_adult'})
    elif domain_prefix == 'bio':
        df = df.rename(columns={'bio_age_5_17': 'bio_child', 'bio_age_17_': 'bio_adult'})
    elif domain_prefix == 'demo':
        df = df.rename(columns={'demo_age_5_17': 'demo_child', 'demo_age_17_': 'demo_adult'})

    # B. Fix State Names (CRITICAL: Must apply to all 3 DFs)
    df['state'] = df['state'].str.title().str.strip()
    state_map = {
        'Westbengal': 'West Bengal', 'West  Bengal': 'West Bengal',
        'Orissa': 'Odisha', 'Pondicherry': 'Puducherry',
        'Uttaranchal': 'Uttarakhand', 'Delhi': 'NCT Of Delhi'
    }
    df['state'] = df['state'].replace(state_map)

    # C. Date & Time Aggregation
    df['date'] = pd.to_datetime(df['date'], format='%d-%m-%Y', errors='coerce')  # Handle errors
    df['month'] = df['date'].dt.to_period('M')

    # D. Aggregate to District-Month level (To prepare for merging)
    # We group by State, District, Month so the rows match perfectly
    group_cols = ['state', 'district', 'month']
    numeric_cols = [c for c in df.columns if c not in group_cols and c != 'date' and c != 'pincode']

    df_agg = df.groupby(group_cols)[numeric_cols].sum().reset_index()

    return df_agg


def generate_master_dataset():
    """
    Step 3: Orchestrates the whole process and Merges everything.
    """
    # 1. Load & Combine Chunks
    print("--- Phase 1: Loading Chunks ---")
    raw_enrol = load_and_combine_chunks("api_data_aadhar_enrolment_*.csv")  # Adjust pattern!
    raw_bio = load_and_combine_chunks("api_data_aadhar_biometric_*.csv")
    raw_demo = load_and_combine_chunks("api_data_aadhar_demographic_*.csv")

    # 2. Clean Separately
    print("\n--- Phase 2: Cleaning Domains ---")
    clean_enrol = clean_domain_df(raw_enrol, 'enrol')
    clean_bio = clean_domain_df(raw_bio, 'bio')
    clean_demo = clean_domain_df(raw_demo, 'demo')

    # 3. Merge Together
    print("\n--- Phase 3: Merging to Master ---")
    # Start with Enrolment, merge Biometric
    if not clean_enrol.empty:
        master = clean_enrol
    else:
        master = clean_bio  # Fallback if enrolment missing

    if not clean_bio.empty:
        master = pd.merge(master, clean_bio, on=['state', 'district', 'month'], how='outer')

    if not clean_demo.empty:
        master = pd.merge(master, clean_demo, on=['state', 'district', 'month'], how='outer')

    # Fill NaNs with 0 (Crucial for Outer Joins)
    master = master.fillna(0)

    print(f"SUCCESS: Master Dataset Created with {len(master)} rows.")
    return master

# To run it:
# df = generate_master_dataset()