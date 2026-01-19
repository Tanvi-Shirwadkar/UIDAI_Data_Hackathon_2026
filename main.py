from src.analyzer import (
    load_and_engineer_data,  # Core Engine: Loads data, runs EDA (Charts 1 & 2), generates features
    generate_health_index,  # Solution: ML Clustering (Chart 4)
    analyze_mbu_gap,  # Insight: Child Risk (Chart 5)
    analyze_fraud_spikes,  # Security: Fraud Sentinel (Chart 7)
    analyze_pincode_variance  # Insight: Intra-district Inequality (Chart 8)
)


# If you created src/time_forensics.py, uncomment the next line to get Chart 6
# from src.time_forensics import run_time_forensics

def main():
    print("===========================================")
    print("   UIDAI IDENTITY HEALTH ENGINE (MASTER)   ")
    print("===========================================\n")

    # ---------------------------------------------------------
    # STEP 1: LOAD & ENGINEER FEATURES
    # ---------------------------------------------------------
    # This automatically runs Independent EDA (Charts 1 & 2) inside the function
    try:
        df_pincode_features = load_and_engineer_data()

        if df_pincode_features.empty:
            print("CRITICAL ERROR: No data processed. Check your CSV files in 'data/'.")
            return

        print(f"SUCCESS: Feature Engineering Complete. Processed {len(df_pincode_features)} Pincode records.")
    except Exception as e:
        print(f"CRITICAL ERROR during loading: {e}")
        return

    # ---------------------------------------------------------
    # STEP 2: ANALYZE INEQUALITY (Chart 8)
    # ---------------------------------------------------------
    # Shows that even within a busy district, some pincodes do 0 work
    analyze_pincode_variance(df_pincode_features)
    print("[ANALYSIS] Chart 8: Pincode Variance Map generated.")

    # ---------------------------------------------------------
    # STEP 3: RUN THE ML SOLUTION (Chart 4)
    # ---------------------------------------------------------
    # Uses K-Means to cluster districts into Critical/Moderate/Healthy
    health_df = generate_health_index(df_pincode_features)
    health_df.to_csv("output/final_health_index.csv", index=False)

    print("[SOLUTION] Chart 4: Identity Health Index Generated via ML Clustering.")
    critical = health_df[health_df['risk_category'] == 'Critical Risk']
    print(f"ALERT: {len(critical)} Districts flagged as 'Critical Risk' (High Volume / Low Compliance).")

    if not critical.empty:
        print(">>> Top 3 Critical Districts to Audit:")
        print(critical[['state', 'district', 'health_index']].head(3).to_string(index=False))

    # ---------------------------------------------------------
    # STEP 4: ANALYZE CHILD RISK (Chart 5)
    # ---------------------------------------------------------
    # Identifies districts where children enroll but never update biometrics
    analyze_mbu_gap(df_pincode_features)
    print("[INSIGHT] Chart 5: Child 'Silent ID' Risk Map saved.")

    # ---------------------------------------------------------
    # STEP 5: RUN SECURITY SENTINEL (Chart 7)
    # ---------------------------------------------------------
    # Detects daily volume spikes > 3 Sigma (Z-Score)
    analyze_fraud_spikes()
    print("\n==========================================")
    print("   PIPELINE COMPLETE. ALL CHARTS SAVED.")
    print("   CHECK THE 'output/' FOLDER.")
    print("==========================================")


if __name__ == "__main__":
    main()
# from src.analyzer import load_and_clean_data, analyze_pareto, analyze_reliability, analyze_anomalies, \
#     analyze_weekend_effect
#
#
# def main():
#     print("==========================================")
#     print("   UIDAI DATATHON 2025 - ANALYTICS ENGINE ")
#     print("==========================================\n")
#
#     # 1. Load Data
#     try:
#         df = load_and_clean_data()
#         print(f"SUCCESS: Data Loaded. {df.shape[0]} clean rows ready for analysis.\n")
#     except Exception as e:
#         print(f"CRITICAL ERROR: {e}")
#         return
#
#     # 2. Run Differentiator 1: Inequality
#     pareto_data = analyze_pareto(df)
#     top_20_perc_vol = pareto_data[pareto_data['pincode_rank_perc'] <= 0.2]['total_transactions'].sum()
#     total_vol = pareto_data['total_transactions'].sum()
#     print(
#         f"[INSIGHT 1] Resource Inequality: The top 20% of centers handle {top_20_perc_vol / total_vol:.1%} of all traffic.")
#
#     # 3. Run Differentiator 2: Reliability
#     stats = analyze_reliability(df)
#     crisis_count = stats[stats['category'] == 'High Vol / Erratic (Crisis)'].shape[0]
#     print(f"[INSIGHT 2] Reliability Scan: {crisis_count} Districts are in 'Crisis Mode' (High Volume + Unstable).")
#
#     # 4. Run Differentiator 3: Anomalies
#     analyze_anomalies(df)
#     print(f"[INSIGHT 3] Sentinel System: Anomaly detection complete. Check 'output/3_anomaly_audit.png'.")
#
#     # 5. Run Differentiator 4: Weekend Effect
#     multiplier = analyze_weekend_effect(df)
#     print(f"[INSIGHT 4] Weekend Multiplier: Centers are {multiplier:.2f}x busier on weekends.")
#
#     print("\n==========================================")
#     print("   ANALYSIS COMPLETE. CHECK 'output/' FOLDER")
#     print("==========================================")
#
#
# if __name__ == "__main__":
#     main()