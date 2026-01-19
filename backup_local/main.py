# 1. Add analyze_fraud_spikes to the import list
from src.analyzer import load_and_engineer_data, generate_health_index, analyze_mbu_gap, analyze_fraud_spikes, analyze_pincode_variance


def main():
    print("--- UIDAI ANALYTICS: RIGOROUS PIPELINE ---")

    # ... (Your existing code for loading and index generation) ...
    df_pincode_features = load_and_engineer_data()
    analyze_pincode_variance(df_pincode_features)
    generate_health_index(df_pincode_features)
    analyze_mbu_gap(df_pincode_features)

    # 2. CALL THE SENTINEL HERE (At the end)
    analyze_fraud_spikes()

    print("\nProcessing Complete. All Charts saved.")


if __name__ == "__main__":
    main()
# from src.analyzer import load_and_clean_data, generate_health_index, analyze_mbu_gap
#
#
# def main():
#     print("===========================================")
#     print("   UIDAI IDENTITY HEALTH ENGINE (MASTER)   ")
#     print("===========================================\n")
#
#     # 1. Load the Merged Master Dataset
#     try:
#         df = load_and_clean_data()
#         print(f"SUCCESS: Master Dataset built. {df.shape[0]} District-Month records.")
#         print(f"Columns: {list(df.columns)}")
#     except Exception as e:
#         print(f"CRITICAL ERROR: {e}")
#         return
#
#     # 2. Run Solution: The Risk Index
#     # This uses Enrolment + Biometric + Demographic signals combined
#     health_df = generate_health_index(df)
#
#     print("\n[SOLUTION] Identity Health Index Generated.")
#     critical = health_df[health_df['risk_category'] == 'Critical Risk (High Gap)']
#     print(f"ALERT: {len(critical)} Districts flagged as Critical.")
#     print("Top 3 Critical Districts:")
#     print(critical[['state', 'district', 'health_index']].head(3))
#
#     # 3. Run Insight: Child MBU Gap
#     analyze_mbu_gap(df)
#     print("\n[INSIGHT] Child 'Silent ID' Risk map saved to output/5_child_risk_gap.png")
#
#     print("\nAnalysis Complete.")
#
#
# if __name__ == "__main__":
#     main()

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