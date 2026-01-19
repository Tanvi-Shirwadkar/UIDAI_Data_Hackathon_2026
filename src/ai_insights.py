
import os
from dotenv import load_dotenv
from groq import Groq


class AIInsightsGenerator:
    """Generate business insights using Groq API"""
    
    def __init__(self):
        load_dotenv()
        api_key = os.getenv('api_key')
        
        if not api_key or api_key == 'your_groq_api_key_here':
            raise ValueError(
                "⚠️ Groq API key not found! Please add your API key to the .env file.\n"
                "Get your key from: https://console.groq.com/keys"
            )
        
        self.client = Groq(api_key=api_key)
    
    def get_graph_insights(self):
    
        return {
            "chart_1_enrolment_trends": {
                "title": "Enrolment Temporal Consistency",
                "key_findings": [
                    "Child enrolments show strong seasonality with sharp peaks around August-September",
                    "Adult enrolments remain consistently low with only minor month-to-month variation",
                    "Child enrolments are driven by academic admission cycles and school calendars",
                    "Adult enrolments are need-based and less dependent on fixed timelines"
                ],
                "implications": [
                    "Seasonal dependence makes child enrolment vulnerable to short-term disruptions",
                    "Adult enrolments provide a stable baseline demand across the year",
                    "Capacity planning must account for 4x spike during school admission periods"
                ]
            },
            
            "chart_2_biometric_splits": {
                "title": "Biometric Update Distribution (Child vs Adult)",
                "key_findings": [
                    "Nearly 50-50 split between child and adult biometric updates",
                    "Children's biometrics change rapidly and require periodic updates",
                    "Adults require far fewer updates over time"
                ],
                "implications": [
                    "A near-50 split is NOT healthy parity - it signals systemic under-updating of child biometrics",
                    "Silent accumulation of outdated child records",
                    "Increased risk of future authentication failures for children",
                    "Gaps in awareness, access, or operational prioritization"
                ]
            },
            
            "chart_4_health_clusters": {
                "title": "Identity Health Clusters (Machine Learning)",
                "key_findings": [
                    "Districts classified into Critical Risk, Moderate, and Healthy clusters",
                    "High Aadhaar volume does NOT guarantee system health",
                    "Compliance and stability matter more than transaction volume",
                    "Large districts below trend line are failing despite scale"
                ],
                "implications": [
                    "Some large districts process massive volumes but fail to maintain biometric freshness",
                    "Large failing districts are highest-value targets for audits",
                    "Infrastructure alone doesn't ensure operational success",
                    "Need compliance checks and fraud monitoring for high-volume low-health districts"
                ]
            },
            
            "chart_5_child_risk_zones": {
                "title": "Child Identity Risk Zones (Enrolment-Update Gap)",
                "key_findings": [
                    "Large gaps between child enrolments and biometric updates in specific districts",
                    "Children enrolled in Aadhaar but fail to complete mandatory biometric updates",
                    "Top risk districts: Pune, Delhi, West Champaran, Bangalore Urban"
                ],
                "implications": [
                    "Aadhaar enrolment without updates creates silent identity failure",
                    "Service denial in PDS, DBT, and scholarships",
                    "Increased risk of identity misuse or proxy authentication",
                    "System overestimates true identity coverage",
                    "High migration, urban churn, and geographic barriers contribute to gaps"
                ]
            },
            
            "chart_7_anomaly_detection": {
                "title": "Automated Fraud Detection (Security Sentinel)",
                "key_findings": [
                    "Z-score based algorithm flags abnormal daily transaction spikes (>3σ)",
                    "Massive spike on 01 March flagged as >3σ anomaly",
                    "Low false-positive rate - routine fluctuations stay within ±2σ"
                ],
                "implications": [
                    "True anomalies isolated without noise",
                    "Enables proactive fraud response and rapid audits",
                    "Damage containment before systemic impact",
                    "Explainable, statistically sound, and scalable approach"
                ]
            },
            
            "chart_8_pincode_variance": {
                "title": "Intra-District Inequality (Pincode Workload Distribution)",
                "key_findings": [
                    "High Coefficient of Variation (CV = 0.96) indicates severe intra-district imbalance",
                    "Small number of pincodes handle disproportionately high workload",
                    "Urban and high-density pincodes attract more transactions"
                ],
                "implications": [
                    "Districts may appear functional overall but hide severe service overload at specific pincodes",
                    "Uneven workload creates service pressure in high-demand areas",
                    "Underutilized capacity persists in peripheral areas",
                    "Need for pincode-level staff and resource reallocation"
                ]
            }
        }
    
    def generate_business_insights(self):
        """
        Generate comprehensive business insights for UIDAI
        using all graph interpretations as context
        """
        
        graph_insights = self.get_graph_insights()
        
        # Create comprehensive prompt for Gemini
        prompt = f"""
You are a senior data consultant providing strategic insights to UIDAI (Unique Identification Authority of India) leadership tp help them Improve the Aadhaar identity system and support informed decision-making and system improvements.

Based on the analysis of 6 different aspects of the Aadhaar identity system, generate a comprehensive business report in PLAIN ENGLISH that can be understood by non-technical UIDAI officials and policy makers.

Here are the detailed findings from each analysis:

1. {graph_insights['chart_1_enrolment_trends']['title']}
Key Findings: {', '.join(graph_insights['chart_1_enrolment_trends']['key_findings'])}
Implications: {', '.join(graph_insights['chart_1_enrolment_trends']['implications'])}

2. {graph_insights['chart_2_biometric_splits']['title']}
Key Findings: {', '.join(graph_insights['chart_2_biometric_splits']['key_findings'])}
Implications: {', '.join(graph_insights['chart_2_biometric_splits']['implications'])}

3. {graph_insights['chart_4_health_clusters']['title']}
Key Findings: {', '.join(graph_insights['chart_4_health_clusters']['key_findings'])}
Implications: {', '.join(graph_insights['chart_4_health_clusters']['implications'])}

4. {graph_insights['chart_5_child_risk_zones']['title']}
Key Findings: {', '.join(graph_insights['chart_5_child_risk_zones']['key_findings'])}
Implications: {', '.join(graph_insights['chart_5_child_risk_zones']['implications'])}

5. {graph_insights['chart_7_anomaly_detection']['title']}
Key Findings: {', '.join(graph_insights['chart_7_anomaly_detection']['key_findings'])}
Implications: {', '.join(graph_insights['chart_7_anomaly_detection']['implications'])}

6. {graph_insights['chart_8_pincode_variance']['title']}
Key Findings: {', '.join(graph_insights['chart_8_pincode_variance']['key_findings'])}
Implications: {', '.join(graph_insights['chart_8_pincode_variance']['implications'])}

INSTRUCTIONS:
1. BE CONCISE - Quality over quantity, maximum 800 words total
2. REASONING MANDATORY - Explain WHY behind every recommendation
3. Use simple, layman terms - no jargon
4. Back every point with specific data from the graphs
5. Focus on TOP 3 most critical issues only
6. Each recommendation must include: WHAT to do, WHY it matters, EXPECTED outcome( in brief)
7. bcoz our goal is to : Identify meaningful patterns, trends, anomalies, or predictive indicators and translate them into clear insights or solution frameworks that can support informed decision-making and system improvements.
I want you to give helpful, actionable, and data-driven insights and not stuff like add more enrolment centres for adhar etc.

Generate a CONCISE report with these sections:


TOP 3 CRITICAL ISSUES (Priority ranked)

**Issue #1: [Name] **
• Evidence: [Chart + specific finding]
• Why it matters: [Business impact]
• Root cause: [Why this happens]

**Issue #2: [Name] **
• Evidence: [Chart + specific finding]
• Why it matters: [Business impact]
• Root cause: [Why this happens]

**Issue #3: [Name] **
• Evidence: [Chart + specific finding]
• Why it matters: [Business impact]
• Root cause: [Why this happens]

TOP 3 RECOMMENDED ACTIONS


**Action 1: [Specific action]**
• Why: [Clear reasoning - what problem this solves]
• Expected outcome: [Quantified impact]


**Action 2: [Specific action]**
• Why: [Clear reasoning - what problem this solves]
• Expected outcome: [Quantified impact]


**Action 3: [Specific action]**
• Why: [Clear reasoning - what problem this solves]
• Expected outcome: [Quantified impact]


Make it compelling, data-driven, and actionable for UIDAI leadership.
"""
        
        try:
            response = self.client.chat.completions.create(
                model="llama-3.3-70b-versatile",
                messages=[
                    {"role": "system", "content": "You are a senior data consultant for UIDAI. Be concise, focus on quality over quantity. Every recommendation must have clear reasoning."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                max_tokens=4000
            )
            return response.choices[0].message.content
        
        except Exception as e:
            return f"""
**Error Generating AI Insights**

There was an issue connecting to Groq API: {str(e)}



"""


def get_ai_insights():
    """
    Main function to get AI insights
    Returns the generated insights text or error message
    """
    try:
        generator = AIInsightsGenerator()
        insights = generator.generate_business_insights()
        return insights, None
    except ValueError as e:
        return None, str(e)
    except Exception as e:
        return None, f"Unexpected error: {str(e)}"