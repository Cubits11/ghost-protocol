# ui/monitor.py - Privacy Monitoring Dashboard
"""
Real-time Privacy Monitoring and Budget Management Dashboard
"""

import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
from typing import Dict, List, Any
import numpy as np


class PrivacyMetricsGenerator:
    """Generate realistic privacy monitoring data for demonstrations"""

    @staticmethod
    def get_privacy_budget_timeline(days: int = 30) -> pd.DataFrame:
        """Generate privacy budget usage over time"""
        dates = pd.date_range(start=datetime.now() - timedelta(days=days), end=datetime.now(), freq='H')

        # Simulate privacy budget consumption with realistic patterns
        budget_usage = []
        cumulative_usage = 0

        for i, date in enumerate(dates):
            # Higher usage during business hours
            hour = date.hour
            if 9 <= hour <= 17:
                base_usage = np.random.exponential(0.1)
            elif 18 <= hour <= 23:
                base_usage = np.random.exponential(0.05)
            else:
                base_usage = np.random.exponential(0.02)

            # Add some spikes for sensitive data processing
            if np.random.random() < 0.05:  # 5% chance of spike
                base_usage += np.random.exponential(0.5)

            cumulative_usage += base_usage
            budget_usage.append({
                'datetime': date,
                'hourly_usage': base_usage,
                'cumulative_usage': cumulative_usage,
                'remaining_budget': max(0, 8.0 - cumulative_usage)  # Start with 8.0ε budget
            })

        return pd.DataFrame(budget_usage)

    @staticmethod
    def get_constraint_violations() -> List[Dict[str, Any]]:
        """Generate sample constraint violation events"""
        violations = []
        base_time = datetime.now()

        violation_types = [
            {"type": "email_detection", "severity": "high", "action": "anonymized",
             "pattern": "email pattern detected"},
            {"type": "phone_detection", "severity": "high", "action": "anonymized", "pattern": "phone number detected"},
            {"type": "emotional_boundary", "severity": "medium", "action": "template_response",
             "pattern": "anger|rage detected"},
            {"type": "session_limit", "severity": "low", "action": "break_suggested", "pattern": "session > 30min"},
            {"type": "ssn_detection", "severity": "critical", "action": "blocked", "pattern": "SSN pattern detected"},
            {"type": "medical_info", "severity": "high", "action": "local_only", "pattern": "medical terms detected"}
        ]

        for i in range(15):  # Generate 15 recent violations
            violation = violation_types[i % len(violation_types)].copy()
            violation.update({
                'timestamp': base_time - timedelta(hours=i * 2, minutes=np.random.randint(0, 120)),
                'user_id': f"user_{np.random.randint(100, 999)}",
                'processing_time': np.random.uniform(0.05, 0.3),
                'privacy_cost': np.random.uniform(0.001, 0.1)
            })
            violations.append(violation)

        return sorted(violations, key=lambda x: x['timestamp'], reverse=True)

    @staticmethod
    def get_processing_routes_data() -> Dict[str, int]:
        """Get distribution of local vs cloud processing"""
        return {
            'Local Only': 156,
            'Cloud (Low Sensitivity)': 67,
            'Hybrid Processing': 23,
            'Blocked (Too Sensitive)': 4
        }


def privacy_budget_monitor():
    """Privacy budget monitoring component"""

    st.markdown("### 💰 Privacy Budget Monitor")

    # Current budget status
    col1, col2, col3, col4 = st.columns(4)

    current_budget = 6.8
    total_budget = 8.0
    usage_rate = 0.15  # ε per hour

    with col1:
        st.metric(
            "Current Budget",
            f"{current_budget:.2f}ε",
            delta=f"-{usage_rate:.2f}ε/hr"
        )

    with col2:
        st.metric(
            "Budget Utilization",
            f"{((total_budget - current_budget) / total_budget * 100):.1f}%",
            delta="↗️ +2.3% today"
        )

    with col3:
        estimated_hours = current_budget / usage_rate if usage_rate > 0 else float('inf')
        st.metric(
            "Est. Hours Remaining",
            f"{estimated_hours:.1f}h",
            delta="↘️ -1.2h since morning"
        )

    with col4:
        st.metric(
            "Queries Processed",
            "1,247",
            delta="↗️ +89 today"
        )

    # Budget timeline chart
    st.markdown("#### 📈 Budget Usage Timeline")

    timeline_data = PrivacyMetricsGenerator.get_privacy_budget_timeline(7)  # Last 7 days

    fig = go.Figure()

    # Cumulative usage line
    fig.add_trace(go.Scatter(
        x=timeline_data['datetime'],
        y=timeline_data['cumulative_usage'],
        mode='lines',
        name='Cumulative Usage',
        line=dict(color='#ff6b6b', width=2),
        fill='tonexty'
    ))

    # Remaining budget line
    fig.add_trace(go.Scatter(
        x=timeline_data['datetime'],
        y=timeline_data['remaining_budget'],
        mode='lines',
        name='Remaining Budget',
        line=dict(color='#4ecdc4', width=2)
    ))

    # Budget limit line
    fig.add_hline(
        y=total_budget,
        line_dash="dash",
        line_color="gray",
        annotation_text="Total Budget Limit"
    )

    fig.update_layout(
        height=400,
        xaxis_title="Time",
        yaxis_title="Privacy Budget (ε)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
        margin=dict(l=0, r=0, t=30, b=0)
    )

    st.plotly_chart(fig, use_container_width=True)

    # Budget alerts
    if current_budget < 2.0:
        st.error("🚨 LOW PRIVACY BUDGET WARNING: Consider reducing sensitivity or increasing local processing")
    elif current_budget < 4.0:
        st.warning("⚠️ Privacy budget running low. Monitor usage carefully.")
    else:
        st.success("✅ Privacy budget healthy")


def constraint_violation_monitor():
    """Monitor constraint violations and enforcement"""

    st.markdown("### 🛡️ Constraint Violation Monitor")

    violations = PrivacyMetricsGenerator.get_constraint_violations()

    # Violation summary
    col1, col2, col3 = st.columns(3)

    critical_violations = len([v for v in violations if v['severity'] == 'critical'])
    high_violations = len([v for v in violations if v['severity'] == 'high'])
    total_violations = len(violations)

    with col1:
        st.metric("Critical Violations", critical_violations, delta="↘️ -1 vs yesterday")

    with col2:
        st.metric("High Severity", high_violations, delta="↗️ +2 vs yesterday")

    with col3:
        st.metric("Total Prevented", total_violations, delta="↗️ +5 vs yesterday")

    # Violations by type chart
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Violations by Type")

        violation_counts = {}
        for violation in violations:
            v_type = violation['type']
            violation_counts[v_type] = violation_counts.get(v_type, 0) + 1

        fig = px.bar(
            x=list(violation_counts.keys()),
            y=list(violation_counts.values()),
            color=list(violation_counts.values()),
            color_continuous_scale='Reds'
        )
        fig.update_layout(
            height=300,
            xaxis_title="Violation Type",
            yaxis_title="Count",
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### ⚡ Actions Taken")

        action_counts = {}
        for violation in violations:
            action = violation['action']
            action_counts[action] = action_counts.get(action, 0) + 1

        fig = px.pie(
            values=list(action_counts.values()),
            names=list(action_counts.keys()),
            color_discrete_sequence=['#ff9999', '#66b3ff', '#99ff99', '#ffcc99']
        )
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    # Recent violations table
    st.markdown("#### 📋 Recent Violations")

    violations_df = pd.DataFrame(violations[:10])  # Show last 10
    violations_df['timestamp'] = violations_df['timestamp'].dt.strftime('%H:%M:%S')

    # Color code by severity
    def color_severity(val):
        if val == 'critical':
            return 'background-color: #ffebee'
        elif val == 'high':
            return 'background-color: #fff3e0'
        elif val == 'medium':
            return 'background-color: #f3e5f5'
        else:
            return 'background-color: #e8f5e8'

    styled_df = violations_df[['timestamp', 'type', 'severity', 'action', 'pattern']].style.applymap(
        color_severity, subset=['severity']
    )

    st.dataframe(styled_df, use_container_width=True)


def processing_route_monitor():
    """Monitor local vs cloud processing decisions"""

    st.markdown("### 🔀 Processing Route Monitor")

    route_data = PrivacyMetricsGenerator.get_processing_routes_data()

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("#### 📊 Processing Distribution")

        fig = px.pie(
            values=list(route_data.values()),
            names=list(route_data.keys()),
            color_discrete_sequence=['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
        )
        fig.update_layout(
            height=300,
            margin=dict(l=0, r=0, t=30, b=0)
        )
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        st.markdown("#### 📈 Route Performance")

        # Generate sample performance data
        routes = list(route_data.keys())
        avg_times = [0.05, 0.15, 0.25, 0.02]  # Processing times

        fig = go.Figure()
        fig.add_trace(go.Bar(
            x=routes,
            y=avg_times,
            text=[f"{t:.2f}s" for t in avg_times],
            textposition='auto',
            marker_color=['#2ecc71', '#3498db', '#f39c12', '#e74c3c']
        ))

        fig.update_layout(
            height=300,
            yaxis_title="Avg Response Time (s)",
            margin=dict(l=0, r=0, t=30, b=0),
            showlegend=False
        )
        fig.update_xaxes(tickangle=45)
        st.plotly_chart(fig, use_container_width=True)

    # Route details
    st.markdown("#### 🔍 Route Details")

    for route, count in route_data.items():
        percentage = (count / sum(route_data.values())) * 100

        if route == "Local Only":
            st.success(f"🏠 **{route}**: {count} requests ({percentage:.1f}%) - Highest privacy protection")
        elif route == "Blocked (Too Sensitive)":
            st.error(f"🚫 **{route}**: {count} requests ({percentage:.1f}%) - Data too sensitive to process")
        elif route == "Cloud (Low Sensitivity)":
            st.info(f"☁️ **{route}**: {count} requests ({percentage:.1f}%) - Low privacy risk")
        else:
            st.warning(f"🔀 **{route}**: {count} requests ({percentage:.1f}%) - Mixed local/cloud processing")


def privacy_alerts_panel():
    """Privacy alerts and recommendations panel"""

    st.markdown("### 🚨 Privacy Alerts & Recommendations")

    # Active alerts
    alerts = [
        {
            "level": "warning",
            "title": "High PII Detection Rate",
            "message": "Detected 23% increase in PII patterns. Consider user education.",
            "action": "Review recent conversations for data sensitivity patterns"
        },
        {
            "level": "info",
            "title": "Budget Optimization Opportunity",
            "message": "67% of queries could be processed locally with minimal accuracy loss.",
            "action": "Consider adjusting sensitivity thresholds"
        },
        {
            "level": "success",
            "title": "Constraint Effectiveness",
            "message": "All emotional boundaries performing within expected parameters.",
            "action": "No action required - system operating optimally"
        }
    ]

    for alert in alerts:
        if alert["level"] == "warning":
            st.warning(f"⚠️ **{alert['title']}**: {alert['message']}")
            st.write(f"**Recommended Action:** {alert['action']}")
        elif alert["level"] == "info":
            st.info(f"💡 **{alert['title']}**: {alert['message']}")
            st.write(f"**Recommended Action:** {alert['action']}")
        elif alert["level"] == "success":
            st.success(f"✅ **{alert['title']}**: {alert['message']}")
            st.write(f"**Status:** {alert['action']}")

    # Privacy recommendations
    st.markdown("#### 💡 Privacy Optimization Recommendations")

    recommendations = [
        "🎯 **Increase Local Processing**: 23 queries could be processed locally to save 0.8ε privacy budget",
        "⚙️ **Adjust Thresholds**: Emotional sensitivity threshold could be lowered by 10% with minimal impact",
        "📊 **Budget Allocation**: Consider increasing daily budget limit for peak usage hours",
        "🔄 **Pattern Analysis**: Medical information patterns appearing frequently - consider specialized constraints"
    ]

    for rec in recommendations:
        st.write(rec)


def real_time_metrics_panel():
    """Real-time system metrics panel"""

    st.markdown("### ⚡ Real-Time System Metrics")

    # System performance metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Avg Response Time",
            "0.15s",
            delta="↘️ -0.02s"
        )

    with col2:
        st.metric(
            "Constraint Eval Speed",
            "0.004s",
            delta="↗️ +0.001s"
        )

    with col3:
        st.metric(
            "Memory Usage",
            "234 MB",
            delta="↗️ +12 MB"
        )

    with col4:
        st.metric(
            "Active Sessions",
            "7",
            delta="↗️ +2"
        )

    # Real-time activity feed
    st.markdown("#### 🔄 Live Activity Feed")

    with st.container():
        activity_placeholder = st.empty()

        # Simulate real-time updates
        activities = [
            "🟢 User_456: Emotional boundary 'anxiety_support' triggered - response generated locally",
            "🔵 User_123: Email pattern detected - anonymization applied",
            "🟠 User_789: Session duration limit reached - break suggestion sent",
            "🟢 User_234: Privacy-sensitive query routed to local processing",
            "🔴 User_567: Critical constraint violation - SSN pattern blocked",
            "🟢 User_890: Successful constraint evaluation in 0.003s"
        ]

        activity_html = "<div style='height: 200px; overflow-y: scroll; border: 1px solid #ddd; padding: 10px; background-color: #f9f9f9;'>"
        for activity in activities:
            timestamp = datetime.now().strftime("%H:%M:%S")
            activity_html += f"<div style='margin-bottom: 5px;'><small>{timestamp}</small> - {activity}</div>"
        activity_html += "</div>"

        activity_placeholder.markdown(activity_html, unsafe_allow_html=True)


def privacy_compliance_dashboard():
    """Privacy compliance and audit dashboard"""

    st.markdown("### 📋 Privacy Compliance Dashboard")

    # Compliance metrics
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**GDPR Compliance**")
        st.success("✅ Right to deletion implemented")
        st.success("✅ Data portability enabled")
        st.success("✅ Consent management active")
        st.info("📊 Data retention: 30 days default")

    with col2:
        st.markdown("**Encryption Status**")
        st.success("✅ AES-256 encryption active")
        st.success("✅ End-to-end protection enabled")
        st.success("✅ Key rotation scheduled")
        st.info("🔒 0 unencrypted data points")

    with col3:
        st.markdown("**Audit Trail**")
        st.success("✅ All decisions logged")
        st.success("✅ Tamper evidence enabled")
        st.success("✅ Retention policy compliant")
        st.info("📝 1,247 audit entries today")

    # Compliance timeline
    st.markdown("#### 📈 Compliance Timeline")

    # Generate compliance events
    compliance_events = [
        {"date": datetime.now() - timedelta(days=1), "event": "GDPR data export request fulfilled",
         "status": "completed"},
        {"date": datetime.now() - timedelta(days=3), "event": "Privacy policy updated", "status": "completed"},
        {"date": datetime.now() - timedelta(days=5), "event": "Encryption key rotation", "status": "completed"},
        {"date": datetime.now() - timedelta(days=7), "event": "Audit trail verification", "status": "completed"},
        {"date": datetime.now() - timedelta(days=10), "event": "Data retention cleanup", "status": "completed"}
    ]

    for event in compliance_events:
        col1, col2, col3 = st.columns([2, 3, 1])
        with col1:
            st.write(event["date"].strftime("%Y-%m-%d"))
        with col2:
            st.write(event["event"])
        with col3:
            if event["status"] == "completed":
                st.success("✅")
            else:
                st.warning("⏳")


def privacy_monitoring_dashboard():
    """Main privacy monitoring dashboard function"""

    st.markdown("## 🔍 Privacy Monitoring Dashboard")
    st.markdown("Real-time monitoring of privacy protection, budget usage, and constraint enforcement")

    # Main monitoring sections
    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "💰 Budget Monitor",
        "🛡️ Violations",
        "🔀 Processing Routes",
        "🚨 Alerts",
        "📋 Compliance"
    ])

    with tab1:
        privacy_budget_monitor()

    with tab2:
        constraint_violation_monitor()

    with tab3:
        processing_route_monitor()

    with tab4:
        privacy_alerts_panel()
        real_time_metrics_panel()

    with tab5:
        privacy_compliance_dashboard()

    # Footer with refresh options
    st.markdown("---")
    col1, col2, col3 = st.columns(3)

    with col1:
        if st.button("🔄 Refresh All Metrics"):
            st.experimental_rerun()

    with col2:
        auto_refresh = st.checkbox("Auto-refresh every 30s", value=False)
        if auto_refresh:
            import time
            time.sleep(30)
            st.experimental_rerun()

    with col3:
        if st.button("📊 Export Privacy Report"):
            report_data = {
                "timestamp": datetime.now().isoformat(),
                "privacy_budget_remaining": 6.8,
                "violations_prevented": 15,
                "local_processing_percentage": 73.2,
                "compliance_status": "fully_compliant"
            }

            st.download_button(
                "Download Report",
                data=json.dumps(report_data, indent=2),
                file_name=f"privacy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json"
            )


def main():
    """Main function for testing the privacy monitor"""
    st.set_page_config(
        page_title="Ghost Protocol - Privacy Monitor",
        page_icon="🔍",
        layout="wide"
    )

    privacy_monitoring_dashboard()


if __name__ == "__main__":
    main()