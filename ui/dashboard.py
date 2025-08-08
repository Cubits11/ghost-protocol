# ui/dashboard.py - Ghost Protocol Streamlit Dashboard
"""
Ghost Protocol v0.1 - Professional Streamlit Interface
Demonstrates emotionally sovereign AI with constitutional constraints
"""

import streamlit as st
import asyncio
import json
import time
import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
from datetime import datetime, timedelta
import sys
import os

# Add parent directory to path for imports
parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(parent_dir)

try:
    from ghost_protocol import GhostProtocolSystem
    from core.constraints import EvaluationContext
    from core.vault import MemoryQuery
except ImportError:
    st.error("⚠️ Could not import Ghost Protocol components. Please check your installation.")
    st.stop()

# Page configuration
st.set_page_config(
    page_title="Ghost Protocol v0.1",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional styling
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f1f1f;
        text-align: center;
        margin-bottom: 0.5rem;
    }
    .sub-header {
        font-size: 1.2rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .constraint-applied {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
    .privacy-warning {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
    .success-message {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        border-radius: 5px;
        padding: 0.5rem;
        margin: 0.25rem 0;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def initialize_ghost_protocol():
    """Initialize Ghost Protocol system with caching"""
    try:
        ghost = GhostProtocolSystem()
        return ghost
    except Exception as e:
        st.error(f"Failed to initialize Ghost Protocol: {e}")
        return None


def display_system_status(ghost):
    """Display real-time system status in sidebar"""
    if not ghost:
        st.sidebar.error("System not initialized")
        return

    try:
        status = ghost.get_system_status()

        st.sidebar.markdown("### 🔍 System Status")

        # Privacy Budget
        privacy = status.get('privacy_budget', {})
        remaining_epsilon = privacy.get('remaining_epsilon', 0)
        utilization = privacy.get('utilization_percent', 0)

        # Color coding for privacy budget
        if utilization < 50:
            budget_color = "🟢"
        elif utilization < 80:
            budget_color = "🟡"
        else:
            budget_color = "🔴"

        st.sidebar.metric(
            f"{budget_color} Privacy Budget",
            f"{remaining_epsilon:.1f}ε",
            f"{utilization:.1f}% used"
        )

        # System Health
        health = status.get('system_health', 'unknown')
        health_emoji = "✅" if health == "operational" else "⚠️"

        st.sidebar.metric(
            f"{health_emoji} System Health",
            health.title(),
            "All systems operational" if health == "operational" else "Check logs"
        )

        # Constraints Loaded
        constraints = status.get('constraints', {})
        total_constraints = constraints.get('total_loaded', 0)

        st.sidebar.metric(
            "⚖️ Constraints Active",
            total_constraints,
            "Constitutional rules loaded"
        )

        # Active Sessions
        active_sessions = status.get('active_sessions', 0)
        st.sidebar.metric(
            "👥 Active Sessions",
            active_sessions,
            "Current user sessions"
        )

        # Refresh button
        if st.sidebar.button("🔄 Refresh Status"):
            st.cache_resource.clear()
            st.rerun()

    except Exception as e:
        st.sidebar.error(f"Status update failed: {e}")


def chat_interface():
    """Main chat interface for Ghost Protocol"""
    st.markdown('<div class="main-header">💬 Ghost Protocol Chat</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Experience emotionally sovereign AI with constitutional constraints</div>',
                unsafe_allow_html=True)

    # Initialize ghost protocol
    ghost = initialize_ghost_protocol()
    if not ghost:
        st.error("Cannot initialize Ghost Protocol system")
        return

    # User input section
    col1, col2 = st.columns([3, 1])

    with col1:
        user_input = st.text_area(
            "Enter your message:",
            placeholder="Try: 'I'm feeling really angry about work' or 'My email is john@example.com'",
            height=100,
            key="user_input"
        )

    with col2:
        user_id = st.text_input("User ID:", value="demo_user", key="user_id")

        # Predefined demo scenarios
        st.markdown("**🎭 Demo Scenarios:**")
        demo_scenarios = {
            "😡 Anger Trigger": "I am absolutely furious about my work situation!",
            "🔒 Privacy Test": "My email is jane.doe@example.com and my SSN is 123-45-6789",
            "⏰ Session Limit": "I've been chatting for over an hour now",
            "😰 High Emotion": "I'm extremely overwhelmed and anxious about everything!"
        }

        for scenario_name, scenario_text in demo_scenarios.items():
            if st.button(scenario_name, key=f"scenario_{scenario_name}"):
                st.session_state.user_input = scenario_text
                st.rerun()

    # Process message
    if user_input and st.button("🚀 Send Message", type="primary"):
        with st.spinner("🧠 Processing with constitutional constraints..."):
            start_time = time.time()

            try:
                # Process with Ghost Protocol
                result = asyncio.run(ghost.process_user_input(user_input, user_id))
                processing_time = (time.time() - start_time) * 1000

                # Display results
                display_chat_result(result, processing_time)

            except Exception as e:
                st.error(f"Processing failed: {e}")
                st.info("Check system logs for details")


def display_chat_result(result, processing_time):
    """Display the chat processing result with detailed breakdown"""

    # Main response
    st.markdown("### 🤖 AI Response")
    response_container = st.container()
    with response_container:
        st.markdown(f'<div class="success-message"><strong>{result.response}</strong></div>',
                    unsafe_allow_html=True)

    # Processing details in columns
    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("#### 🛤️ Processing Route")
        route_color = {
            "local_only": "🟢",
            "cloud_anonymized": "🟡",
            "blocked": "🔴",
            "error": "⚫"
        }.get(result.processing_route, "⚪")

        st.write(f"{route_color} **{result.processing_route.replace('_', ' ').title()}**")

        # Route explanation
        route_explanations = {
            "local_only": "Processed locally for privacy",
            "cloud_anonymized": "Cloud processed with anonymization",
            "blocked": "Blocked by safety constraints",
            "error": "Processing error occurred"
        }
        st.caption(route_explanations.get(result.processing_route, "Unknown route"))

    with col2:
        st.markdown("#### ⚡ Performance")
        st.metric("Processing Time", f"{processing_time:.1f}ms")
        st.metric("Confidence Score", f"{result.confidence_score:.2f}")

        # Performance indicator
        if processing_time < 200:
            st.success("🚀 Excellent performance")
        elif processing_time < 500:
            st.info("✅ Good performance")
        else:
            st.warning("⚠️ Consider optimization")

    with col3:
        st.markdown("#### 🔒 Privacy Impact")
        budget_used = result.privacy_budget_used

        if budget_used > 0:
            st.metric("Privacy Budget Used", f"{budget_used:.3f}ε")
            st.warning("🔒 Privacy budget consumed")
        else:
            st.metric("Privacy Budget Used", "0.000ε")
            st.success("🛡️ No privacy cost")

    # Constraints Applied
    if result.constraints_applied:
        st.markdown("### ⚖️ Constitutional Constraints Applied")
        for constraint in result.constraints_applied:
            st.markdown(
                f'<div class="constraint-applied">🛡️ <strong>{constraint}</strong> - Constitutional boundary enforced</div>',
                unsafe_allow_html=True)

    # System Status
    if hasattr(result, 'system_status') and result.system_status:
        with st.expander("🔧 Detailed System Status"):
            st.json(result.system_status)

    # Memory Storage Status
    if result.emotional_context_stored:
        st.success("💾 Emotional context securely stored with encryption")
    else:
        st.warning("⚠️ Context not stored (privacy protection or error)")


def constraint_editor():
    """Constitutional constraint editor interface"""
    st.markdown('<div class="main-header">⚖️ Constitutional Constraints Editor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Define your emotional boundaries and privacy rules</div>',
                unsafe_allow_html=True)

    ghost = initialize_ghost_protocol()
    if not ghost:
        st.error("Cannot initialize Ghost Protocol system")
        return

    # Constraint templates
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### 📝 Constraint Definition (YAML)")

        # Default constraints for editing
        default_constraints = """constraints:
  emotional_boundaries:
    - name: "anger_support"
      pattern: "angry|furious|rage|mad|pissed"
      response: "acknowledge_emotion"
      severity: "high"
      template: "I can sense you're feeling frustrated. I'm here to help you work through this constructively."

    - name: "anxiety_guidance"
      pattern: "anxious|worried|stressed|overwhelmed|panic"
      response: "acknowledge_emotion"
      severity: "medium"
      template: "It sounds like you're feeling anxious. Let's take this step by step."

  privacy_rules:
    - name: "pii_protection"
      condition: "contains_pii == true"
      action: "anonymize"
      severity: "high"

    - name: "high_privacy_local_only"
      condition: "privacy_score > 0.7"
      action: "block"
      severity: "high"

  interaction_limits:
    - name: "session_duration_limit"
      condition: "session_duration > 45"
      action: "suggest_break"
      severity: "medium"
      threshold: 45

    - name: "emotional_intensity_limit"
      condition: "emotional_intensity > 0.85"
      action: "suggest_break"
      severity: "high"
      threshold: 0.85"""

        # YAML editor
        constraints_yaml = st.text_area(
            "Constitutional Constraints (YAML):",
            value=default_constraints,
            height=400,
            help="Define your emotional boundaries, privacy rules, and interaction limits"
        )

        # Validation and loading
        col_validate, col_load = st.columns(2)

        with col_validate:
            if st.button("🔍 Validate Constraints"):
                try:
                    is_valid, errors = ghost.parser.validate_constraint_syntax(constraints_yaml)
                    if is_valid:
                        st.success("✅ Constraints are valid!")
                    else:
                        st.error("❌ Validation failed:")
                        for error in errors:
                            st.write(f"• {error}")
                except Exception as e:
                    st.error(f"Validation error: {e}")

        with col_load:
            if st.button("📥 Load Constraints"):
                try:
                    success = ghost.load_user_constraints(constraints_yaml)
                    if success:
                        st.success("✅ Constraints loaded successfully!")
                        st.cache_resource.clear()  # Refresh system status
                    else:
                        st.error("❌ Failed to load constraints")
                except Exception as e:
                    st.error(f"Loading error: {e}")

    with col2:
        st.markdown("#### 📚 Constraint Guide")

        with st.expander("🎭 Emotional Boundaries", expanded=True):
            st.markdown("""
            **Purpose:** Define how the AI responds to emotional content

            **Fields:**
            - `name`: Unique identifier
            - `pattern`: Regex pattern to match
            - `response`: Action type (acknowledge_emotion, redirect)
            - `severity`: low, medium, high
            - `template`: Custom response message
            """)

        with st.expander("🔒 Privacy Rules"):
            st.markdown("""
            **Purpose:** Protect sensitive information

            **Conditions:**
            - `contains_pii == true`: Personal information detected
            - `privacy_score > X`: Privacy sensitivity threshold
            - `contains_financial_info == true`: Financial data

            **Actions:**
            - `anonymize`: Remove sensitive data
            - `block`: Prevent processing
            """)

        with st.expander("⏰ Interaction Limits"):
            st.markdown("""
            **Purpose:** Prevent unhealthy usage patterns

            **Conditions:**
            - `session_duration > X`: Minutes in session
            - `emotional_intensity > X`: Emotion level (0-1)
            - `daily_interactions > X`: Daily usage count

            **Actions:**
            - `suggest_break`: Recommend pause
            """)

        # Current constraints summary
        st.markdown("#### 📊 Current Constraints")
        try:
            summary = ghost.parser.export_constraints_summary()

            st.metric("Total Constraints", summary.get('total_constraints', 0))

            # Breakdown by type
            for constraint_type, count in summary.get('by_type', {}).items():
                st.write(f"• {constraint_type.replace('_', ' ').title()}: {count}")

        except Exception as e:
            st.error(f"Could not load constraint summary: {e}")


def privacy_monitor():
    """Privacy monitoring and violation tracking interface"""
    st.markdown('<div class="main-header">📊 Privacy & Violation Monitor</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Real-time monitoring of privacy budget and constraint violations</div>',
                unsafe_allow_html=True)

    ghost = initialize_ghost_protocol()
    if not ghost:
        st.error("Cannot initialize Ghost Protocol system")
        return

    # Privacy Budget Overview
    col1, col2 = st.columns([2, 1])

    with col1:
        st.markdown("#### 💰 Privacy Budget Status")

        try:
            status = ghost.get_system_status()
            privacy = status.get('privacy_budget', {})

            # Privacy budget gauge
            remaining = privacy.get('remaining_epsilon', 8.0)
            total = privacy.get('total_epsilon', 8.0)
            used = total - remaining
            utilization = (used / total) * 100

            # Create gauge chart
            fig_gauge = go.Figure(go.Indicator(
                mode="gauge+number+delta",
                value=utilization,
                domain={'x': [0, 1], 'y': [0, 1]},
                title={'text': "Privacy Budget Utilization (%)"},
                delta={'reference': 0},
                gauge={
                    'axis': {'range': [None, 100]},
                    'bar': {'color': "darkblue"},
                    'steps': [
                        {'range': [0, 50], 'color': "lightgray"},
                        {'range': [50, 80], 'color': "yellow"},
                        {'range': [80, 100], 'color': "red"}
                    ],
                    'threshold': {
                        'line': {'color': "red", 'width': 4},
                        'thickness': 0.75,
                        'value': 90
                    }
                }
            ))
            fig_gauge.update_layout(height=300)
            st.plotly_chart(fig_gauge, use_container_width=True)

        except Exception as e:
            st.error(f"Could not load privacy status: {e}")

    with col2:
        st.markdown("#### 📈 Privacy Metrics")

        try:
            # Mock data for demonstration - in real implementation would come from system
            st.metric("Epsilon Remaining", f"{remaining:.2f}ε", f"-{used:.2f}ε")
            st.metric("Operations Today", privacy.get('operations_today', 0))

            # Privacy status indicator
            if utilization < 50:
                st.success("🟢 Privacy budget healthy")
            elif utilization < 80:
                st.warning("🟡 Monitor privacy usage")
            else:
                st.error("🔴 Privacy budget critical")

        except Exception as e:
            st.error(f"Metrics error: {e}")

    # Constraint Violations Timeline
    st.markdown("#### 🚨 Constraint Violations Timeline")

    try:
        # Get violation history
        violations = ghost.constraint_enforcer.get_violation_history()

        if violations:
            # Create violations dataframe
            violation_data = []
            for violation in violations[-20:]:  # Last 20 violations
                violation_data.append({
                    'timestamp': violation.timestamp,
                    'constraint': violation.constraint_name,
                    'type': violation.constraint_type.value,
                    'severity': violation.severity,
                    'reason': violation.violation_reason[:50] + "..." if len(
                        violation.violation_reason) > 50 else violation.violation_reason
                })

            if violation_data:
                df_violations = pd.DataFrame(violation_data)

                # Violations by type chart
                violations_by_type = df_violations['type'].value_counts()
                fig_violations = px.bar(
                    x=violations_by_type.index,
                    y=violations_by_type.values,
                    title="Violations by Constraint Type",
                    labels={'x': 'Constraint Type', 'y': 'Count'}
                )
                st.plotly_chart(fig_violations, use_container_width=True)

                # Recent violations table
                st.markdown("#### 📋 Recent Violations")
                st.dataframe(
                    df_violations[['timestamp', 'constraint', 'type', 'severity', 'reason']],
                    use_container_width=True
                )
            else:
                st.info("No violations recorded yet")
        else:
            st.info("No violations recorded yet")

    except Exception as e:
        st.error(f"Could not load violation history: {e}")

    # Enforcement Statistics
    with st.expander("📊 Detailed Enforcement Statistics"):
        try:
            stats = ghost.constraint_enforcer.get_enforcement_statistics()
            st.json(stats)
        except Exception as e:
            st.error(f"Could not load enforcement statistics: {e}")


def main():
    """Main application entry point"""

    # Header
    st.markdown('<div class="main-header">👻 Ghost Protocol v0.1</div>', unsafe_allow_html=True)
    st.markdown('<div class="sub-header">Emotionally Sovereign AI with Constitutional Constraints</div>',
                unsafe_allow_html=True)

    # Initialize and display system status in sidebar
    ghost = initialize_ghost_protocol()
    display_system_status(ghost)

    # Main application tabs
    tab1, tab2, tab3 = st.tabs(["💬 Chat Demo", "⚖️ Constraints Editor", "📊 Privacy Monitor"])

    with tab1:
        chat_interface()

    with tab2:
        constraint_editor()

    with tab3:
        privacy_monitor()

    # Footer
    st.markdown("---")
    st.markdown(
        '<div style="text-align: center; color: #666; font-size: 0.9rem;">'
        'Ghost Protocol v0.1 - Built for Emotional Sovereignty | '
        '<a href="https://github.com/yourusername/ghost_protocol_v01" target="_blank">GitHub</a>'
        '</div>',
        unsafe_allow_html=True
    )


if __name__ == "__main__":
    main()