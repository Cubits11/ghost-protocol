# ui/demo.py - Ghost Protocol Streamlit Demo
"""
Ghost Protocol v0.1 - Interactive Demo
Professional interface for demonstrating emotionally sovereign AI
"""

import streamlit as st
import asyncio
import json
import sys
import os
from datetime import datetime

# Add parent directory to path for imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ghost_protocol_main import GhostProtocolSystem
from core.constraints import EXAMPLE_CONSTRAINTS

# Page configuration
st.set_page_config(
    page_title="👻 Ghost Protocol v0.1",
    page_icon="👻",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for professional appearance
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: 700;
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
    .metric-container {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 1rem;
        border-radius: 10px;
        color: white;
        margin: 0.5rem 0;
    }
    .constraint-badge {
        background: #ff6b6b;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
        margin: 0.2rem;
        display: inline-block;
    }
    .success-badge {
        background: #51cf66;
        color: white;
        padding: 0.2rem 0.5rem;
        border-radius: 5px;
        font-size: 0.8rem;
    }
    .processing-route {
        font-weight: bold;
        color: #495057;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_ghost_protocol():
    """Load and cache the Ghost Protocol system"""
    return GhostProtocolSystem()


def display_system_status(ghost):
    """Display system status in sidebar"""
    status = ghost.get_system_status()

    st.sidebar.markdown("## 🔧 System Status")

    # System health
    health_color = "🟢" if status.get("system_health") == "healthy" else "🔴"
    st.sidebar.markdown(f"**Health:** {health_color} {status.get('system_health', 'Unknown').title()}")

    # Privacy budget
    budget = status.get("privacy_budget", {})
    remaining = budget.get("remaining_epsilon", 0)
    total = budget.get("total_epsilon", 8.0)
    utilization = budget.get("utilization_percent", 0)

    st.sidebar.markdown(f"**Privacy Budget:** {remaining:.1f}ε / {total}ε")
    st.sidebar.progress(utilization / 100)

    # Constraints
    constraints = status.get("constraints", {})
    st.sidebar.markdown(f"**Constraints:** {constraints.get('enabled', 0)} active")

    # Memory vault
    memory = status.get("memory_vault", {})
    st.sidebar.markdown(f"**Memory:** {memory.get('total_contexts', 0)} contexts stored")


def display_example_prompts():
    """Display example prompts for testing"""
    st.sidebar.markdown("## 💡 Try These Examples")

    examples = {
        "😡 Emotional Boundary": "I am so angry and frustrated about this situation!",
        "🔒 Privacy Protection": "My email is john.doe@example.com, can you help me?",
        "😰 Anxiety Support": "I'm really anxious and worried about my presentation tomorrow",
        "💬 Normal Chat": "Hello! How are you doing today?",
        "📱 PII Detection": "My phone number is 555-123-4567 and I need assistance"
    }

    for label, prompt in examples.items():
        if st.sidebar.button(label, key=f"example_{label}"):
            st.session_state.user_input = prompt
            st.rerun()


def format_processing_result(result):
    """Format processing result for display"""

    # Main response
    st.markdown("### 🤖 AI Response")
    st.success(result.response)

    # Create columns for metrics
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        route_emoji = {"blocked": "🚫", "local_only": "💻", "cloud_anonymized": "☁️", "local_default": "🔒"}
        emoji = route_emoji.get(result.processing_route, "❓")
        st.metric("Processing Route", f"{emoji} {result.processing_route}")

    with col2:
        st.metric("Privacy Budget Used", f"{result.privacy_budget_used}ε")

    with col3:
        st.metric("Processing Time", f"{result.processing_time_ms:.1f}ms")

    with col4:
        confidence_color = "🟢" if result.confidence_score > 0.8 else "🟡" if result.confidence_score > 0.5 else "🔴"
        st.metric("Confidence", f"{confidence_color} {result.confidence_score:.2f}")

    # Constraints applied
    if result.constraints_applied:
        st.markdown("### ⚖️ Constitutional Constraints Applied")
        constraint_html = ""
        for constraint in result.constraints_applied:
            constraint_html += f'<span class="constraint-badge">{constraint}</span> '
        st.markdown(constraint_html, unsafe_allow_html=True)
    else:
        st.markdown("### ✅ No Constraint Violations")
        st.markdown('<span class="success-badge">All constraints satisfied</span>', unsafe_allow_html=True)

    # Technical details in expander
    with st.expander("🔍 Technical Details"):
        col1, col2 = st.columns(2)

        with col1:
            st.json({
                "interaction_id": result.interaction_id,
                "emotional_context_stored": result.emotional_context_stored,
                "processing_route": result.processing_route,
                "constraints_applied": result.constraints_applied
            })

        with col2:
            st.json({
                "system_status": result.system_status
            })


def main():
    """Main Streamlit application"""

    # Header
    st.markdown('<h1 class="main-header">👻 Ghost Protocol v0.1</h1>', unsafe_allow_html=True)
    st.markdown('<p class="sub-header">Emotionally Sovereign AI with Constitutional Constraints</p>',
                unsafe_allow_html=True)

    # Load Ghost Protocol system
    try:
        ghost = load_ghost_protocol()

        if not ghost.system_initialized:
            st.error("❌ Ghost Protocol system failed to initialize!")
            st.stop()

    except Exception as e:
        st.error(f"❌ Failed to load Ghost Protocol: {e}")
        st.stop()

    # Sidebar
    display_system_status(ghost)
    display_example_prompts()

    # Add constraint viewer in sidebar
    with st.sidebar.expander("📋 View Active Constraints"):
        st.code(EXAMPLE_CONSTRAINTS, language="yaml")

    # Main interface
    st.markdown("## 💬 Interactive Demo")
    st.markdown("Experience emotionally sovereign AI that respects your boundaries and privacy.")

    # Chat interface
    user_input = st.text_area(
        "Your message:",
        height=100,
        placeholder="Try: 'I'm feeling really anxious about my presentation tomorrow...'",
        key="user_input_area",
        value=st.session_state.get("user_input", "")
    )

    # User ID input
    col1, col2 = st.columns([3, 1])
    with col1:
        user_id = st.text_input("User ID (optional):", value="demo_user", key="user_id_input")
    with col2:
        st.markdown("<br>", unsafe_allow_html=True)  # Spacing
        process_button = st.button("🚀 Process Message", type="primary", use_container_width=True)

    # Process message
    if process_button and user_input.strip():
        with st.spinner("🧠 Processing with Ghost Protocol..."):
            try:
                # Run async function
                result = asyncio.run(ghost.process_user_input(user_input, user_id))

                # Display results
                format_processing_result(result)

                # Clear input for next message
                if "user_input" in st.session_state:
                    del st.session_state.user_input

            except Exception as e:
                st.error(f"❌ Processing failed: {e}")

    elif process_button and not user_input.strip():
        st.warning("⚠️ Please enter a message to process.")

    # Information sections
    st.markdown("---")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### 🔒 Privacy First")
        st.markdown("""
        - **Encrypted Memory:** All emotional data stored with military-grade encryption
        - **Local Processing:** Sensitive content processed locally only  
        - **Privacy Budget:** Differential privacy prevents data leakage
        - **No Surveillance:** User-controlled emotional boundaries
        """)

    with col2:
        st.markdown("### ⚖️ Constitutional AI")
        st.markdown("""
        - **Emotional Boundaries:** Detects and respects emotional states
        - **Privacy Rules:** Automatic PII detection and protection
        - **Interaction Limits:** Prevents harmful long-term usage
        - **Behavioral Guidelines:** Ensures respectful communication
        """)

    with col3:
        st.markdown("### 🚀 Technical Innovation")
        st.markdown("""
        - **<5ms Constraint Evaluation:** Real-time boundary enforcement
        - **Hybrid Processing:** Smart local/cloud routing decisions
        - **Cryptographic Enforcement:** Technically guaranteed constraints
        - **Memory Decay:** Automatic cleanup of old emotional data
        """)

    # Footer
    st.markdown("---")
    st.markdown("""
    <div style="text-align: center; color: #666; padding: 1rem;">
        <strong>Ghost Protocol v0.1</strong> - The first technically enforceable framework for emotionally sovereign AI<br>
        Built with constitutional constraints, encrypted memory, and privacy-preserving technology
    </div>
    """, unsafe_allow_html=True)


if __name__ == "__main__":
    main()