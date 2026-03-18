"""
Streamlit interface for SupportMailAgent.
Run with: streamlit run streamlit_app.py
"""

import streamlit as st
import requests
import json
from datetime import datetime

# Page config
st.set_page_config(
    page_title="Support Mail Agent",
    page_icon="📧",
    layout="wide",
    initial_sidebar_state="expanded"
)

# API configuration
API_BASE_URL = st.secrets.get("API_BASE_URL", "http://localhost:8000")
API_ENDPOINT = f"{API_BASE_URL}/emails/process"

# Custom CSS
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        margin-bottom: 0.5rem;
    }
    .success-box {
        background-color: #d4edda;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #28a745;
    }
    .escalation-box {
        background-color: #fff3cd;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #ffc107;
    }
    .error-box {
        background-color: #f8d7da;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #dc3545;
    }
    .intent-badge {
        display: inline-block;
        padding: 0.4rem 0.8rem;
        border-radius: 0.3rem;
        font-weight: bold;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)

# Title
st.markdown('<div class="main-header">📧 Support Mail Agent</div>', unsafe_allow_html=True)
st.markdown("AI-powered email classification, response generation, and escalation management")

# Sidebar
with st.sidebar:
    st.header("⚙️ Settings")
    api_url = st.text_input(
        "API Base URL",
        value=API_BASE_URL,
        help="URL where FastAPI server is running"
    )
    if api_url != API_BASE_URL:
        API_BASE_URL = api_url
        API_ENDPOINT = f"{API_BASE_URL}/emails/process"

    st.markdown("---")
    st.markdown("**How it works:**")
    st.markdown("""
    1. **Classify** - AI detects email intent (billing, technical, etc.)
    2. **Retrieve** - Searches knowledge base for relevant docs
    3. **Respond** - Generates personalized response
    4. **Escalate** - Routes complex issues to humans
    5. **Follow-up** - Schedules follow-ups for escalated tickets
    """)

# Tabs
tab1, tab2 = st.tabs(["Process Email", "History"])

# ============ TAB 1: Process Email ============
with tab1:
    st.header("Process Support Email")

    col1, col2 = st.columns([1, 1])

    with col1:
        sender = st.text_input(
            "Sender Email",
            placeholder="customer@example.com",
            help="Email address of the customer"
        )

    with col2:
        subject = st.text_input(
            "Subject",
            placeholder="e.g., Payment issue, Technical support",
            help="Email subject line"
        )

    body = st.text_area(
        "Email Body",
        placeholder="Paste the customer's email content here...",
        height=200,
        help="Full email message content"
    )

    col1, col2, col3 = st.columns([1, 1, 2])

    with col1:
        process_button = st.button("🚀 Process Email", type="primary", use_container_width=True)

    with col2:
        clear_button = st.button("🔄 Clear", use_container_width=True)

    if clear_button:
        st.session_state.clear()
        st.rerun()

    # Process email when button is clicked
    if process_button:
        if not sender or not subject or not body:
            st.error("❌ Please fill in all fields (sender, subject, body)")
        elif "@" not in sender:
            st.error("❌ Please enter a valid email address")
        else:
            with st.spinner("⏳ Processing email through AI workflow..."):
                try:
                    # Prepare payload
                    payload = {
                        "sender": sender,
                        "subject": subject,
                        "body": body
                    }

                    # Call API
                    response = requests.post(API_ENDPOINT, json=payload, timeout=60)

                    if response.status_code == 200:
                        result = response.json()

                        # Get detailed results
                        email_id = result.get("email_id")
                        details_url = f"{API_BASE_URL}/emails/details/{email_id}"
                        details_response = requests.get(details_url, timeout=10)

                        # Store in session for history
                        if "processed_emails" not in st.session_state:
                            st.session_state.processed_emails = []

                        if details_response.status_code == 200:
                            details = details_response.json()
                            st.session_state.processed_emails.append(details)
                            st.session_state.current_result = details
                        else:
                            st.session_state.current_result = {
                                "email": result,
                                "workflow": {
                                    "intent": "unknown",
                                    "confidence": 0,
                                    "kb_results": [],
                                    "should_escalate": result.get("escalated", False)
                                },
                                "timestamp": datetime.now().isoformat()
                            }

                        st.success("✅ Email processed successfully!")
                    else:
                        st.error(f"❌ API Error: {response.status_code} - {response.text}")

                except requests.exceptions.ConnectionError:
                    st.error(f"❌ Cannot connect to API at {API_BASE_URL}. Make sure FastAPI server is running on port 8000")
                except requests.exceptions.Timeout:
                    st.error("❌ Request timeout. The API took too long to respond.")
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")

    # Display results
    if "current_result" in st.session_state:
        result = st.session_state.current_result

        st.markdown("---")
        st.header("📊 Processing Results")

        # Email info
        st.subheader("Original Email")
        col1, col2 = st.columns([1, 1])
        with col1:
            st.text(f"**From:** {result['input']['sender']}")
        with col2:
            st.text(f"**ID:** {result['email']['email_id']}")
        st.text(f"**Subject:** {result['input']['subject']}")

        # Workflow results in columns
        col1, col2, col3 = st.columns([1, 1, 1])

        with col1:
            intent = result['workflow'].get('intent', 'unknown').upper()
            confidence = result['workflow'].get('confidence', 0)
            st.metric("Intent Detected", intent)
            st.metric("Confidence", f"{confidence:.1%}")

        with col2:
            escalated = result['workflow'].get('should_escalate', False)
            status = "🔴 ESCALATED" if escalated else "🟢 HANDLED"
            st.metric("Status", status)
            followup = result['workflow'].get('followup_scheduled', False)
            st.metric("Follow-up", "✅ Yes" if followup else "❌ No")

        with col3:
            kb_count = len(result['workflow'].get('kb_results', []))
            st.metric("KB Results", kb_count)

        # Knowledge Base Results
        if result['workflow'].get('kb_results'):
            st.subheader("📚 Knowledge Base Retrieved")
            for i, kb_result in enumerate(result['workflow']['kb_results'][:3], 1):
                with st.expander(f"Result {i}"):
                    st.write(kb_result if isinstance(kb_result, str) else json.dumps(kb_result, indent=2))

        # Generated Response
        st.subheader("💬 AI-Generated Response")
        if result['workflow'].get('should_escalate'):
            st.markdown('<div class="escalation-box">', unsafe_allow_html=True)
            st.warning("⚠️ This issue has been escalated to human support")
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="success-box">', unsafe_allow_html=True)
            st.success("✅ Response generated by AI")
            st.markdown('</div>', unsafe_allow_html=True)

        st.text_area(
            "Response",
            value=result['email']['body'],
            height=150,
            disabled=True
        )

        # Metadata
        with st.expander("📋 Raw Details"):
            st.json(result)

# ============ TAB 2: History ============
with tab2:
    st.header("📜 Processing History")

    if "processed_emails" in st.session_state and st.session_state.processed_emails:
        emails = st.session_state.processed_emails

        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            st.metric("Total Processed", len(emails))
        with col2:
            escalated = sum(1 for e in emails if e['workflow'].get('should_escalate'))
            st.metric("Escalated", escalated)
        with col3:
            st.metric("Success Rate", f"{((len(emails) - escalated) / len(emails) * 100):.0f}%")

        st.markdown("---")

        # Display email list
        for i, email in enumerate(reversed(emails), 1):
            col1, col2, col3, col4 = st.columns([2, 1, 1, 1])

            with col1:
                st.text(f"**{email['input']['subject'][:50]}**")
                st.caption(f"From: {email['input']['sender']}")

            with col2:
                intent = email['workflow'].get('intent', 'unknown').upper()
                st.text(f"*{intent}*")

            with col3:
                escalated = "🔴" if email['workflow'].get('should_escalate') else "🟢"
                st.text(escalated)

            with col4:
                confidence = email['workflow'].get('confidence', 0)
                st.text(f"{confidence:.0%}")

            st.divider()
    else:
        st.info("No emails processed yet. Process an email to see history here.")

# Footer
st.markdown("---")
st.markdown(
    """
    <div style="text-align: center; color: gray; font-size: 0.9rem;">
    Built with ❤️ using Streamlit, FastAPI, and Claude AI
    </div>
    """,
    unsafe_allow_html=True
)
