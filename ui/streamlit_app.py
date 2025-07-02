import streamlit as st
import time
import pandas as pd
import sys
import os
from dotenv import load_dotenv
import logging
from workflow.workflow import app as workflow_app
from utils.logger import get_logger
import re
from utils.status_tracker import StatusTracker
from langsmith import traceable

# --- Setup ---
logging.basicConfig(
    filename="app.log",
    level=logging.INFO,
    format="%(asctime)s %(levelname)s:%(name)s:%(message)s"
)
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
logger = get_logger("streamlit_app")
logger.info("Streamlit app started")
st.set_page_config(page_title="GenAI Social Support", layout="wide", page_icon="🤖")
load_dotenv() 
os.environ["LANGCHAIN_TRACING_V2"] = "true"

api_key = os.environ.get("LANGSMITH_API_KEY")

st.markdown("""
<style>
/* Body styling - Light mode */
body { 
    color: #333333; 
    background-color: #f8f9fa;
}

/* Dark mode overrides */
@media (prefers-color-scheme: dark) {
    body {
        color: #e0e0e0 !important;
        background-color: #0e1117 !important;
    }
    .stForm {
        background-color: #1e1e1e !important;
        border: 1px solid #333;
    }
    .stTextInput input, 
    .stNumberInput input, 
    .stTextArea textarea {
        background-color: #2d2d2d !important;
        color: #e0e0e0 !important;
        border: 1px solid #444 !important;
    }
    .stTextInput label, 
    .stNumberInput label, 
    .stTextArea label, 
    .stFileUploader label, 
    .stSelectbox label {
        color: #e0e0e0 !important;
    }
    .dataframe {
        background-color: #1e1e1e !important;
        color: #e0e0e0 !important;
    }
    .mismatch {
        background-color: #5c0000 !important;
    }
}

/* Form container */
.stForm {
    border-radius: 10px;
    padding: 2rem;
    background: #ffffff;
    color: #333333;
    border: 1px solid #e0e0e0;
    box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    margin-bottom: 2rem;
}

/* Input fields */
.stTextInput input, 
.stNumberInput input, 
.stTextArea textarea {
    background-color: #f8f9fa !important;
    color: #222 !important;
    border: 1px solid #ced4da !important;
}

/* Labels */
.stTextInput label, 
.stNumberInput label, 
.stTextArea label, 
.stFileUploader label, 
.stSelectbox label {
    color: #333333 !important;
    font-weight: 500 !important;
}

/* Button styling */
.stButton > button {
    background-color: #007bff !important;  /* Bright blue */
    color: #ffffff !important;             /* White text */
    border-radius: 5px !important;
    padding: 0.75rem 1.5rem !important;
    font-weight: bold !important;
    font-size: 1.1rem !important;
    border: none !important;
    width: 100% !important;
    margin-top: 1.5rem !important;
    box-shadow: 0 2px 8px rgba(0, 123, 255, 0.3) !important;
    transition: background-color 0.3s ease, transform 0.2s ease !important;
}

.stButton > button:hover {
    background-color: #0056b3 !important;  /* Darker blue on hover */
    transform: scale(1.03) !important;
}

.stButton > button:disabled {
    background-color: #6c757d !important;
    color: #ced4da !important;
    transform: none !important;
    cursor: not-allowed !important;
}

/* Headings */
h1, h2, h3, h4, h5, h6 {
    color: inherit !important;
}

/* Form sections */
.stSubheader {
    color: inherit !important;
    border-bottom: 2px solid #4CAF50;
    padding-bottom: 0.5rem;
    margin-top: 1.5rem !important;
    margin-bottom: 1rem !important;
}

/* Progress bar color */
.stProgress > div > div { 
    background-color: #4CAF50 !important; 
}

/* DataFrame appearance */
.dataframe { 
    font-size: 14px; 
    background-color: white;
}

/* Mismatch class background */
.mismatch { 
    background-color: #ffcccc !important; 
}

/* Sidebar styles */
[data-testid="stSidebar"] {
    background-color: #ffffff !important;
    color: #222 !important;
}

[data-testid="stSidebar"] .stHeader, 
[data-testid="stSidebar"] .stInfo, 
[data-testid="stSidebar"] .stSubheader, 
[data-testid="stSidebar"] .stCaption {
    color: #222 !important;
}

/* Dark mode sidebar */
@media (prefers-color-scheme: dark) {
    [data-testid="stSidebar"] {
        background-color: #1e1e1e !important;
        color: #e0e0e0 !important;
    }
    [data-testid="stSidebar"] .stHeader, 
    [data-testid="stSidebar"] .stInfo, 
    [data-testid="stSidebar"] .stSubheader, 
    [data-testid="stSidebar"] .stCaption {
        color: #e0e0e0 !important;
    }
}

/* Chat message styling */
.stChatMessage {
    border-radius: 10px !important;
    padding: 1rem !important;
    margin-bottom: 1rem !important;
    box-shadow: 0 2px 5px rgba(0,0,0,0.1) !important;
}

/* Footer styling */
footer {
    color: #6c757d !important;
    font-size: 0.9rem !important;
    text-align: center !important;
    padding: 1rem 0 !important;
}
</style>
""", unsafe_allow_html=True)

# Initialize DB
from db.database import init_db, insert_application
init_db()
logger.info("Database initialized.")

# --- Session State ---
if 'chat_history' not in st.session_state:
    st.session_state.chat_history = []
if 'form_data' not in st.session_state:
    st.session_state.form_data = None
if 'processing' not in st.session_state:
    st.session_state.processing = False
if 'show_details' not in st.session_state:
    st.session_state.show_details = False
if 'final_state' not in st.session_state:
    st.session_state.final_state = {}

# Add to your session state initialization
if 'current_status' not in st.session_state:
    st.session_state.current_status = "Ready for submission"

# --- Header ---
st.title("📋 UAE Social Support Application")
st.caption("Powered by AI-assisted document processing")

# --- Helper Functions ---
def format_phone(phone):
    """Format phone number for display"""
    if not phone:
        return ""
    cleaned = ''.join(filter(str.isdigit, phone))
    if len(cleaned) == 9:
        return f"0{cleaned[:2]} {cleaned[2:5]} {cleaned[5:]}"
    elif len(cleaned) == 10:
        return f"0{cleaned[:2]} {cleaned[2:6]} {cleaned[6:]}"
    return phone

def format_currency(value):
    """Format currency values"""
    return f"AED {float(value):,.2f}" if value else "N/A"

def normalize_address(addr):
    if not addr:
        return ""
    # Remove punctuation, lower case, and normalize whitespace
    addr = re.sub(r'[^\w\s]', '', addr)
    addr = re.sub(r'\s+', ' ', addr)
    return addr.strip().lower()

# --- Form ---
with st.form("application_form", clear_on_submit=False):
    st.subheader("👤 Applicant Information")
    
    # Personal details in 2 columns
    col1, col2 = st.columns([1, 1])
    with col1:
        # emirates_id = st.text_input("Emirates ID*", help="Enter your 15-digit Emirates ID")
        # name = st.text_input("Full Name*")
        emirates_id = st.text_input("Emirates ID*", help="Enter 15-digit ID. Add '911' to trigger bank/credit validation failures (demo)")
        name = st.text_input(
            "Full Name*", 
            help="Enter your name. Include 'DEMO' to trigger govt validation failure"
        )
        phone = st.text_input("Phone Number*", placeholder="05X-XXX-XXXX")
        
    with col2:
        # address = st.text_area("Address*", height=100)
        address = st.text_area(
        "Address*", 
        height=100,
        help="Enter your address. Include 'DEMO' to trigger govt validation failure"
        )
        dependents = st.number_input("Number of Dependents*", min_value=0, step=1, value=0)
    
    st.divider()
    st.subheader("💰 Financial Information")
    
    # Financial details in 2 columns
    col3, col4 = st.columns([1, 1])
    with col3:
        income = st.number_input("Monthly Income (AED)*", min_value=0.0, format="%.2f", value=0.0)
    with col4:
        loans = st.number_input("Total Loan Amount (AED)*", min_value=0.0, format="%.2f", value=0.0)
    
    st.divider()
    st.subheader("📄 Document Upload")
    
    # Document upload in 2 columns
    col5, col6 = st.columns([1, 1])
    with col5:
        emirates_id_file = st.file_uploader("Emirates ID (PDF)", type=["pdf"], help="Upload scanned Emirates ID")
    with col6:
        bank_statement_file = st.file_uploader("Bank Statement (Excel)", type=["xlsx", "xls"], help="Upload bank statement in Excel format")
    
   
    st.divider()
    st.subheader("🎓 Career Development (Optional)")
    resume_file = st.file_uploader("Upload Resume (PDF)", type=["pdf"], 
                              help="Upload your resume to receive career recommendations")
    
    # Submit button
    submitted = st.form_submit_button("Submit Application", 
                                      disabled=st.session_state.processing,
                                      help="Review all fields before submission")

# --- Form Processing ---
if submitted and not st.session_state.processing:
    required_fields = [emirates_id, name, phone, address]
    if not all(required_fields):
        st.error("Please fill all required fields marked with *")
        st.stop()
    
    st.session_state.processing = True
    st.session_state.form_data = {
        "emirates_id": emirates_id,
        "name": name,
        "phone": phone,
        "address": address,
        "dependents": dependents,
        "income": income,
        "loans": loans,
        "emirates_id_file": emirates_id_file,
        "bank_statement_file": bank_statement_file
    }
    
    # Force UI update before heavy processing
    st.rerun()

if st.session_state.processing:
    with st.status("🔍 Processing your application...", expanded=True) as status:
        try:
            def process_application(initial_state):
                return workflow_app.invoke(initial_state)
            
            # Initialize state with cached form data
            initial_state = {
                **st.session_state.form_data,
                "resume_file": resume_file,
                "extracted_emirates_id": "",
                "extracted_name": "",
                "extracted_address": "",
                "extracted_phone": "",
                "extracted_income": 0.0,
                "extracted_loans": 0.0,
                "mismatches": [],
                "ollama_response": ""
            }
            
            # Update status for document extraction
            st.session_state.current_status = "📄 Extracting documents"
            st.write(st.session_state.current_status)
            time.sleep(0.5)  # Simulate processing time
            
            final_state = process_application(initial_state)
            # Run the workflow
            st.session_state.current_status = "🔍 Validating information"
            st.write(st.session_state.current_status)
            final_state = workflow_app.invoke(initial_state)
            st.session_state.final_state = final_state
            
            # Store in database
            insert_application(
                final_state['extracted_emirates_id'], 
                name, phone, address, dependents,
                income, loans,
                final_state['extracted_income'],
                final_state['extracted_loans']
            )
            
            # Update to complete status
            st.session_state.current_status = "✅ Processing complete"
            st.write(st.session_state.current_status)
            status.update(label="Processing complete!", state="complete", expanded=False)

            # Show validation results
            validation_results = final_state.get('validation_results', {})
            if validation_results:
                validation_status = "PASSED ✅" if validation_results.get('all_valid') else "FAILED ❌"
                st.session_state.chat_history.append(
                    ("🤖 Validation Agent", 
                     f"Validation Status: {validation_status}")
                )                
                # Add detailed validation messages
                for val_type in ['bank_validation', 'credit_validation', 'govt_validation']:
                    if val_type in validation_results:
                        result = validation_results[val_type]
                        status_icon = "✅" if result.get('valid') else "❌"
                        st.session_state.chat_history.append(
                            ("🤖 Validation Agent", 
                             f"{status_icon} {val_type.replace('_', ' ').title()}: {result.get('message')}")
                        )

            if final_state.get('validation_result'):
                ml_result = final_state['validation_result']
                decision_msg = (
                    "🧠 AI Eligibility Decision: " 
                    f"{'✅ Eligible' if ml_result['eligible'] else '❌ Not Eligible'} "
                    f"(Confidence: {ml_result['confidence']*100:.1f}%)"
                )
                st.session_state.chat_history.append(("🤖 AI Validator", decision_msg))
                
                # Detailed factors if available
                if ml_result.get('decision_factors'):
                    factors_msg = "Key Decision Factors:\n"
                    if ml_result['decision_factors'].get('top_positive'):
                        factors_msg += "➕ Supporting Factors:\n"
                        for factor, impact in ml_result['decision_factors']['top_positive'].items():
                            factors_msg += f"- {factor.replace('_', ' ').title()}\n"
                    
                    if ml_result['decision_factors'].get('top_negative'):
                        factors_msg += "➖ Limiting Factors:\n"
                        for factor, impact in ml_result['decision_factors']['top_negative'].items():
                            factors_msg += f"- {factor.replace('_', ' ').title()}\n"
                    
                    st.session_state.chat_history.append(("🤖 AI Validator", factors_msg))

            # Show discrepancies if any
            if final_state['mismatches']:
                # Reconciliation warning
                st.session_state.chat_history.append(
                    ("🤖 Reconciliation Agent", 
                     f"⚠️ Found discrepancies in: {', '.join(final_state['mismatches'])}")
                )
                # LLM explanation for failure
                llm_msg = (
                    "❌ Data reconciliation failed. "
                    "Your input for the following field(s) does not match the supporting documents: "
                    f"{', '.join(final_state['mismatches'])}. "
                    "Please review and correct your application or upload the correct documents."
                )
                st.session_state.chat_history.append(
                    ("🤖 Ollama", llm_msg)
                )
                st.session_state.show_details = True
            else:
                # Always show a success LLM message if not present
                llm_msg = final_state.get('ollama_response') or "✅ Data reconciliation completed successfully. Now moving to validation."
                st.session_state.chat_history.append(
                    ("🤖 Ollama", llm_msg)
                )
                st.balloons()


            if final_state.get('recommendations', {}).get('status') == 'success':
                st.session_state.chat_history.append(
                    ("🤖 Career Advisor", "Based on your resume, here are some career development suggestions:")
                )
                # If recommendations are a list of strings
                if isinstance(final_state['recommendations'].get('recommendations'), list):
                    for rec in final_state['recommendations']['recommendations']:
                        st.session_state.chat_history.append(
                            ("💼 Recommendation", rec)
                        )
                # If recommendations are in a single string with newlines
                elif isinstance(final_state['recommendations'].get('recommendations'), str):
                    for rec in final_state['recommendations']['recommendations'].split('\n'):
                        if rec.strip():  # Skip empty lines
                            st.session_state.chat_history.append(
                                ("💼 Recommendation", rec.strip())
                            )


        except RuntimeError as e:
            error_msg = str(e)
            st.session_state.current_status = "❌ Processing error"
            if "Financial evaluation error" in error_msg:
                st.error("Financial evaluation failed. Please try again.")
            else:
                st.error(f"Processing error: {error_msg}")
            logger.error(f"Processing failed: {error_msg}")
            st.session_state.chat_history.append(
                ("🤖 System", f"Error: {error_msg}")
            )
                
        except Exception as e:
            StatusTracker.set_status("❌ Processing error")
            logger.error(f"Processing failed: {str(e)}")
            st.error(f"Processing error: {str(e)}")
            st.session_state.chat_history.append(
                ("🤖 System", f"Error processing application: {str(e)}")
            )
        finally:
            st.session_state.processing = False
            st.rerun()  # Refresh UI

# --- Reconciliation Details ---
if st.session_state.get('show_details', False) and st.session_state.form_data and st.session_state.final_state:
    st.subheader("📊 Document Reconciliation")
    
    final_state = st.session_state.final_state  # for convenience

    recon_data = {
        "Field": ["Emirates ID", "Name", "Phone", "Address", 
                  "Monthly Income", "Loan Amount"],
        "Form Value": [
            st.session_state.form_data['emirates_id'],
            st.session_state.form_data['name'],
            format_phone(st.session_state.form_data['phone']),
            st.session_state.form_data['address'],
            format_currency(st.session_state.form_data['income']),
            format_currency(st.session_state.form_data['loans'])
        ],
        "Document Value": [
            final_state.get('extracted_emirates_id', ""),
            final_state.get('extracted_name', ""),
            format_phone(final_state.get('extracted_phone', "")),
            final_state.get('extracted_address', ""),
            format_currency(final_state.get('extracted_income', "")),
            format_currency(final_state.get('extracted_loans', ""))
        ],
        "Status": [
            "Match" if st.session_state.form_data['emirates_id'] == final_state.get('extracted_emirates_id', "") else "Mismatch",
            "Match" if st.session_state.form_data['name'].strip().lower() == final_state.get('extracted_name', "").strip().lower() else "Mismatch",
            "Match" if st.session_state.form_data['phone'].replace(" ", "") == (final_state.get('extracted_phone', "") or "").replace(" ", "") else "Mismatch",
            "Match" if normalize_address(st.session_state.form_data['address']) == normalize_address(final_state.get('extracted_address', "")) else "Mismatch",
            "Match" if abs(st.session_state.form_data['income'] - float(final_state.get('extracted_income', 0.0))) <= 500 else "Mismatch",
            "Match" if abs(st.session_state.form_data['loans'] - float(final_state.get('extracted_loans', 0.0))) <= 500 else "Mismatch"
        ]
    }
    
    recon_df = pd.DataFrame(recon_data)
    
    def highlight_status(row):
        styles = [''] * len(row)
        if row['Status'] == 'Mismatch':
            styles = ['background-color: #ffcccc'] * len(row)
        return styles
    
    st.dataframe(
        recon_df.style.apply(highlight_status, axis=1),
        hide_index=True,
        use_container_width=True
    )
    
    # Resubmit button
    if st.button("🔄 Update and Resubmit", key="resubmit"):
        st.session_state.show_details = False
        st.session_state.processing = False
        st.session_state.form_data = None
        st.rerun()

# --- Chat History ---
if st.session_state.chat_history:
    st.divider()
    st.subheader("📜 Application History")
    
    for sender, message in reversed(st.session_state.chat_history):
        with st.chat_message(sender.split()[0].lower()):
            st.write(message)

# --- Sidebar ---
with st.sidebar:
    st.header("ℹ️ Application Guidance")
    st.info("""
        **Required Documents:**
        1. Emirates ID (PDF scan)
        2. Bank statement (Excel format)
        
        **Tips for Success:**
        - Ensure all fields are filled
        - Upload clear document scans
        - Verify extracted details match your input
        """)
    
    st.divider()
    st.subheader("⏱️ Application Status")
    # Define status icons and colors
    status_config = {
        "Ready for submission": {"icon": "📝", "color": "blue"},
        "📄 Extracting documents": {"icon": "📄", "color": "orange"},
        "🔍 Validating information": {"icon": "🔍", "color": "orange"},
        "🤖 Running AI validation": {"icon": "🤖", "color": "orange"},
        "✅ Processing complete": {"icon": "✅", "color": "green"},
        "Error occurred": {"icon": "❌", "color": "red"}
    }
    
    current_status = StatusTracker.get_status() or "Ready for submission"
    current_status = StatusTracker.get_status() or "Ready for submission"
    status_info = {
        "Ready for submission": {"icon": "📝", "color": "blue", "text": "Fill out the form and submit your application"},
        "📄 Extracting documents": {"icon": "📄", "color": "orange", "text": "Extracting data from uploaded documents"},
        "🔍 Reconciling data": {"icon": "🔍", "color": "orange", "text": "Comparing form data with documents"},
        "🔍 Validating information": {"icon": "🔒", "color": "orange", "text": "Validating against government databases"},
        "🤖 Running AI evaluation": {"icon": "🤖", "color": "orange", "text": "Assessing financial eligibility"},
        "💼 Generating career recommendations": {"icon": "💼", "color": "purple", "text": "Creating personalized career suggestions"},
        "✅ Processing complete": {"icon": "✅", "color": "green", "text": "Your application has been processed"},
        "❌ Processing error": {"icon": "❌", "color": "red", "text": "An error occurred during processing"}
    }.get(current_status, {"icon": "ℹ️", "color": "blue", "text": current_status})
    
    st.markdown(f"""
        <div style="
            background-color: var(--{status_info['color']}-50);
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid var(--{status_info['color']}-500);
            margin-bottom: 1rem;
        ">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.5rem;">{status_info['icon']}</span>
                <span style="font-weight: bold;">{current_status}</span>
            </div>
            <p style="margin: 0.5rem 0 0; color: var(--text-color);">{status_info['text']}</p>
        </div>
    """, unsafe_allow_html=True)
    
def get_status_details(status):
    """Return additional details for each status"""
    details = {
        "Ready for submission": "Fill out the form and submit your application",
        "📄 Extracting documents": "Extracting data from uploaded documents",
        "🔍 Validating information": "Validating against government databases",
        "🤖 Running AI validation": "Evaluating financial eligibility with AI",
        "✅ Processing complete": "Your application has been processed",
        "Error occurred": "An error occurred during processing"
    }
    return f'<p style="margin: 0.5rem 0 0; color: var(--text-color);">{details.get(status, "")}</p>'

    st.markdown(f"""
        <div style="
            background-color: var(--{status_info['color']}-50);
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid var(--{status_info['color']}-500);
            margin-bottom: 1rem;
        ">
            <div style="display: flex; align-items: center; gap: 0.5rem;">
                <span style="font-size: 1.5rem;">{status_info['icon']}</span>
                <span style="font-weight: bold;">{current_status}</span>
            </div>
            {get_status_details(current_status)}
        </div>
    """, unsafe_allow_html=True)
    
    st.divider()
    st.caption("Need help? Contact support@uaesocialsupport.ae")

# --- Footer ---
st.divider()
st.caption("© 2023 UAE Social Support Program | All applications are subject to verification")