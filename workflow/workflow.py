from callbacks.logging_callback import LoggingCallbackHandler
from langgraph.graph import StateGraph, END
from typing import TypedDict, List, Optional, Any
from agents.document_loader_agent import load_documents_and_extract_fields
from agents.reconciliation_agent import reconcile_fields
from agents.validation_agent import run_all_validations
from utils.utils import ollama_financial_assistance_response
from utils.logger import get_logger
from utils.xgboost_validator import validator
from utils.status_tracker import StatusTracker
from langsmith import traceable

logger = get_logger("workflow")

class ApplicationState(TypedDict):
    emirates_id: str
    name: str
    phone: str
    address: str
    dependents: int
    income: float
    loans: float
    emirates_id_file: Optional[Any]
    bank_statement_file: Optional[Any]
    extracted_emirates_id: str
    extracted_name: str
    extracted_address: str
    extracted_phone: str
    extracted_income: float
    extracted_loans: float
    mismatches: List[str]
    ollama_response: str
    validation_results: dict
    resume_file: Optional[Any]
    recommendations: Optional[dict[str, Any]]


def log_state_change(node_name: str, state: ApplicationState):
    """Log state changes with sensitive data redaction"""
    safe_state = {
        **state,
        "emirates_id": f"{state['emirates_id'][:3]}..." if state['emirates_id'] else "",
        "extracted_emirates_id": f"{state['extracted_emirates_id'][:3]}..." if state['extracted_emirates_id'] else "",
        "phone": f"{state['phone'][:3]}..." if state['phone'] else "",
        "extracted_phone": f"{state['extracted_phone'][:3]}..." if state.get('extracted_phone') else "",
        "name": f"{state['name'][:1]}***" if state['name'] else "",
        "extracted_name": f"{state['extracted_name'][:1]}***" if state.get('extracted_name') else "",
        "address": f"{state['address'][:10]}..." if state['address'] else "",
        "extracted_address": f"{state['extracted_address'][:10]}..." if state.get('extracted_address') else "",
        "emirates_id_file": "[FILE]" if state['emirates_id_file'] else "None",
        "bank_statement_file": "[FILE]" if state['bank_statement_file'] else "None"
    }
    logger.info(f"STATE CHANGE AFTER {node_name}: {safe_state}")

@traceable(name="Extract Documents", tags=["agent"], metadata={"type": "agent"})
def extract_documents_node(state: ApplicationState) -> ApplicationState:
    logger.info("Starting document extraction node")
    logger.info(f"Status update: Extracting documents")
    StatusTracker.set_status("📄 Extracting documents")
    try:
        doc_result = load_documents_and_extract_fields(
            state['emirates_id_file'],
            state['emirates_id'],
            state['bank_statement_file']
        )
        new_state = {
            **state,
            'extracted_emirates_id': doc_result['extracted_emirates_id'],
            'extracted_name': doc_result['extracted_name'],
            'extracted_address': doc_result['extracted_address'],
            'extracted_phone': doc_result['extracted_phone'],
            'extracted_income': doc_result['extracted_income'],
            'extracted_loans': doc_result['extracted_loans']
        }
        logger.info(
            f"Document extraction completed. Extracted Emirates ID: {new_state['extracted_emirates_id'][:6]}..., "
            f"Name: {new_state['extracted_name'][:10] if new_state['extracted_name'] else ''}..., "
            f"Address: {new_state['extracted_address'][:10] if new_state['extracted_address'] else ''}..., "
            f"Phone: {new_state['extracted_phone'][:6] if new_state['extracted_phone'] else ''}..., "
            f"Income: {new_state['extracted_income']}, Loans: {new_state['extracted_loans']}"
        )
        log_state_change("EXTRACT_DOCUMENTS", new_state)
        return new_state
    except Exception as e:
        logger.error(f"Document extraction failed: {str(e)}")
        raise

@traceable(name="Reconcile Data", tags=["agent"], metadata={"type": "agent"})
def reconcile_data_node(state: ApplicationState) -> ApplicationState:
    logger.info("Starting data reconciliation node")
    logger.info(f"Status update: Recocilliation fields")
    StatusTracker.set_status("🔍 Reconciling data")
    try:
        mismatches = reconcile_fields(
            state['name'], state['extracted_name'],
            state['phone'], state['extracted_phone'],
            state['address'], state['extracted_address'],
            state['income'], state['extracted_income'],
            state['loans'], state['extracted_loans']
        )
        new_state = {**state, 'mismatches': mismatches}
        
        if mismatches:
            logger.warning(f"Data reconciliation found mismatches: {', '.join(mismatches)}")
        else:
            logger.info("All data reconciled successfully")
            
        log_state_change("RECONCILE_DATA", new_state)
        return new_state
    except Exception as e:
        logger.error(f"Data reconciliation failed: {str(e)}")
        raise

@traceable(name="Run Validation", tags=["agent"], metadata={"type": "agent"})
def run_validation_node(state: ApplicationState) -> ApplicationState:
    """Run all data validations"""
    logger.info("Starting data validation node")
    logger.info(f"Status update: 🔍 Validating information")
    StatusTracker.set_status("🔍 Validating information")
    try:
        validation_results = run_all_validations(
            state['emirates_id'],
            state['name'],
            state['address'],
            state['dependents']
        )
        
        new_state = {
            **state,
            'validation_results': validation_results
        }
        
        logger.info(f"Validation completed - All valid: {validation_results['all_valid']}")
        log_state_change("RUN_VALIDATION", new_state)
        return new_state
    except Exception as e:
        logger.error(f"Data validation failed: {str(e)}")
        raise

@traceable(name="check Validation", tags=["tool"], metadata={"type": "tool"})
def check_validation(state: ApplicationState) -> str:
    """Determine next step based on validation results"""
    try:
        logger.info("Starting data validation from third party API services")
        if not state.get('validation_results'):
            logger.error("Validation results missing!")
            return "end"  # Fail safe

        if state['validation_results'].get('all_valid', False):
            return "continue_to_evaluate"  # Changed from "evaluate"
        return "end"
    except Exception as e:
        logger.error(f"Exception in check_validation: {str(e)}")
        return "end"

@traceable(name="Evaluate Financial Assistance", tags=["llm", "financial"], metadata={"type": "llm"})
def evaluate_financial_assistance_node(state: ApplicationState) -> ApplicationState:
    logger.info("Starting financial assistance evaluation")
    # logger.info(f"Current workflow state at evaluation: {state}")
    logger.info(f"Status update: 🤖 Running AI evaluation")
    StatusTracker.set_status("🔍 🤖 Running AI evaluation")
    try:
        # Prepare data for XGBoost validator
        validation_input = {
            'income': state['extracted_income'],
            'loans': state['extracted_loans'], 
            'dependents': state['dependents']
        }
        
        # Call the validator
        validation_result = validator.validate(validation_input)
        logger.info(f"ML validation_result received: {validation_result}")
        
        # Prepare LLM input with all required fields and detailed ML output
        llm_input = {
            'income': state['extracted_income'],
            'loans': state['extracted_loans'],
            'dependents': state['dependents'],
            'ml_validation': {
                'eligible': validation_result['eligible'],
                'confidence': validation_result.get('confidence'),
                'model_version': validation_result.get('model_version'),
                'status': validation_result.get('status'),
                'shap_values': validation_result.get('shap_values')  # SHAP analysis
            },
            'eligibility_status': 'eligible' if validation_result['eligible'] else 'not_eligible',
            'name': state['name'],
            'emirates_id': state['emirates_id'],
            'phone': state['phone'],
            'mismatches': state.get('mismatches', []),
            'address': state['address']
        }
        
        # Get LLM response
        response = ollama_financial_assistance_response(**llm_input)
        
        return {
            **state,
            'ollama_response': response,
            'validation_result': validation_result
        }
        
    except Exception as e:
        logger.error(f"Evaluation failed: {str(e)}")
        return {
            **state,
            'errors': str(e),
            'validation_result': None
        }

def check_reconciliation(state: ApplicationState) -> str:
    if state['mismatches']:
        logger.info("Routing to END due to data mismatches")
        return "end"
    logger.info("Routing to data valiation on third party API services")
    return "validate"

@traceable(name="Generate Recommendations", run_type="chain")
def generate_recommendations_node(state: ApplicationState) -> ApplicationState:
    logger.info("Starting recommendation generation")
    StatusTracker.set_status("💼 Generating career recommendations") 
    try:
        if state.get('validation_result', {}).get('eligible') and state.get('resume_file'):
            from agents.recommendation_agent import RecommendationAgent
            
            recommender = RecommendationAgent()
            recommendations = recommender.generate_recommendations(
                financial_approved=True,
                resume_text=state.get('resume_file'),
                financial_data={
                    'income': state['extracted_income'],
                    'loans': state['extracted_loans'],
                    'dependents': state['dependents']
                }
            )
            
            return {
                **state,
                'recommendations': recommendations
            }
        return state
    except Exception as e:
        logger.error(f"Recommendation generation failed: {str(e)}")
        return state

workflow = StateGraph(ApplicationState)

workflow.add_node("extract_documents", extract_documents_node)
workflow.add_node("reconcile_data", reconcile_data_node)
workflow.add_node("run_validation", run_validation_node)
workflow.add_node("evaluate_financial_assistance", evaluate_financial_assistance_node)
workflow.add_node("generate_recommendations", generate_recommendations_node)

workflow.set_entry_point("extract_documents")
workflow.add_edge("extract_documents", "reconcile_data")
workflow.add_conditional_edges(
    "reconcile_data",
    check_reconciliation,
    {
        "end": END,
        "validate": "run_validation"
    }
)
workflow.add_conditional_edges(
    "run_validation",
    check_validation,
    {
        "end": "evaluate_financial_assistance",  # Route both to LLM node
        "continue_to_evaluate": "evaluate_financial_assistance"  # Match node name exactly
    }
)
# workflow.add_edge("evaluate_financial_assistance", END)
workflow.add_edge("evaluate_financial_assistance", "generate_recommendations")
workflow.add_edge("generate_recommendations", END)

app = workflow.compile()