from llm_utils.ollama_wrapper import get_local_llm
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from utils.logger import get_logger
from langsmith import traceable

logger = get_logger("utils")

@traceable(name="Ollama Financial Assistance LLM", tags=["llm", "financial"], metadata={"type": "llm"})
def ollama_financial_assistance_response(
    income: float,
    loans: float,
    dependents: int,
    ml_validation: dict,
    eligibility_status: str,
    name: str,
    mismatches: list,
    emirates_id: str,
    phone: str,
    address: str
) -> str:
    logger.info(f"Calling LLM for financial assistance evaluation")
    
    # Create the LLM chain
    llm = get_local_llm()
    prompt = ChatPromptTemplate.from_template(
        "You are a financial assistance advisor for the UAE government. "
        "Evaluate the following application for social support:\n\n"
        "Emirates ID: {emirates_id}\n"
        "Name: {name}\n"
        "Phone: {phone}\n"
        "Address: {address}\n"
        "Dependents: {dependents}\n"
        "Monthly Income: AED {income}\n"
        "Total Loans: AED {loans}\n\n"
        "Based on UAE social support policies, determine if the applicant is eligible for assistance. "
        "Eligibility criteria: Monthly income < AED 5000 AND at least 2 dependents.\n\n"
        "Provide a detailed response explaining your decision."
    )
    chain = prompt | llm | StrOutputParser()
    
    # Invoke the chain
    response = chain.invoke({
        "emirates_id": emirates_id,
        "name": name,
        "phone": phone,
        "address": address,
        "dependents": dependents,
        "income": income,
        "loans": loans,
        "eligibility_status": 'eligible' if ml_validation['eligible'] else 'not eligible',
        "confidence": ml_validation['confidence'] * 100  # Convert to percentage
    })
    
    logger.info(f"LLM evaluation completed")
    return response

def format_currency(value):
    return f"AED {value:,.2f}"