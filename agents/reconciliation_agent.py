from utils.logger import get_logger
from typing import List
# from langsmith import traceable

logger = get_logger("reconciliation_agent")

def normalize_text(text: str) -> str:
    """Normalize text for comparison"""
    return text.strip().lower().replace(" ", "").replace("-", "").replace("+", "")

def normalize_address(addr: str) -> str:
    """Normalize address for comparison"""
    if not addr:
        return ""
    import re
    # Remove all punctuation
    addr = re.sub(r'[^\w\s]', '', addr)
    # Replace all whitespace with single space
    addr = re.sub(r'\s+', ' ', addr)
    return addr.strip().lower()

# @traceable(name="Reconciliation Agent", run_type="tool")
def reconcile_fields(
    name_sub: str, name_ext: str,
    phone_sub: str, phone_ext: str,
    address_sub: str, address_ext: str,
    income_sub: float, income_ext: float,
    loans_sub: float, loans_ext: float
) -> List[str]:
    """
    Reconcile fields between form submission and extracted data
    Returns list of mismatched field names
    """
    mismatches = []
    
    # Prepare data for tracing
    trace_data = {
        "submitted": {
            "name": name_sub,
            "phone": phone_sub,
            "address": address_sub,
            "income": income_sub,
            "loans": loans_sub
        },
        "extracted": {
            "name": name_ext,
            "phone": phone_ext,
            "address": address_ext,
            "income": income_ext,
            "loans": loans_ext
        }
    }
    
    # Name reconciliation
    if name_ext and normalize_text(name_sub) != normalize_text(name_ext):
        logger.warning(f"Name mismatch: '{name_sub}' vs '{name_ext}'")
        mismatches.append("Name")
    
    # Phone reconciliation
    if phone_ext:
        norm_phone_sub = normalize_text(phone_sub)
        norm_phone_ext = normalize_text(phone_ext)
        
        # Check substring match
        if (norm_phone_sub not in norm_phone_ext and 
            norm_phone_ext not in norm_phone_sub):
            logger.warning(f"Phone mismatch: '{phone_sub}' vs '{phone_ext}'")
            mismatches.append("Phone Number")
    
    # Address reconciliation
    if address_ext:
        norm_addr_sub = normalize_address(address_sub)
        norm_addr_ext = normalize_address(address_ext)
        
        # Calculate similarity score
        similarity = 0
        if norm_addr_sub and norm_addr_ext:
            common_chars = set(norm_addr_sub) & set(norm_addr_ext)
            similarity = len(common_chars) / max(len(norm_addr_sub), len(norm_addr_ext))
        
        if similarity < 0.7:  # 70% similarity threshold
            logger.warning(f"Address mismatch: '{address_sub}' vs '{address_ext}' (similarity: {similarity:.2f})")
            mismatches.append("Address")
    
    # Income reconciliation
    if abs(float(income_sub) - float(income_ext)) > 500:
        logger.warning(f"Income mismatch: {income_sub} vs {income_ext}")
        mismatches.append("Income")
    
    # Loan reconciliation
    if abs(float(loans_sub) - float(loans_ext)) > 500:
        logger.warning(f"Loan amount mismatch: {loans_sub} vs {loans_ext}")
        mismatches.append("Loan Amount")
    
    # Add results to trace data
    trace_data["result"] = {
        "mismatches": mismatches,
        "mismatch_count": len(mismatches)
    }
    
    logger.info(f"Found {len(mismatches)} mismatches: {mismatches}")
    return mismatches