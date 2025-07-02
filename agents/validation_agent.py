# validation_agent.py
import requests
import time
import random
from utils.logger import get_logger
from langsmith import traceable


logger = get_logger("validation_agent")

# Dummy API endpoints (simulated)
BANK_VALIDATION_API = "https://dummy-bank-api.com/validate"
CREDIT_VALIDATION_API = "https://dummy-credit-api.com/verify"
GOVT_VALIDATION_API = "https://dummy-govt-api.com/validate"

def should_fail_demo(field_name: str, field_value: str) -> bool:
    """Check if field contains demo failure trigger"""
    if isinstance(field_value, str):
        # For text fields (name, address)
        if "DEMO" in field_value.upper():
            logger.warning(f"Demo failure triggered for {field_name}")
            return True
    elif isinstance(field_value, (int, float)):
        # For numeric fields (emirates ID, phone)
        if "911" in str(field_value):
            logger.warning(f"Demo failure triggered for {field_name}")
            return True
    return False

def validate_bank_data(emirates_id: str) -> dict:
    """Validate bank data - fails if emirates_id contains '911'"""
    if should_fail_demo("bank_validation", emirates_id):
        return {
            "valid": False,
            "message": "Bank validation failed (DEMO)",
            "details": "Demo failure triggered via Emirates ID containing '911'"
        }
    
    try:
        time.sleep(0.5)
        success = random.random() > 0.1  # 90% success rate
        return {
            "valid": success,
            "message": "Bank account validated" if success else "Bank account validation failed",
            "details": f"Emirates ID: {emirates_id} validated"
        }
    except Exception as e:
        logger.error(f"Bank validation error: {str(e)}")
        return {"valid": False, "message": "Bank validation service unavailable"}

def validate_credit_data(emirates_id: str) -> dict:
    """Validate credit data - fails if emirates_id contains '911'"""
    if should_fail_demo("credit_validation", emirates_id):
        return {
            "valid": False,
            "message": "Credit validation failed (DEMO)",
            "details": "Demo failure triggered via Emirates ID containing '911'"
        }
    
    try:
        time.sleep(0.7)
        success = random.random() > 0.15  # 85% success rate
        return {
            "valid": success,
            "message": "Credit data validated" if success else "Credit validation failed",
            "details": f"Credit history verified for Emirates ID: {emirates_id}"
        }
    except Exception as e:
        logger.error(f"Credit validation error: {str(e)}")
        return {"valid": False, "message": "Credit validation service unavailable"}

def validate_govt_data(emirates_id: str, name: str, address: str, dependents: int) -> dict:
    """Validate govt data - fails if any field contains 'DEMO'"""
    for field, value in [("name", name), ("address", address), ("emirates_id", emirates_id)]:
        if should_fail_demo(f"govt_validation_{field}", value):
            return {
                "valid": False,
                "message": f"Government validation failed (DEMO - {field})",
                "details": f"Demo failure triggered via {field} containing 'DEMO' or '911'"
            }
    
    try:
        time.sleep(1.0)
        success = random.random() > 0.05  # 95% success rate
        return {
            "valid": success,
            "message": "Government records validated" if success else "Government validation failed",
            "details": f"Verified identity for {name}"
        }
    except Exception as e:
        logger.error(f"Government validation error: {str(e)}")
        return {"valid": False, "message": "Government validation service unavailable"}

def run_all_validations(emirates_id: str, name: str, address: str, dependents: int) -> dict:
    """Run all validation checks"""
    logger.info("Starting comprehensive data validation")
    
    bank_validation = validate_bank_data(emirates_id)
    credit_validation = validate_credit_data(emirates_id)
    govt_validation = validate_govt_data(emirates_id, name, address, dependents)
    
    all_valid = (
        bank_validation["valid"] and 
        credit_validation["valid"] and 
        govt_validation["valid"]
    )
    
    return {
        "all_valid": all_valid,
        "bank_validation": bank_validation,
        "credit_validation": credit_validation,
        "govt_validation": govt_validation
    }