def decide_eligibility(income, employment_status, dependents):
    if income < 4000 and employment_status == "unemployed" and dependents >= 2:
        return {"status": "Approved", "confidence": 0.91}
    return {"status": "Soft Decline", "reason": "Income or employment criteria not met"}