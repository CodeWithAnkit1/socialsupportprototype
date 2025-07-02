import os
from typing import Dict, Optional
from utils.logger import get_logger
from utils.resume_parser import parse_resume  # We'll create this next
from llm_utils.ollama_wrapper import get_local_llm
from langsmith import traceable
logger = get_logger("recommendation_agent")

class RecommendationAgent:
    def __init__(self):
        self.llm = get_local_llm()
        
    def generate_recommendations(self, 
                               financial_approved: bool,
                               resume_text: Optional[str] = None,
                               financial_data: Optional[Dict] = None) -> Dict:
        """
        Generate recommendations based on resume and financial approval status
        
        Args:
            financial_approved: Whether financial assistance was approved
            resume_text: Extracted text from resume
            financial_data: Dictionary containing financial information
            
        Returns:
            Dictionary containing recommendations
        """
        if not financial_approved:
            return {"recommendations": [], "status": "financial_assistance_not_approved"}
            
        if not resume_text:
            return {"recommendations": [], "status": "no_resume_provided"}
            
        try:
            # Parse resume and extract skills/experience
            resume_summary = parse_resume(resume_text)
            
            # Generate recommendations using LLM
            prompt = self._build_prompt(resume_summary, financial_data)
            recommendations = self.llm.generate(prompt)
            
            return {
                "recommendations": recommendations,
                "resume_summary": resume_summary,
                "status": "success"
            }
        except Exception as e:
            logger.error(f"Recommendation generation failed: {str(e)}")
            return {"recommendations": [], "status": "error"}

    def _build_prompt(self, resume_summary: Dict, financial_data: Dict) -> str:
        """Build LLM prompt for generating recommendations"""
        return f"""
        You are a career advisor for UAE government social support recipients.
        Based on the following applicant profile, suggest specific job opportunities
        or educational programs that would help improve their financial situation.

        Applicant Profile:
        - Skills: {resume_summary.get('skills', [])}
        - Experience: {resume_summary.get('experience', [])}
        - Education: {resume_summary.get('education', [])}
        - Current Financial Situation:
          * Monthly Income: AED {financial_data.get('income', 0):,.2f}
          * Dependents: {financial_data.get('dependents', 0)}
          * Loan Amount: AED {financial_data.get('loans', 0):,.2f}

        Provide 3-5 concrete recommendations including:
        - Specific job titles to apply for
        - Training programs that would increase employability
        - Educational opportunities
        - Government support programs they may qualify for

        Format your response as a bulleted list with brief explanations for each recommendation.
        """