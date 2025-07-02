import re
from typing import Dict, List
import pdfplumber  # For PDF text extraction
from utils.logger import get_logger

logger = get_logger("resume_parser")

def parse_resume(pdf_path: str) -> Dict[str, List[str]]:
    """
    Extract text from resume PDF and identify key sections
    
    Args:
        pdf_path: Path to resume PDF file
        
    Returns:
        Dictionary containing parsed resume sections
    """
    try:
        with pdfplumber.open(pdf_path) as pdf:
            text = "\n".join(page.extract_text() for page in pdf.pages)
        
        return {
            "skills": _extract_skills(text),
            "experience": _extract_experience(text),
            "education": _extract_education(text),
            "raw_text": text[:1000] + "..."  # Store first 1000 chars
        }
    except Exception as e:
        logger.error(f"Failed to parse resume: {str(e)}")
        return {"error": str(e)}

def _extract_skills(text: str) -> List[str]:
    """Extract skills section from resume text"""
    skills = re.findall(r"(?i)(?:skills|technical skills|competencies)[:\n](.*?)(?:\n\n|\n\w|$)", text, re.DOTALL)
    return [skill.strip() for skill in skills[0].split("\n") if skill.strip()] if skills else []

def _extract_experience(text: str) -> List[str]:
    """Extract work experience from resume text"""
    experience = re.findall(r"(?i)(?:experience|work history)[:\n](.*?)(?:\n\n|\n\w|$)", text, re.DOTALL)
    return [exp.strip() for exp in experience[0].split("\n") if exp.strip()] if experience else []

def _extract_education(text: str) -> List[str]:
    """Extract education section from resume text"""
    education = re.findall(r"(?i)(?:education|qualifications)[:\n](.*?)(?:\n\n|\n\w|$)", text, re.DOTALL)
    return [edu.strip() for edu in education[0].split("\n") if edu.strip()] if education else []