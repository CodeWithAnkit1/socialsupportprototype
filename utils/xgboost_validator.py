import joblib
import pandas as pd
from typing import Dict, Any
from utils.logger import get_logger
import shap  # <-- Add this import
from langsmith import traceable, trace


logger = get_logger("xgboost_validator")

class XGBoostValidator:
    def __init__(self):
        self.model = None
        self.features = None
        self.explainer = None  # <-- Add this line
        self._load_model()
        

    def _load_model(self):
        try:
            self.model = joblib.load("models/social_support_xgboost_model.pkl")
            self.features = joblib.load("models/model_features.pkl")
            logger.info("XGBoost model loaded successfully")
            # Initialize SHAP explainer after model is loaded
            import xgboost
            self.explainer = shap.TreeExplainer(self.model)
        except Exception as e:
            logger.error(f"Failed to load model: {str(e)}")
            raise
    @traceable(name="XGBoost Validator", tags=["tool", "ml"], metadata={"type": "tool"})
    def validate(self, input_data: Dict[str, Any]) -> Dict[str, Any]:
        """Validate eligibility using the pre-trained model and capture SHAP values"""
        try:
            # Prepare input DataFrame
            input_df = pd.DataFrame([{
                'income': input_data.get('income', 0),
                'loans': input_data.get('loans', 0),
                'dependents': input_data.get('dependents', 0),
                'employment_status': 1,  # Default assumptions
                'existing_benefits': 0
            }], columns=self.features)
            
            # Make prediction
            proba = self.model.predict_proba(input_df)[0][1]
            
            # SHAP analysis
            shap_values = self.explainer.shap_values(input_df)
            # Convert SHAP values to a dict for easy serialization
            shap_dict = {feature: float(shap_values[0][i]) for i, feature in enumerate(self.features)}
            
            return {
                'eligible': bool(proba > 0.5),
                'confidence': float(proba),
                'model_version': '1.0',
                'status': 'success',
                'shap_values': shap_dict  # <-- Add SHAP values to output
            }
        except Exception as e:
            logger.error(f"Validation failed: {str(e)}")
            return {
                'eligible': False,
                'error': str(e),
                'status': 'error'
            }

# Singleton instance
validator = XGBoostValidator()