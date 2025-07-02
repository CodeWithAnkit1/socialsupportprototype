from langchain_core.callbacks import BaseCallbackHandler
from langchain.callbacks.manager import CallbackManager
from utils.logger import get_logger

logger = get_logger("llm_interaction")
# client = Client()


class LoggingCallbackHandler(BaseCallbackHandler):
    def on_llm_start(self, serialized, prompts, **kwargs):
        """Log the prompts sent to LLM"""
        for i, prompt in enumerate(prompts):
            logger.info(f"LLM INPUT [{i+1}/{len(prompts)}]:\n{prompt}")
    
    def on_llm_end(self, response, **kwargs):
        """Log the LLM response"""
        generations = response.generations
        for i, generation in enumerate(generations):
            for j, gen in enumerate(generation):
                logger.info(f"LLM OUTPUT [{i+1}.{j+1}]:\n{gen.text}")
    
    def on_llm_error(self, error, **kwargs):
        """Log LLM errors"""
        logger.error(f"LLM ERROR: {str(error)}")

callback_manager = CallbackManager([LoggingCallbackHandler()])