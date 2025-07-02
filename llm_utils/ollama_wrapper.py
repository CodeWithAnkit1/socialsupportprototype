from langchain_community.llms import Ollama
from callbacks.logging_callback import LoggingCallbackHandler
from utils.logger import get_logger

logger = get_logger("ollama_wrapper")

def get_local_llm():
    callback = LoggingCallbackHandler()
    llm = Ollama(
        model="llama3",
        temperature=0.3,
        callbacks=[callback]
    )
    logger.info("Initialized Ollama LLM with logging callback")
    return llm