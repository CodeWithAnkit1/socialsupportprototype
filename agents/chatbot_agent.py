from langchain.agents import initialize_agent, AgentType, Tool
from langchain.tools import tool
from llm_utils.ollama_wrapper import get_local_llm
from callbacks.logging_callback import LoggingCallbackHandler
from agents.validation_agent import run_all_validations

@tool
def greet(name: str) -> str:
    """Returns a greeting for the user."""
    return f"Hello, {name}! Welcome to the Social Support Chatbot."

def get_chat_agent():
    callback = LoggingCallbackHandler()
    llm = get_local_llm()
    
    tools = [
        Tool(
            name="Greet",
            func=greet,
            description="Greets the user when they type their name"
        )
    ]

@tool
def validate_user_data(emirates_id: str, name: str, address: str, dependents: int) -> dict:
    """Validates user data against government, bank, and credit systems"""
    return run_all_validations(emirates_id, name, address, dependents)

def get_chat_agent():
    callback = LoggingCallbackHandler()
    llm = get_local_llm()
    
    tools = [
        Tool(
            name="Greet",
            func=greet,
            description="Greets the user when they type their name"
        ),
        # NEW VALIDATION TOOL
        Tool(
            name="ValidateUserData",
            func=validate_user_data,
            description=(
                "Validates user information against official systems. "
                "Requires Emirates ID, name, address, and number of dependents. "
                "Use when user asks about data verification or application status."
            )
        )
    ]

    agent = initialize_agent(
        tools=tools,
        llm=llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        verbose=True,
        callbacks=[callback]
    )
    return agent