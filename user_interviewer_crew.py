import json
from crewai import Agent, Task, Crew
from pydantic import BaseModel, Field
from enum import Enum

class LevelEnum(str, Enum):
    unknown = "unknown"
    low = "low"
    high = "high"

class InterviewSummaryDataModel(BaseModel):
    """Data schema for interview summary."""
    noise_level: LevelEnum = Field(..., description="Level of noise in the background of the user's smart home hub")
    interaction_frequency: LevelEnum = Field(..., description="Frequency of user-hub interactions")
    interaction_types: str = Field(..., description="Types of user-hub voice interactions")
    energy_sensitivity: LevelEnum = Field(..., description="How sensitive the user is to energy consumption of the voice recognition AI model in the smart home hub")
    accuracy_sensitivity: LevelEnum = Field(..., description="How sensitive the user is to the accuracy of the voice recognition AI model in the smart home hub")
    latency_sensitivity: LevelEnum = Field(..., description="How sensitive the user is to the latency of the voice recognition AI model in the smart home hub")



def init_user_interviewer_crew(user_chat_tool, final_callback=None):
    interview_agent = Agent(
        role="User Interviewer",
        goal="Chat with the user to gather information about how user uses the voice recognition AI model (federated learning model) in a smart home hub (federated learning client)",
        backstory="Expert in conducting friendly interviews with attention to details."
        "We need to help the federation server to plan quantization level for each user based on how he/she uses the smart home hub and his/her personal preferences."
        "Use the user_chat_tool to send a message to user."
        "Use the ",
        tools=[user_chat_tool],
        verbose=True,
    )

    interview_task = Task(
        description="""Chat with user to learn about how the user uses to the smart home hub and the user's preference.
            You want to collect as much details as possible in the following aspects:
            - the noise level (e.g., low or high) in the background of the user's smart home hub 
            - how often (e.g., less frequent or frequent) the user interacts with the smart home hub 
            - the types of voice interactions the user has (what the user says) with the smart home hub 
            - how sensitive (e.g., low or high) the user is to the energy consumption of the voice recognition AI model in the smart home hub
            - how sensitive (e.g., low or high) the user is to the accuracy of the voice recognition AI model in the smart home hub
            - how sensitive (e.g., low or high) the user is to the latency of the voice recognition AI model in the smart home hub.

            When the interview is over, summarize the information collected and output the summary as a JSON object. 
            Output the JSON content only. Do NOT wrap it inside markdown code block (```json ```).
            """,
        expected_output=json.dumps(InterviewSummaryDataModel.model_json_schema(), indent=2),
        agent=interview_agent,
        # async_execution=True  # Required for waiting on user input
        callback=final_callback,
    )

    crew = Crew(
        agents=[interview_agent],
        tasks=[interview_task],
        verbose=True
    )

    return crew
