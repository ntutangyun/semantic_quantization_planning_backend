import json
from crewai import Agent, Task, Crew, Process
from pydantic import BaseModel, Field

from common_data_models import LevelEnum
from context_quantization_feedback_rag_tool import ContextQuantizationFeedbackRAGTool

class InterviewSummaryDataModel(BaseModel):
    """Data schema for interview summary."""
    noise_level: LevelEnum = Field(..., description="Level of noise in the background of the user's smart home hub")
    interaction_frequency: LevelEnum = Field(..., description="Frequency of user-hub interactions")
    interaction_types: str = Field(..., description="Types of user-hub voice interactions")
    energy_sensitivity: LevelEnum = Field(..., description="How sensitive the user is to energy consumption of the voice recognition AI model in the smart home hub")
    accuracy_sensitivity: LevelEnum = Field(..., description="How sensitive the user is to the accuracy of the voice recognition AI model in the smart home hub")
    latency_sensitivity: LevelEnum = Field(..., description="How sensitive the user is to the latency of the voice recognition AI model in the smart home hub")

class ContextQuantizationEvaluationDataModel(BaseModel):
    """Data schema for context-quantization evaluation."""
    bit_4: LevelEnum = Field(..., description="Potential contribution for 4 bit quantization")
    bit_6: LevelEnum = Field(..., description="Potential contribution for 6 bit quantization")
    bit_8: LevelEnum = Field(..., description="Potential contribution for 8 bit quantization")
    bit_12: LevelEnum = Field(..., description="Potential contribution for 12 bit quantization")
    bit_16: LevelEnum = Field(..., description="Potential contribution for 16 bit quantization")
    bit_32: LevelEnum = Field(..., description="Potential contribution for 32 bit quantization")

def init_user_interviewer_crew(user_chat_tool, interview_callback=None, context_quantization_evaluation_callback=None):
    
    interview_agent = Agent(
        role="User Interviewer",
        goal="Chat with the user to gather information about how user uses the voice recognition AI model (federated learning model) in a smart home hub (federated learning client)",
        backstory="Expert in conducting friendly interviews with attention to details."
        "We need to help the federation server to plan quantization level for each user based on how he/she uses the smart home hub and his/her personal preferences."
        "Use the user_chat_tool to send a message to user.",
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
            Wrap the json content inside the markdown code block (```json ```).
            """,
        expected_output=json.dumps(InterviewSummaryDataModel.model_json_schema(), indent=2),
        agent=interview_agent,
        # async_execution=True  # Required for waiting on user input
        callback=interview_callback,
    )

    context_quantization_evaluation_agent = Agent(
        role="User's Context-Quantization-Feedback Evaluator",
        goal="Evaluate the potential contribution (low, high) for each quantization level (4, 6, 8, 12, 16, 32) based on user's usage context.",
        backstory="We need to help the federation server to plan quantization level for each user based on how he/she uses the smart home hub (with voice recognition AI model participated in federated learning) and his/her personal preferences."
        "Your task is to evaluate the potential contribution (low, high) for each quantization level if it is used during federated learning, based on how the user usages the smart home hub as well as the previous user feedbacks on different context-quantization pairs."
        "Use the context_quantization_feedback_rag_tool to retrieve previous user feedbacks.",
        tools=[ContextQuantizationFeedbackRAGTool()],
        verbose=True,
    )

    context_quantization_evaluation_task = Task(
        description="""Evaluate the potential contribution (low, high) for each quantization level (4, 6, 8, 12, 16, 32) based on user's usage context.
            based on the user's usage context and the previous user feedbacks on different context-quantization pairs,
            output the contribution level (low or high) for each quantization level, formatted in a JSON object.
            Wrap the json content inside the markdown code block (```json ```).
            """,
        expected_output=json.dumps(ContextQuantizationEvaluationDataModel.model_json_schema(), indent=2),
        agent=context_quantization_evaluation_agent,
        # async_execution=True  # Required for waiting on user input
        callback=context_quantization_evaluation_callback,
    )

    crew = Crew(
        agents=[interview_agent, context_quantization_evaluation_agent],
        tasks=[interview_task, context_quantization_evaluation_task],
        verbose=True,
        process=Process.sequential  # or Process.hierarchical
    )

    return crew
