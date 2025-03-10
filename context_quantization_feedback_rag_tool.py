import chromadb
from dotenv import load_dotenv
import os
import chromadb.utils.embedding_functions as embedding_functions

# load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))


openai_ef = embedding_functions.OpenAIEmbeddingFunction(
    api_key=os.getenv("OPENAI_API_KEY"), model_name="text-embedding-3-small"
)

chroma_client = chromadb.PersistentClient(
    path=os.path.join(
        os.path.dirname(__file__), "./context_quantization_feedback_rag_db"
    )
)
chroma_client.heartbeat()

collection = chroma_client.get_or_create_collection(
    name="context_quantization_feedback_rag_collection", embedding_function=openai_ef
)

results = collection.query(
    query_texts=[
        "This is a query document about florida"
    ],  # Chroma will embed this for you
    n_results=2,  # how many results to return
)
print(results)

from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field
from common_data_models import LevelEnum


class ContextQuantizationFeedbackRAGToolInput(BaseModel):
    """Input schema for ContextQuantizationFeedbackRAGTool."""

    noise_level: LevelEnum = Field(
        ..., description="Level of noise in the background of the user's smart home hub"
    )
    interaction_frequency: LevelEnum = Field(
        ..., description="Frequency of user-hub interactions"
    )
    interaction_types: str = Field(
        ..., description="Types of user-hub voice interactions"
    )


class ContextQuantizationFeedbackRAGTool(BaseTool):
    name: str = "context_quantization_feedback_rag_tool"
    description: str = (
        "Tool to search for historical user feedback data on different context-quantization pairs."
    )
    args_schema: Type[BaseModel] = ContextQuantizationFeedbackRAGToolInput

    def _run(
        self,
        noise_level: LevelEnum,
        interaction_frequency: LevelEnum,
        interaction_types: str,
    ) -> str:
        """retrieve historical user feedback data on different context-quantization pairs.
        You only needs to provide the noise level, interaction frequency, and interaction types.
        The tool will return a string describing all feedback data on matched/similar context.
        """

        # query the database
        results = collection.query(
            query_texts=[
                f"noise level: {noise_level}, interaction frequency: {interaction_frequency}, interaction types: {interaction_types}"
            ],
            n_results=3,
        )

        return "\n".join(results["documents"][0])
