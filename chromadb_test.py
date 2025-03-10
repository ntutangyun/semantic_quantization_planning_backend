import chromadb
from dotenv import load_dotenv
import os

# load environment variables
load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), ".env"))

import chromadb.utils.embedding_functions as embedding_functions

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

collection.add(
    documents=[
        "noise level: high, interaction frequency: low, interaction types: turn on lights, set thermostat. quantization level: 8 bit. user feedback: GOOD",
        "noise level: low, interaction frequency: high, interaction types: play music, set alarm. quantization level: 16 bit. user feedback: GOOD",
        "noise level: unknown, interaction frequency: unknown, interaction types: lock doors, unlock doors. quantization level: 4 bit. user feedback: POOR",
        "noise level: low, interaction frequency: low, interaction types: check weather, read news. quantization level: 32 bit. user feedback: GOOD",
        "noise level: high, interaction frequency: high, interaction types: control TV, play music. quantization level: 8 bit. user feedback: GOOD",
        "noise level: unknown, interaction frequency: high, interaction types: set reminders, control lights. quantization level: 16 bit. user feedback: GOOD",
        "noise level: low, interaction frequency: unknown, interaction types: set alarm, control thermostat. quantization level: 32 bit. user feedback: GOOD",
        "noise level: high, interaction frequency: unknown, interaction types: play music, set reminders. quantization level: 4 bit. user feedback: POOR",
        "noise level: unknown, interaction frequency: low, interaction types: control TV, lock doors. quantization level: 8 bit. user feedback: GOOD",
        "noise level: low, interaction frequency: high, interaction types: read news, check weather. quantization level: 16 bit. user feedback: GOOD",
    ],
    metadatas=[
        {"noise_level": "high", "interaction_frequency": "low", "interaction_types": "turn on lights, set thermostat", "quantization_level": 8, "user_feedback": "GOOD"},
        {"noise_level": "low", "interaction_frequency": "high", "interaction_types": "play music, set alarm", "quantization_level": 16, "user_feedback": "GOOD"},
        {"noise_level": "unknown", "interaction_frequency": "unknown", "interaction_types": "lock doors, unlock doors", "quantization_level": 4, "user_feedback": "POOR"},
        {"noise_level": "low", "interaction_frequency": "low", "interaction_types": "check weather, read news", "quantization_level": 32, "user_feedback": "GOOD"},
        {"noise_level": "high", "interaction_frequency": "high", "interaction_types": "control TV, play music", "quantization_level": 8, "user_feedback": "GOOD"},
        {"noise_level": "unknown", "interaction_frequency": "high", "interaction_types": "set reminders, control lights", "quantization_level": 16, "user_feedback": "GOOD"},
        {"noise_level": "low", "interaction_frequency": "unknown", "interaction_types": "set alarm, control thermostat", "quantization_level": 32, "user_feedback": "GOOD"},
        {"noise_level": "high", "interaction_frequency": "unknown", "interaction_types": "play music, set reminders", "quantization_level": 4, "user_feedback": "POOR"},
        {"noise_level": "unknown", "interaction_frequency": "low", "interaction_types": "control TV, lock doors", "quantization_level": 8, "user_feedback": "GOOD"},
        {"noise_level": "low", "interaction_frequency": "high", "interaction_types": "read news, check weather", "quantization_level": 16, "user_feedback": "GOOD"},
    ],
    ids=[
        "1",
        "2",
        "3",
        "4",
        "5",
        "6",
        "7",
        "8",
        "9",
        "10",
    ],
)

results = collection.query(
    query_texts=[
        "noise level: unknown, interaction frequency: low, interaction types: control TV"
    ],  # Chroma will embed this for you
    n_results=3,  # how many results to return
)
print("------")
print("\n".join(results["documents"][0]))
print("------")
