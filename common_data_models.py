from pydantic import BaseModel, Field
from enum import Enum


class LevelEnum(str, Enum):
    unknown = "unknown"
    low = "low"
    high = "high"

