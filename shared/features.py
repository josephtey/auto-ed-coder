from pydantic import BaseModel
from pydantic import Field
import torch


class FeatureSample(BaseModel):
    text: str
    act: float


class Feature(BaseModel):
    index: int
    label: str
    attributes: str
    reasoning: str
    density: float
    confidence: float
    high_act_samples: list[FeatureSample]
    low_act_samples: list[FeatureSample]

    class Config:
        arbitrary_types_allowed = True
