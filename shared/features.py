from pydantic import BaseModel
from pydantic import Field
import torch
from typing import Union


class FeatureSample(BaseModel):
    text: str
    act: float

    def __eq__(self, other):
        if not isinstance(other, FeatureSample):
            return False
        return self.text == other.text and self.act == other.act
    
    def __hash__(self):
        return hash((self.text, self.act))

    class Config:
        allow_mutation = True
        frozen = False


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
