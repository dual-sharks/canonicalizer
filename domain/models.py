# domain/models.py
from dataclasses import dataclass

@dataclass
class Insight:
    phrase: str
    tag: str
    category: str
    descriptor: str
    sentiment: str
    sentence: str
    topic: str
