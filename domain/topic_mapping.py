# domain/topic_mapping.py
TOPIC_MAP = {
    "too fast": "enemy speed",
    "broken": "visual bug",
    "low fps": "performance"
}

def map_descriptor_to_topic(descriptor: str) -> str:
    for key in TOPIC_MAP:
        if key in descriptor.lower():
            return TOPIC_MAP[key]
    return "unspecified"
