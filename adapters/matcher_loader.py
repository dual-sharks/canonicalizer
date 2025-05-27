# adapters/matcher_loader.py
import yaml
from spacy.matcher import PhraseMatcher

def load_matcher(nlp, yaml_path):
    with open(yaml_path) as f:
        data = yaml.safe_load(f)

    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for tag, phrases in data.items():
        matcher.add(tag, [nlp(p) for p in phrases])
    return matcher
