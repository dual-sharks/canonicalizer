import spacy
import yaml
from spacy.matcher import PhraseMatcher
from transformers import pipeline

class Canonicalizer:
    def __init__(self, general_tag_path: str, creature_tag_path: str):
        self.nlp = spacy.load("en_core_web_sm")
        self.sentiment_model = pipeline("sentiment-analysis")

        self.general_matcher = self._build_matcher(general_tag_path)
        self.creature_matcher = self._build_matcher(creature_tag_path)

    def _build_matcher(self, yaml_path):
        with open(yaml_path, "r") as f:
            config = yaml.safe_load(f)

        matcher = PhraseMatcher(self.nlp.vocab, attr="LOWER")
        for tag, phrases in config.items():
            patterns = [self.nlp(phrase) for phrase in phrases]
            matcher.add(tag, patterns)
        return matcher

    def _get_sentiment(self, phrase: str) -> str:
        result = self.sentiment_model(phrase[:512])[0]
        return result['label'].lower()

    def _extract_descriptors(self, sent_doc, match_span):
        root = match_span.root
        while root.head != root and root.pos_ != "VERB":
            root = root.head

        clause_tokens = list(root.subtree)
        clause_end = sent_doc.end
        for token in clause_tokens:
            if token.dep_ == "cc" and token.text.lower() in {"and", "but"}:
                clause_end = token.i
                break

        descriptors = []
        for token in clause_tokens:
            if match_span.start <= token.i < clause_end and token.pos_ == "ADJ":
                modifiers = [child.text for child in token.lefts if child.pos_ == "ADV"]
                modifiers = [m for m in modifiers if m.lower() not in {"very", "really", "super", "way"}]
                phrase = " ".join(modifiers + [token.text])
                descriptors.append(phrase)

            elif (
                match_span.start <= token.i < clause_end
                and token.pos_ in {"ADV", "VERB"}
                and not token.is_stop
            ):
                descriptors.append(token.text)

        return " ".join(descriptors).strip()

    def analyze(self, text: str):
        doc = self.nlp(text)
        results = []

        for matcher, category in [
            (self.general_matcher, "game"),
            (self.creature_matcher, "creature")
        ]:
            matches = matcher(doc)
            for match_id, start, end in matches:
                tag = self.nlp.vocab.strings[match_id]
                span = doc[start:end]
                sent = span.sent
                descriptor = self._extract_descriptors(sent, span)
                sentiment = self._get_sentiment(sent.text)

                results.append({
                    "phrase": span.text,
                    "tag": tag,
                    "category": category,
                    "descriptor": descriptor,
                    "sentiment": sentiment,
                    "sentence": sent.text
                })
        return results
