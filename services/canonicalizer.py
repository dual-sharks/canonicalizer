import spacy
import yaml
from spacy.matcher import PhraseMatcher
from transformers import pipeline
from domain.topic_mapping import map_descriptor_to_topic

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
        descriptors = []
        seen = set()

        for token in match_span:
            # adjectives directly modifying the token
            for child in token.children:
                if child.dep_ in {"amod", "acomp"} and child.pos_ == "ADJ":
                    advs = [c.text for c in child.children if c.dep_ == "advmod"]
                    advs = [a for a in advs if a.lower() not in {"very", "really", "super", "way"}]
                    phrase = " ".join(advs + [child.text])
                    if phrase not in seen:
                        descriptors.append(phrase)
                        seen.add(phrase)

            # adjective complements of governing verbs
            if token.head.pos_ in {"VERB", "AUX"} and token.dep_ in {"nsubj", "dobj", "obj", "nsubjpass"}:
                for child in token.head.children:
                    if child.dep_ == "acomp" and child.pos_ == "ADJ":
                        advs = [c.text for c in child.children if c.dep_ == "advmod"]
                        advs = [a for a in advs if a.lower() not in {"very", "really", "super", "way"}]
                        phrase = " ".join(advs + [child.text])
                        if phrase not in seen:
                            descriptors.append(phrase)
                            seen.add(phrase)

                # optionally include the verb itself unless it's a copula
                if token.head.lemma_ != "be":
                    verb = token.head.text
                    if verb not in seen:
                        descriptors.append(verb)
                        seen.add(verb)

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
                topic = map_descriptor_to_topic(descriptor)

                results.append({
                    "phrase": span.text,
                    "tag": tag,
                    "category": category,
                    "descriptor": descriptor,
                    "sentiment": sentiment,
                    "sentence": sent.text,
                    "topic": topic
                })
        return results
