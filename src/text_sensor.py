from transformers import pipeline
import numpy as np
import re

# The selected Hug Face checkpoint exposes generic LABEL_n names in its config.
# Decode them back to the canonical GoEmotions order before FER mapping.
ID2GOEMOTION = {
    "LABEL_0": "admiration",
    "LABEL_1": "amusement",
    "LABEL_2": "anger",
    "LABEL_3": "annoyance",
    "LABEL_4": "approval",
    "LABEL_5": "caring",
    "LABEL_6": "confusion",
    "LABEL_7": "curiosity",
    "LABEL_8": "desire",
    "LABEL_9": "disappointment",
    "LABEL_10": "disapproval",
    "LABEL_11": "disgust",
    "LABEL_12": "embarrassment",
    "LABEL_13": "excitement",
    "LABEL_14": "fear",
    "LABEL_15": "gratitude",
    "LABEL_16": "grief",
    "LABEL_17": "joy",
    "LABEL_18": "love",
    "LABEL_19": "nervousness",
    "LABEL_20": "optimism",
    "LABEL_21": "pride",
    "LABEL_22": "realization",
    "LABEL_23": "relief",
    "LABEL_24": "remorse",
    "LABEL_25": "sadness",
    "LABEL_26": "surprise",
    "LABEL_27": "neutral",
}

# GoEmotions to FER2013 mapping
# FER2013 categories: Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral
GOEMOTIONS_MAPPING = {
    'admiration': 'Happy',
    'amusement': 'Happy',
    'anger': 'Angry',
    'annoyance': 'Angry',
    'approval': 'Happy',
    'caring': 'Happy',
    'confusion': 'Neutral',
    'curiosity': 'Neutral',
    'desire': 'Happy',
    'disappointment': 'Sad',
    'disapproval': 'Disgust',
    'disgust': 'Disgust',
    'embarrassment': 'Sad',
    'excitement': 'Happy',
    'fear': 'Fear',
    'gratitude': 'Happy',
    'grief': 'Sad',
    'joy': 'Happy',
    'love': 'Happy',
    'nervousness': 'Fear',
    'optimism': 'Happy',
    'pride': 'Happy',
    'realization': 'Neutral',
    'relief': 'Happy',
    'remorse': 'Sad',
    'sadness': 'Sad',
    'surprise': 'Surprise',
    'neutral': 'Neutral'
}

FER_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def neutral_distribution() -> np.ndarray:
    p = np.zeros(len(FER_CLASSES), dtype=float)
    p[FER_CLASSES.index('Neutral')] = 1.0
    return p

class LinguisticSensor:
    def __init__(self):
        # We use a pure BERT-base model fine-tuned on GoEmotions to strictly align with the PDF
        self.classifier = pipeline(task="text-classification", model="nmb-paperspace-hf/bert-base-uncased-go_emotions", top_k=None)
        
    def process(self, text: str, strict_mode: bool = False) -> np.ndarray:
        norm_text = text.lower().strip()
        norm_text = re.sub(r'[^\w\s]', '', norm_text)
        
        if not norm_text:
            return neutral_distribution()
            
        # 1. Run inference on normalized text
        raw_outputs = self.classifier(norm_text)
        outputs = raw_outputs[0] if raw_outputs else []
        if isinstance(outputs, dict):
            outputs = [outputs]
        if not outputs:
            return neutral_distribution()
        
        # 2. Collapse GoEmotions into 7 FER buckets.
        # Strict mode uses max evidence per FER class for a cleaner multi-label collapse.
        # Demo mode uses sum so overlapping affective cues can stack before heuristics.
        fer_scores = {c: 0.0 for c in FER_CLASSES}
        for out in outputs:
            raw_label = out['label']
            score = float(out['score'])
            goemotion_label = ID2GOEMOTION.get(raw_label, raw_label)
            fer_label = GOEMOTIONS_MAPPING.get(goemotion_label, 'Neutral')

            if strict_mode:
                fer_scores[fer_label] = max(fer_scores[fer_label], score)
            else:
                fer_scores[fer_label] += score
            

        # ENHANCED MODE BOOSTS: transparent lexical calibration for short demo utterances.
        # Strict mode intentionally bypasses this so PDF evaluation uses only mapped model output.
        if not strict_mode:
            words = set(norm_text.split())

            def has_any(terms):
                return any(term in words or term in norm_text for term in terms)

            # HAPPY signals
            if has_any([
                "happy", "excited", "joy", "good", "great", "awesome",
                "won", "win", "jackpot", "celebrate", "fun", "funny",
                "laugh", "rich", "hurray", "hurrayy", "hurrayyy", "yay",
                "congrats", "congratulations", "bought"
            ]):
                fer_scores['Happy'] += 0.4

            positive_events = [
                "got my new", "got a new", "new car", "new job",
                "got promoted", "passed my", "bought my"
            ]
            if any(phrase in norm_text for phrase in positive_events):
                fer_scores['Happy'] += 0.4
                
            # SAD signals
            if has_any([
                "sad", "cry", "crying", "cried", "lost", "miss", "alone",
                "lonely", "hurt", "pain", "upset", "depressed", "broken"
            ]):
                fer_scores['Sad'] += 0.4

            # FEAR signals
            if has_any([
                "fear", "scared", "afraid", "nervous", "worried", "anxious",
                "terrified", "panic", "panicked"
            ]):
                fer_scores['Fear'] += 0.4

            # ANGER signals
            if has_any([
                "angry", "mad", "annoyed", "furious", "irritated",
                "frustrated", "hate"
            ]):
                fer_scores['Angry'] += 0.4
            
        # 3. Normalize first
        total = sum(fer_scores.values())
        if total == 0:
            return neutral_distribution()
            
        for k in fer_scores:
            fer_scores[k] /= total
                
        if not strict_mode:
            # If Sad/Happy is close to Neutral, promote it slightly
            if fer_scores['Neutral'] > 0:
                for emo in ['Sad', 'Happy', 'Fear', 'Angry']:
                    if fer_scores[emo] >= 0.4 * fer_scores['Neutral']:
                        fer_scores[emo] += 0.1
                        
            total = sum(fer_scores.values())
            if total > 0:
                for k in fer_scores:
                    fer_scores[k] /= total
                        
        p_text = np.array([fer_scores[c] for c in FER_CLASSES])

        neutral_idx = FER_CLASSES.index('Neutral')
        best_non_neutral_idx = np.argmax(p_text[:-1])
        best_non_neutral = p_text[best_non_neutral_idx]
        neutral_score = p_text[neutral_idx]

        # Demo-mode calibration: GoEmotions often over-represents Neutral after
        # mapping 28 labels into 7 FER classes. Strict mode leaves the mapped
        # model probabilities untouched for baseline PDF alignment.
        if not strict_mode and neutral_score > 0 and best_non_neutral >= 0.70 * neutral_score:
            p_text[neutral_idx] *= 0.45
            p_text = p_text / np.sum(p_text)
            
        return p_text
