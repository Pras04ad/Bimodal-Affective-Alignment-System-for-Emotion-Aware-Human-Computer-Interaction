import numpy as np
from scipy.spatial.distance import cosine

FER_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
DISSONANCE_COSINE_THRESHOLD = 0.20
DISSONANCE_CONFIDENCE_THRESHOLD = 0.40

class FusionLayer:
    def __init__(self, alpha: float = 0.5):
        self.alpha = float(alpha)
        
    def fuse(self, p_text: np.ndarray, p_face: np.ndarray, face_detected: bool, user_text: str = "", strict_mode: bool = False):
        # 0 means identical distributions, ~1-2 means very different.
        severity_score = 0.0
        if face_detected:
            # Add small epsilon to avoid division by zero
            severity_score = float(cosine(p_text + 1e-8, p_face + 1e-8))

        # Fallback to Text-Only if no face was detected
        active_alpha = self.alpha
        if not face_detected:
            active_alpha = 1.0
        elif not strict_mode:
            face_emotion_pred = FER_CLASSES[np.argmax(p_face)]
            if face_emotion_pred == 'Neutral':
                # Neutral face is weak evidence; do not let it override strongly emotional text
                active_alpha = min(0.9, active_alpha + 0.3)
            elif severity_score > 0.6:
                # If they critically disagree, rely heavily on the explicit text
                active_alpha = min(0.9, active_alpha + 0.2)
                
        # Fusion equation: b*_fusion = argmax(α * P_text + (1 - α) * P_face)
        p_fusion = active_alpha * p_text + (1.0 - active_alpha) * p_face
        fused_idx = np.argmax(p_fusion)
        fused_emotion = FER_CLASSES[fused_idx]
        
        # Dissonance detection
        text_idx = np.argmax(p_text)
        face_idx = np.argmax(p_face)
        text_conf = float(np.max(p_text))
        face_conf = float(np.max(p_face))
        
        face_emotion_pred = FER_CLASSES[face_idx]
        text_emotion_pred = FER_CLASSES[text_idx]
        
        dissonance_flag = False

        if strict_mode:
            # Strict mode keeps fusion unchanged and uses the cosine-distance
            # dissonance heuristic described in the report/slides.
            dissonance_flag = bool(
                face_detected
                and text_idx != face_idx
                and text_conf >= DISSONANCE_CONFIDENCE_THRESHOLD
                and face_conf >= DISSONANCE_CONFIDENCE_THRESHOLD
                and severity_score >= DISSONANCE_COSINE_THRESHOLD
            )
        elif face_detected and text_idx != face_idx:
            # Enhanced mode avoids overreacting to tiny probability differences.
            if face_emotion_pred != 'Neutral' or text_emotion_pred != 'Neutral':
                if severity_score >= DISSONANCE_COSINE_THRESHOLD:
                    dissonance_flag = True
        
        if not strict_mode:
            # Even if text model predicted Neutral, these idioms indicate explicit social masking
            masking_phrases = ["i'm fine", "i am fine", "it's okay", "its okay", "im fine"]
            if any(phrase in user_text.lower() for phrase in masking_phrases):
                dissonance_flag = True
            
        return {
            "fused_emotion": fused_emotion,
            "dissonance_flag": dissonance_flag,
            "dissonance_severity": severity_score,
            "p_fusion": p_fusion.tolist(),
            "active_alpha": active_alpha
        }
