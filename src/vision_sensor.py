import cv2
import numpy as np
import torch
from transformers import pipeline
from PIL import Image
from collections import deque

FER_CLASSES = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

class VisualSensor:
    def __init__(self):
        # We use a lightweight PyTorch-based FER model from HuggingFace to avoid TensorFlow dependencies
        # 'mrm8488/vit-mer-cv' or 'trpakov/vit-face-expression' are good, but let's use a standard one:
        # 'Celal11/resnet-50-finetuned-FER2013-0.001' (ResNet for FER2013)
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        try:
            self.classifier = pipeline(task="image-classification", model="Celal11/resnet-50-finetuned-FER2013-0.001", device=0 if torch.cuda.is_available() else -1)
        except Exception as e:
            print(f"Warning: Could not load HuggingFace FER model: {e}")
            self.classifier = None
            
        # History queue for temporal smoothing (rolling short-window average)
        self.p_face_history = deque(maxlen=3)
            
        # OpenCV Haar Cascade for face detection
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        
        # Mapping from this specific model's output to FER_CLASSES
        # The model usually outputs: 'sad', 'disgust', 'angry', 'neutral', 'fear', 'surprise', 'happy'
        self.label_mapping = {
            'angry': 'Angry',
            'disgust': 'Disgust',
            'fear': 'Fear',
            'happy': 'Happy',
            'sad': 'Sad',
            'surprise': 'Surprise',
            'neutral': 'Neutral'
        }

    def process(self, frame: np.ndarray) -> tuple[np.ndarray, bool, list]:
        """
        Process a BGR or RGB image frame.
        Returns:
            p_face: 7-element probability array matching FER_CLASSES.
            face_detected: boolean indicating if a face was found.
            failure_reasons: list of strings explaining failures (if any).
        """
        failure_reasons = []
        p_face = np.zeros(7)
        p_face[FER_CLASSES.index('Neutral')] = 1.0
        
        if self.classifier is None:
            failure_reasons.append("FER Model not loaded.")
            return p_face, False, failure_reasons

        try:
            # 1. Face Detection using OpenCV
            # Gradio webcam images are RGB, so convert RGB to GRAY
            gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
            gray = cv2.equalizeHist(gray)

            # First pass is conservative. If it misses a clear webcam face, fall back
            # to a more permissive pass while still requiring a human-scale box.
            faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=7, minSize=(90, 90))
            if len(faces) == 0:
                faces = self.face_cascade.detectMultiScale(gray, scaleFactor=1.05, minNeighbors=5, minSize=(60, 60))
            
            if len(faces) == 0:
                failure_reasons.append("No Face Detected.")
                return p_face, False, failure_reasons
                
            # 2. Handle multiple faces (Fallback rule: largest box)
            if len(faces) > 1:
                failure_reasons.append("Multiple faces detected. Using the largest bounding box.")
                # Sort by area (w * h) descending
                faces = sorted(faces, key=lambda f: f[2] * f[3], reverse=True)
                
            x, y, w, h = faces[0]
            
            # Crop face and convert to PIL Image
            face_crop = frame[y:y+h, x:x+w]
            # Frame is already RGB from Gradio, no need to convert
            pil_img = Image.fromarray(face_crop)
            
            # 3. Model Inference via Pipeline
            outputs = self.classifier(pil_img)
            
            # 4. Map to standardized FER_CLASSES array
            fer_scores = {c: 0.0 for c in FER_CLASSES}
            for out in outputs:
                label_str = out['label'].lower()
                mapped_label = self.label_mapping.get(label_str, 'Neutral')
                # Add score (pipelines output probabilities natively)
                fer_scores[mapped_label] += out['score']
                
            # Normalize
            total = sum(fer_scores.values())
            if total > 0:
                for k in fer_scores:
                    fer_scores[k] /= total
                    
            p_face_raw = np.array([fer_scores[c] for c in FER_CLASSES])
            
            # Reduce false face errors (ignore weak detections)
            if np.max(p_face_raw) < 0.5:
                p_face_raw = np.zeros(7)
                p_face_raw[FER_CLASSES.index('Neutral')] = 1.0
                
            # temporal smoothing
            self.p_face_history.append(p_face_raw)
            p_face_smoothed = np.mean(self.p_face_history, axis=0)
            
            return p_face_smoothed, True, failure_reasons
            
        except Exception as e:
            failure_reasons.append(f"Unexpected error: {str(e)}")
            return p_face, False, failure_reasons
