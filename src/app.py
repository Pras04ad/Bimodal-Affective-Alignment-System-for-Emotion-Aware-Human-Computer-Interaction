import gradio as gr
import time
import pandas as pd
import os
import numpy as np
import cv2
import json

from text_sensor import LinguisticSensor, FER_CLASSES
from vision_sensor import VisualSensor
from fusion import FusionLayer
from llm_synth import EmpatheticSynthesizer

# Global instances
print("Loading Models... This will take a moment.")
text_sensor = LinguisticSensor()
vision_sensor = VisualSensor()
synthesizer = EmpatheticSynthesizer()
print("Models Loaded.")

LOG_FILE = "experiment_logs.csv"
LOG_COLUMNS = [
    "Timestamp", "User_Text", "P_text_argmax", "P_face_argmax", "Face_Detected",
    "Failure_Reasons", "Active_Alpha", "Text_Confidence", "Face_Confidence",
    "Fused_Confidence", "P_text", "P_face", "P_fusion",
    "Fused_Emotion", "Dissonance_Flag", "Dissonance_Severity", "Latency_ms",
    "Generated_Response"
]

def ensure_log_file():
    if not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0:
        pd.DataFrame(columns=LOG_COLUMNS).to_csv(LOG_FILE, index=False)
        return

    existing = pd.read_csv(LOG_FILE)
    changed = False
    for column in LOG_COLUMNS:
        if column not in existing.columns:
            existing[column] = ""
            changed = True

    if list(existing.columns) != LOG_COLUMNS:
        existing = existing[LOG_COLUMNS]
        changed = True

    if changed:
        existing.to_csv(LOG_FILE, index=False)

ensure_log_file()

def log_experiment(data: dict):
    df = pd.DataFrame([data])
    write_header = not os.path.exists(LOG_FILE) or os.path.getsize(LOG_FILE) == 0
    df.to_csv(LOG_FILE, mode='a', header=write_header, index=False)

def top_emotion(probabilities: np.ndarray) -> tuple[str, float]:
    idx = int(np.argmax(probabilities))
    return FER_CLASSES[idx], float(probabilities[idx])

def format_distribution(probabilities: np.ndarray) -> str:
    return ", ".join(
        f"{emotion}:{prob:.2f}"
        for emotion, prob in zip(FER_CLASSES, probabilities)
    )

def format_response_mode(override_status):
    if not override_status:
        return "LLM Generation"
    return override_status.replace("(", "").replace(")", "")

def update_face_state(image):
    if image is not None:
        p_face, face_detected, failures = vision_sensor.process(image)
        return {'p_face': p_face, 'face_detected': face_detected, 'failures': failures}
    return {'p_face': np.zeros(7), 'face_detected': False, 'failures': ["No image."]}

def process_interaction(face_state, text, alpha, strict_mode):
    start_time = time.time()
    
    # 1. Text Inference (Apply strict mode)
    p_text = text_sensor.process(text, strict_mode)
    
    # 2. Vision Inference (Already processed in parallel stream, read from State)
    p_face = face_state['p_face']
    face_detected = face_state['face_detected']
    failures = face_state['failures']
        
    # 3. Fusion Logic (Apply strict mode)
    fusion_layer = FusionLayer(alpha=alpha)
    fusion_results = fusion_layer.fuse(p_text, p_face, face_detected, user_text=text, strict_mode=strict_mode)
    
    # 4. Synthesizer
    response, override_status = synthesizer.generate_response(text, fusion_results['fused_emotion'], fusion_results['dissonance_flag'], strict_mode)
    
    latency_ms = (time.time() - start_time) * 1000.0
    text_emotion, text_conf = top_emotion(p_text)
    if face_detected:
        face_emotion, face_conf = top_emotion(p_face)
    else:
        face_emotion, face_conf = "N/A", 0.0
    fusion_probs = np.array(fusion_results['p_fusion'])
    fused_emotion, fused_conf = top_emotion(fusion_probs)
    
    # Logging
    log_data = {
        "Timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "User_Text": text,
        "P_text_argmax": text_emotion,
        "P_face_argmax": face_emotion if face_detected else "None",
        "Face_Detected": face_detected,
        "Failure_Reasons": " | ".join(failures),
        "Active_Alpha": fusion_results['active_alpha'],
        "Text_Confidence": text_conf,
        "Face_Confidence": face_conf,
        "Fused_Confidence": fused_conf,
        "P_text": json.dumps(p_text.tolist()),
        "P_face": json.dumps(p_face.tolist()),
        "P_fusion": json.dumps(fusion_results['p_fusion']),
        "Fused_Emotion": fusion_results['fused_emotion'],
        "Dissonance_Flag": fusion_results['dissonance_flag'],
        "Dissonance_Severity": fusion_results['dissonance_severity'],
        "Latency_ms": latency_ms,
        "Generated_Response": response
    }
    log_experiment(log_data)
    
    # Format output for UI
    dissonance_str = "YES (Possible affective mismatch)" if fusion_results['dissonance_flag'] else "NO (Congruent)"
    
    response_mode = format_response_mode(override_status)
    
    debug_info = f"""
Latency (Target < 400ms): {latency_ms:.1f} ms
Face Detected: {face_detected}
Failure Reasons: {failures}
Active Alpha: {fusion_results['active_alpha']:.2f}

Text Emotion (Argmax): {text_emotion} ({text_conf:.2f})
Face Emotion (Argmax): {face_emotion} ({face_conf:.2f}){'' if face_detected else ' - no valid face frame'}
Fused Ground Truth: {fusion_results['fused_emotion']} ({fused_conf:.2f})
Response Mode: {response_mode}

Dissonance Detected: {dissonance_str}
Dissonance Severity (Cosine Dist): {fusion_results['dissonance_severity']:.3f}

P_text: {format_distribution(p_text)}
P_face: {format_distribution(p_face)}
P_fusion: {format_distribution(fusion_probs)}
"""
    
    return debug_info.strip(), response

# Gradio UI Layout
with gr.Blocks(title="Bimodal Affective Alignment") as demo:
    gr.Markdown("# Bimodal Affective Alignment Framework")
    gr.Markdown("*Resolving Affective Dissonance in Human-Computer Interaction*")
    
    with gr.Row():
        with gr.Column(scale=1):
            cam_input = gr.Image(sources=["webcam"], streaming=True, label="Webcam Feed (Vision Sensor)")
            text_input = gr.Textbox(lines=2, placeholder="Type your message here...", label="User Speech (Linguistic Sensor)")
            alpha_slider = gr.Slider(minimum=0.0, maximum=1.0, value=0.5, step=0.1, label="Trust-Weight Hyperparameter (alpha)")
            strict_mode_toggle = gr.Checkbox(label="Strict Mode (Disable enhancements)", value=True)
            gr.Markdown("`alpha = 1.0` (Text-Only) | `alpha = 0.5` (Fused) | `alpha = 0.0` (Vision-Only)")
            submit_btn = gr.Button("Submit Message", variant="primary")
            
        with gr.Column(scale=1):
            response_output = gr.Textbox(label="Agent Response (Empathetic Synthesizer)", lines=5)
            debug_output = gr.Textbox(label="System Telemetry & Metrics", lines=10)
            
    # Asynchronous real-time vision sensor updating
    current_face_state = gr.State({'p_face': np.zeros(7), 'face_detected': False, 'failures': ["No image yet."]})
    cam_input.stream(fn=update_face_state, inputs=[cam_input], outputs=[current_face_state])
            
    submit_btn.click(
        fn=process_interaction,
        inputs=[current_face_state, text_input, alpha_slider, strict_mode_toggle],
        outputs=[debug_output, response_output]
    )
    
    gr.Markdown("**Privacy Policy**: *The system processes facial expressions locally in real-time and does not record or store video data.*")

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
