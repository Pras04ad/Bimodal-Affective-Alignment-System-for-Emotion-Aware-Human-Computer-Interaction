BIMODAL AFFECTIVE ALIGNMENT SYSTEM: RESOLVING AFFECTIVE DISSONANCE IN HCI


1. SYSTEM OVERVIEW

This project implements a real-time, bimodal affective inference framework designed to empirically evaluate and synthesize divergent states of human emotional dissonance. 

The software utilizes a pure dual-sensor architecture:
  - Linguistic Sensor: Processes user text utilizing a `bert-base` model fine-tuned on the GoEmotions taxonomy (nmb-paperspace-hf/bert-base-uncased-go_emotions). The pipeline actively resolves standard multi-label collapse by semantically re-basing pure cognitive states (e.g., curiosity, realization) into the 'Neutral' class, preserving absolute affective signal fidelity when mapping to the 7 standard FER metrics.
  - Visual Sensor: Synchronously intercepts webcam frames, executing real-time bounding-box extraction against a ResNet-50 Convolutional Neural Network deployed on FER2013 (Celal11/resnet-50-finetuned-FER2013-0.001), stabilized by an accelerated temporal smoothing queue.

Fusion is executed strictly via the dynamic weighting equation `b*_fusion = argmax(alpha * P_text + (1 - alpha) * P_face)`. We actively evaluate distance modalities via cosine similarity to detect Affective Dissonance mismatches. The system formally concludes with an LLM Generation wrapper executing responsive dialogue iterations via FLAN-T5-small.

2. REQUIREMENTS & DEPENDENCIES

The project executes on Python 3.10+ environments.

Major Modules Required:
- PyTorch (torch, torchvision)
- Transformers (huggingface inference pipeline)
- OpenCV (cv2, for Haar Cascades webcam frame scraping)
- Gradio (gr, front-end synchronous state streaming)
- NumPy, Pandas, SciPy

Install requirements utilizing:
pip install torch torchvision transformers opencv-python gradio pandas scipy

NOTE ON HARDWARE: The framework dynamically targets CUDA environments to instantiate tensor models across GPUs autonomously, failing cleanly back to CPU processing if hardware limits are detected.

3. RUNNING THE SYSTEM:

1. Ensure your webcam is securely connected and functioning.
2. Navigate terminal to the root project directory.
3. Execute the module:
    python src/app.py
4. Gradio will launch the visual interface. Navigate any browser to: http://127.0.0.1:7860
   (Note: The absolute first launch downloads required BERT / ResNet-50 / T5 weights from HuggingFace to the local cache.)
5. IMPORTANT: To capture real-time affective states, you must hit the "Record" button on the webcam feed first before submitting your text, and then click "Stop" when done capturing your facial feedback.

4. EVALUATION ENVIRONMENT (THE "STRICT" TOGGLE)

The Web UI launches featuring a structural testing evaluation toggle set to 'True' by default:  
[x] Strict Mode

When ACTIVE:
The system mathematically restricts all background heuristics, forcing pure pipeline inference to validate theoretical integration. 
- Operates totally bound to the `argmax` fusion equation.
- Eliminates hardcoded phrase "overrides".
- Post-Normalization Dominance Rebalancing: We natively instituted a mathematical post-normalization rebalancing step. It corrects the inherent conversational Neutral bias in raw GoEmotions deployments using pure fractional threshold testing, guaranteeing explicit emotional spikes are structurally extracted before formal pooling. 

When DISABLED (Applied HCI Demonstration):
The application unlocks semantic/conversational adaptations necessary for real-world interactive deployment. This engages soft semantic bias shifting, conversational masking detection (accounting for users socially typing "I'm okay" while physically displaying intense despair), and rapid inference-latency loop routing to maximize smooth User Experience.

5. EXPERIMENT LOGGING:

The underlying script automatically generates empirical testing telemetry. Every input iteration structurally logs all raw branch probabilities, scaled fusion outputs, total latency margins, and triggered dissonance states linearly to `experiment_logs.csv` situated within the root folder for later formal academic review.
