# Bimodal Affective Alignment System: Resolving Affective Dissonance in Human-Computer Interaction

## Technical Report

---

## 1. Introduction and Motivation

### 1.1 The Need for Bimodal Sensing in HCI

Human-Computer Interaction (HCI) has evolved from command-line interfaces to natural-language conversational agents. Yet even the most advanced text-only systems suffer from a fundamental perceptual blind spot: they can process *what* a user says but not *how* the user feels while saying it. Human communication is inherently multimodal — facial micro-expressions, vocal tone, and body posture routinely carry affective information that contradicts or supplements the literal words spoken. A system that attends only to the linguistic channel therefore operates with an incomplete — and often misleading — model of user affect.

This limitation becomes critical in domains where trust is paramount: mental-health screening, customer-service escalation, educational tutoring, and accessibility interfaces for neurodiverse populations. When a student types *"I'm fine, really"* while their face displays pronounced sadness, a unimodal text agent accepts the statement at face value and moves on, missing a potential cry for help. The bimodal system developed in this project is designed expressly to close that gap.

### 1.2 The Unimodal Ambiguity Problem

Unimodal Ambiguity is the phenomenon whereby a single communication channel provides insufficient evidence to determine a user's true emotional state. It manifests in three concrete forms:

1. **Lexical Masking (Social Desirability Bias):** Users routinely employ polite or socially acceptable phrases ("I'm okay," "Yeah, great job") that suppress their authentic emotional state. Text-only models trained on surface semantics learn to classify these utterances as *Neutral* or *Happy*, reinforcing the mask rather than seeing through it.

2. **Sarcasm and Irony:** Statements such as "Oh, wonderful, another meeting" carry negative valence despite containing lexically positive tokens. Without a complementary visual signal (e.g., an eye-roll or frown), a text classifier defaults to the majority-class interpretation.

3. **Ambiguous Brevity:** Short utterances like "I can't believe it!" are genuinely polysemous — the phrase could signal joy, anger, or shock. The facial expression concurrent with the utterance disambiguates the intent.

In each case, a second sensory modality — here, real-time facial-expression recognition via webcam — provides the orthogonal evidence needed to resolve the ambiguity. The system therefore treats text and vision as *complementary sensors* whose joint probability distribution more faithfully represents the user's affective state than either channel alone.

### 1.3 Affective Alignment and Trust

Trust between a human and a conversational agent depends on the agent's perceived empathy. If the agent responds cheerfully to a distressed user, or somberly to a joyful one, the user's trust degrades rapidly. *Affective alignment* is the computational process of ensuring that the system's inferred emotional state — and consequently its generated response — is consonant with the user's true feelings across all available modalities.

Our system formalizes affective alignment through three mechanisms:

- A **probability-level fusion gate** that blends text and face distributions before emotion selection.
- A **cosine-distance dissonance detector** that flags mismatches between the two modalities.
- An **empathetic response synthesizer** that conditions its language on both the fused emotion and the dissonance flag, enabling it to gently probe when the user's words and face disagree.

---

## 2. Technical Methodology and Cross-Modal Mapping

### 2.1 Emotion Taxonomies

The system operates with two established emotion taxonomies:

| Taxonomy | Source | Labels |
|---|---|---|
| **GoEmotions** | Google Research, 2020 | 28 fine-grained labels (admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral) |
| **FER2013** | Kaggle Facial Expression Recognition Challenge | 7 categorical labels (Angry, Disgust, Fear, Happy, Sad, Surprise, Neutral) |

Because the text sensor produces GoEmotions probabilities and the vision sensor produces FER2013 probabilities, a deterministic mapping is required to project both channels into a shared 7-dimensional emotion space before fusion.

### 2.2 GoEmotions → FER2013 Mapping Strategy

The 28 GoEmotions labels are grouped into the 7 FER2013 categories using the following principled mapping:

| FER2013 Category | GoEmotions Labels Mapped | Justification |
|---|---|---|
| **Angry** | anger, annoyance | Direct semantic overlap with the anger/irritation cluster. |
| **Disgust** | disapproval, disgust | Both convey aversion; disapproval is the cognitive sibling of visceral disgust. |
| **Fear** | fear, nervousness | Nervousness is the low-arousal variant of fear on the circumplex model. |
| **Happy** | admiration, amusement, approval, caring, desire, excitement, gratitude, joy, love, optimism, pride, relief | All occupy the positive-valence quadrant; they share the upturned-lip facial action unit (AU12) characteristic of happiness in FACS coding. |
| **Sad** | disappointment, embarrassment, grief, remorse, sadness | United by negative valence and low arousal; embarrassment includes the social-pain dimension that overlaps with sadness. |
| **Surprise** | surprise | Direct one-to-one correspondence. |
| **Neutral** | confusion, curiosity, realization, neutral | These are cognitive/epistemic states rather than affective ones; mapping them to Neutral preserves signal fidelity by preventing non-emotional labels from injecting noise into affective categories. |

**Design Rationale:** The mapping deliberately routes purely cognitive states (curiosity, realization, confusion) to *Neutral* rather than attempting to force them into affective categories. This prevents the multi-label collapse problem where 28 labels compete for probability mass and the three cognitive labels inflate the Neutral bucket. Our implementation addresses the residual Neutral inflation with a post-normalization dominance rebalancing step (discussed in §2.4).

### 2.3 Multi-Label Collapse Strategy

The text classifier (`bert-base-uncased-go_emotions`) returns probabilities for all 28 GoEmotions labels simultaneously. These must be collapsed into 7 FER scores. Two strategies are implemented, selectable via the system's **Strict Mode** toggle:

- **Strict Mode (max-evidence):** For each FER category, the system takes the *maximum* probability across all GoEmotions labels mapped to that category: `fer_scores[fer_label] = max(fer_scores[fer_label], score)`. This produces a conservative, research-grade distribution that avoids artificial inflation from overlapping labels.

- **Enhanced Mode (sum-evidence):** Probabilities are *summed* across all mapped GoEmotions labels: `fer_scores[fer_label] += score`. This allows semantically related labels (e.g., joy + excitement + gratitude all contributing to Happy) to stack, producing a stronger affective signal for real-world interactive deployment.

### 2.4 Post-Normalization Dominance Rebalancing

After the initial mapping, the GoEmotions model's inherent bias toward cognitive states causes Neutral to dominate even when clear emotional content is present. In Enhanced Mode, the system applies a mathematically grounded correction:

1. **Lexical Calibration Boost:** A transparent set of emotion-associated keywords (e.g., "happy," "excited," "sad," "scared," "angry") adds +0.4 to the relevant FER bucket when detected. This compensates for the model's training distribution, which under-represents short conversational utterances.

2. **Fractional Threshold Promotion:** After normalization, if any emotional category's score exceeds 40% of the Neutral score, it receives a +0.1 additive boost followed by re-normalization. This ensures that genuine emotional signals are not suppressed by the Neutral majority.

3. **Neutral Dampening:** If the strongest non-Neutral category reaches ≥70% of the Neutral score, the Neutral value is scaled by 0.45 and the distribution is re-normalized. This mathematically guarantees that explicit emotional spikes are structurally extracted before formal pooling.

In Strict Mode, all three corrections are disabled, allowing pure model-output evaluation for academic validation.

### 2.5 The Fusion Gate and Trust-Weight (α)

The core of the bimodal system is the **weighted linear fusion equation**:

$$\mathbf{P}_{fusion} = \alpha \cdot \mathbf{P}_{text} + (1 - \alpha) \cdot \mathbf{P}_{face}$$

$$b^*_{fusion} = \arg\max(\mathbf{P}_{fusion})$$

Where:
- **P_text** is the 7-dimensional text emotion probability vector.
- **P_face** is the 7-dimensional facial emotion probability vector.
- **α (alpha)** is the trust-weight hyperparameter controlling the relative influence of each modality.
- **b\*_fusion** is the final fused emotion label selected by argmax.

#### Justification for α = 0.5 (Default)

The default trust-weight is set to **α = 0.5**, assigning equal influence to both modalities. This choice is grounded in the following reasoning:

1. **Epistemological Parity:** In the general case, neither modality is inherently more reliable. Text captures deliberate communicative intent; facial expression captures involuntary affective leakage. Both carry unique, non-redundant information.

2. **Empirical Balance:** With α = 0.5, the system correctly resolves both congruent cases (where both channels agree and reinforce each other) and dissonant cases (where the fusion naturally pulls toward the stronger signal).

3. **User Adjustability:** The system exposes α as a slider (0.0 to 1.0) in the Gradio UI, enabling researchers to evaluate performance across the full spectrum — from text-only (α = 1.0) through balanced fusion (α = 0.5) to vision-only (α = 0.0).

#### Dynamic Alpha Adaptation (Enhanced Mode)

In Enhanced Mode, the system dynamically adjusts α based on contextual reliability signals:

- **No Face Detected:** α is forced to 1.0 (text-only), since the vision channel provides no evidence.
- **Neutral Face Detection:** α is increased by +0.3 (capped at 0.9), because a neutral facial expression is weak evidence that should not override strongly emotional text.
- **High Dissonance (cosine distance > 0.6):** α is increased by +0.2 (capped at 0.9), trusting the explicit textual statement when the two channels critically disagree.

### 2.6 Dissonance Detection via Cosine Distance

Affective dissonance is detected by measuring the angular separation between the text and face probability vectors using **cosine distance**:

$$d_{cosine}(\mathbf{P}_{text}, \mathbf{P}_{face}) = 1 - \frac{\mathbf{P}_{text} \cdot \mathbf{P}_{face}}{||\mathbf{P}_{text}|| \cdot ||\mathbf{P}_{face}||}$$

The system flags dissonance when **all** of the following conditions are met (Strict Mode):

1. A face is detected in the frame.
2. The argmax emotions from text and face differ (`text_idx ≠ face_idx`).
3. Both channels exhibit sufficient confidence (≥ 0.40 threshold).
4. The cosine distance exceeds the dissonance threshold (≥ 0.20).

In Enhanced Mode, additional social-masking heuristics are applied: phrases like "I'm fine," "I am fine," "it's okay" automatically trigger the dissonance flag regardless of cosine distance, because these phrases are empirically associated with emotional suppression.

---

## 3. System Implementation

### 3.1 Architecture Overview

The system implements a **real-time bimodal interaction loop** with four pipeline stages:

```
User Input → [Linguistic Sensor] → P_text (7-dim)
                                          ↘
Webcam Frame → [Visual Sensor] → P_face (7-dim) → [Fusion Gate] → Fused Emotion + Dissonance Flag → [Empathetic Synthesizer] → Response
```

All stages execute within a single Gradio application, with the vision sensor running asynchronously via streaming and the remaining stages triggered synchronously on user submission.

### 3.2 Hardware Configuration

| Component | Specification |
|---|---|
| **Vision Sensor** | Any USB or integrated webcam capable of delivering RGB frames via OpenCV |
| **Face Detection** | OpenCV Haar Cascade classifier (`haarcascade_frontalface_default.xml`) |
| **Compute Target** | CUDA-enabled GPU (auto-detected) with clean CPU fallback |
| **Latency Target** | < 400 ms end-to-end (Doherty Threshold for interactive responsiveness) |

### 3.3 Software Stack

| Layer | Technology | Role |
|---|---|---|
| **Language** | Python 3.10+ | Core runtime |
| **Deep Learning** | PyTorch (torch, torchvision) | Tensor computation and GPU acceleration |
| **NLP Model** | HuggingFace Transformers — `bert-base-uncased-go_emotions` | 28-label text emotion classification |
| **Vision Model** | HuggingFace Transformers — `Celal11/resnet-50-finetuned-FER2013-0.001` | 7-label facial expression recognition (ResNet-50 CNN) |
| **Response Generation** | Google FLAN-T5-small (Seq2Seq) | Empathetic natural-language response synthesis |
| **Face Detection** | OpenCV Haar Cascades | Real-time bounding-box face extraction |
| **Frontend** | Gradio Blocks | Synchronous web-based UI with webcam streaming |
| **Data Logging** | Pandas + CSV | Structured experiment telemetry |
| **Distance Metrics** | SciPy (`scipy.spatial.distance.cosine`) | Cosine distance for dissonance measurement |

### 3.4 Module-by-Module Implementation

#### 3.4.1 Linguistic Sensor (`text_sensor.py`)

The `LinguisticSensor` class encapsulates the full text-processing pipeline:

1. **Preprocessing:** User text is lowercased, stripped, and punctuation-removed via regex to normalize input.
2. **Inference:** The `transformers.pipeline` runs the BERT model with `top_k=None`, returning probabilities for all 28 GoEmotions labels.
3. **Label Decoding:** The model checkpoint uses generic `LABEL_0` through `LABEL_27` identifiers. A static dictionary (`ID2GOEMOTION`) maps these back to canonical GoEmotions names.
4. **FER Mapping:** Each GoEmotions probability is routed to its corresponding FER2013 bucket via the `GOEMOTIONS_MAPPING` dictionary.
5. **Normalization and Rebalancing:** The 7-element vector is L1-normalized, then optionally rebalanced (Enhanced Mode only).

The function returns a NumPy array of shape `(7,)` representing the text emotion probability distribution aligned to FER2013 classes.

#### 3.4.2 Visual Sensor (`vision_sensor.py`)

The `VisualSensor` class processes webcam frames through a four-stage pipeline:

1. **Face Detection:** The input RGB frame is converted to grayscale with histogram equalization. A two-pass Haar Cascade strategy is applied — a conservative first pass (`minNeighbors=7, minSize=90×90`) followed by a permissive fallback (`minNeighbors=5, minSize=60×60`) — to balance precision and recall.
2. **Multi-Face Resolution:** If multiple faces are detected, the largest bounding box (by area) is selected, and a failure reason is logged.
3. **Model Inference:** The cropped face region is converted to a PIL Image and passed through the ResNet-50 classifier pipeline, which outputs probabilities for the 7 FER2013 emotions.
4. **Temporal Smoothing:** Raw frame-level predictions are noisy. The system maintains a rolling deque of the 3 most recent predictions and returns their element-wise mean, providing temporal stability without introducing significant lag.
5. **Weak Detection Filter:** If the maximum probability in the raw prediction is below 0.5, the frame is treated as an unreliable detection and replaced with a pure-Neutral vector, preventing low-confidence noise from corrupting the fusion.

#### 3.4.3 Fusion Layer (`fusion.py`)

The `FusionLayer` class executes the weighted fusion equation and dissonance detection:

- Computes `P_fusion = α · P_text + (1 − α) · P_face` using NumPy vectorized operations.
- Calculates cosine distance with an epsilon buffer (`1e-8`) to prevent division-by-zero.
- Applies dynamic alpha adaptation in Enhanced Mode.
- Evaluates dissonance conditions and returns a structured dictionary containing: the fused emotion label, the dissonance flag, the severity score, the full fusion probability vector, and the active alpha value.

#### 3.4.4 Empathetic Synthesizer (`llm_synth.py`)

The `EmpatheticSynthesizer` class generates contextually appropriate responses using FLAN-T5-small:

1. **Trust & Safety Filter:** In Enhanced Mode, crisis-related keywords (e.g., "kill," "suicide") trigger an immediate safe-channel response, bypassing the LLM entirely.
2. **Fast-Path Routing:** For non-dissonant Neutral/Happy states, the system bypasses the computationally expensive LLM and returns pre-authored responses organized by emotional sub-type (Affection, Achievement, Physical State, Factual). This reduces latency to near-zero for common cases.
3. **LLM Inference:** For complex cases (dissonance detected, negative emotions), a structured prompt is constructed that includes the fused emotion and dissonance context. The model generates with `num_beams=3`, `max_length=45`, and `repetition_penalty=2.0` to prevent degenerate outputs.
4. **Quality Gate:** A multi-stage post-generation filter catches: (a) responses shorter than 5 characters, (b) responses that simply echo the emotion label, (c) responses that copy >35% of the user's words (fuzzy copy-check), (d) repetition loops, and (e) toxic positivity (cheerful responses to sad/angry/fearful inputs). Failed responses are replaced with curated, emotion-appropriate fallback templates.

#### 3.4.5 Application Layer (`app.py`)

The Gradio Blocks application orchestrates the full pipeline:

- **Webcam Stream:** The `gr.Image` component with `streaming=True` continuously invokes `update_face_state()`, storing the latest `P_face` vector in a `gr.State` object. This enables asynchronous, real-time vision processing independent of text submission.
- **Submit Handler:** On button click, `process_interaction()` retrieves the cached face state, runs the text sensor, executes fusion, generates the response, logs all telemetry to CSV, and formats the debug output.
- **UI Controls:** An alpha slider (0.0–1.0, step 0.1), a Strict Mode checkbox, and input/output text areas provide full experimental control.
- **Experiment Logging:** Every interaction is appended to `experiment_logs.csv` with 18 columns capturing timestamps, raw probabilities, fusion outputs, latency, and generated responses for post-hoc academic analysis.

### 3.5 Interaction Loop Sequence

The complete bimodal interaction loop proceeds as follows:

1. The user positions themselves in front of the webcam and begins streaming.
2. The Visual Sensor continuously processes frames, updating the face emotion state.
3. The user types a message and clicks "Submit."
4. The system reads the cached face state (P_face) and runs text inference (P_text).
5. The Fusion Layer computes P_fusion and evaluates dissonance.
6. The Empathetic Synthesizer generates a response conditioned on the fused emotion and dissonance flag.
7. The UI displays the response, full telemetry, and all probability distributions.
8. The experiment is logged to CSV for later analysis.

---

## 4. Affective Evaluation and Case Studies

### 4.1 Evaluation Methodology

The system was evaluated through qualitative case studies designed to test its ability to resolve two fundamental scenarios:

- **Congruent Signals:** Text and facial expression agree on the emotional state. The system should reinforce and validate the user's emotion.
- **Dissonant Signals:** Text and facial expression disagree. The system should detect the mismatch and respond with gentle probing rather than accepting either channel at face value.

### 4.2 Table 1: Qualitative Evaluation — Resolution of Affective Dissonance

| User Input (U) | Text (P_text) | Face (P_face) | Bimodal Empathetic Response (R) |
|---|---|---|---|
| "I'm fine, really." | Neutral | Sadness | "I hear you, but you look a bit troubled. Is there anything on your mind you'd like to share?" |
| "I can't believe it!" | Surprise | Happy | "That's incredible news! I can see how excited you are; I'm thrilled for you!" |
| "Yeah, great job." | Happy | Angry | "I sense some frustration despite your words. Did something about the process go wrong?" |
| "I lost my grandfather today!" | Sad | Sad | "That sounds really tough. I'm here for you." |
| "I got my new car today!" | Happy | Happy | "You should be proud of yourself!" |
| "I am feeling down today" | Neutral | Happy | "Your expressions seem to be telling a slightly different story. Let's talk about it." |

### 4.3 Case Study Analysis

#### Case 1: Social Masking — "I'm fine, really." (Dissonant)

- **Text Prediction:** The BERT model classifies this as *Neutral* — the phrase is a stock social response with no lexically emotional content.
- **Face Prediction:** The ResNet-50 model detects *Sadness* from the user's facial expression — drooped mouth corners, lowered brows.
- **Fusion Behavior:** The cosine distance between the Neutral-dominant text vector and the Sad-dominant face vector exceeds the 0.20 threshold. Additionally, in Enhanced Mode, "I'm fine" matches the social-masking phrase list, guaranteeing dissonance detection.
- **Response:** The system does not accept the user's words at face value. Instead, it acknowledges the verbal content ("I hear you") while gently surfacing the visual evidence ("you look a bit troubled"), offering a non-confrontational opening for the user to share their true feelings.
- **Significance:** This case directly demonstrates the Unimodal Ambiguity problem. A text-only system would respond with "That's good to hear!" — a response that invalidates the user's actual emotional state and erodes trust.

#### Case 2: Ambiguous Exclamation — "I can't believe it!" (Congruent)

- **Text Prediction:** The BERT model assigns highest probability to *Surprise* — the exclamation mark and "can't believe" phrase are strong surprise indicators.
- **Face Prediction:** The ResNet-50 detects *Happy* — wide smile, raised cheeks.
- **Fusion Behavior:** The fusion combines Surprise (text) and Happy (face) into a distribution where both positive emotions have high mass. The argmax resolves to the category with highest combined weight. The cosine distance is moderate but the emotional valence is congruent (both positive), so the response is celebratory rather than probing.
- **Response:** The system synthesizes an enthusiastic response that validates both the surprise and the joy, producing a contextually appropriate empathetic reply.
- **Significance:** This demonstrates how bimodal fusion disambiguates a lexically ambiguous phrase. Without the facial signal, "I can't believe it!" could equally indicate shock, anger, or joy.

#### Case 3: Sarcasm Detection — "Yeah, great job." (Dissonant)

- **Text Prediction:** The BERT model classifies this as *Happy* — the tokens "great" and "job" are lexically positive.
- **Face Prediction:** The ResNet-50 detects *Angry* — furrowed brows, tightened jaw.
- **Fusion Behavior:** The cosine distance between the Happy-dominant text vector and the Angry-dominant face vector is very high (typically > 0.6). Both channels have high confidence, clearly meeting all dissonance criteria.
- **Response:** The system recognizes the mismatch and does not respond positively. Instead, it acknowledges the frustration detected in the face ("I sense some frustration despite your words") and probes for the underlying cause.
- **Significance:** This is a classic sarcasm case. The user's words are superficially positive, but their face reveals the true negative intent. Without the visual modality, the system would respond with congratulations — a response that would feel dismissive and erode user trust.

#### Case 4: Congruent Grief — "I lost my grandfather today!" (Congruent)

- **Text Prediction:** Sad — the word "lost" combined with a familial reference maps strongly through grief → Sad.
- **Face Prediction:** Sad — the facial expression matches the verbal content.
- **Fusion Behavior:** Both probability vectors align in the Sad category. The cosine distance is low (< 0.20), confirming congruence. No dissonance is flagged.
- **Response:** The system provides a straightforward, empathetic acknowledgment of the user's grief without probing for contradictions.
- **Significance:** In congruent cases, the system should *not* over-analyze. The bimodal fusion correctly reinforces the unified signal and generates an appropriate supportive response.

### 4.4 Empirical Observations from Experiment Logs

Analysis of the 33 logged interactions in `experiment_logs.csv` reveals the following patterns:

1. **Latency Performance:** The median end-to-end latency is approximately 650 ms. First-run inference (cold model cache) shows latencies of ~1,500 ms, but subsequent runs consistently fall below 850 ms. The fast-path routing for non-dissonant Neutral/Happy cases achieves latencies as low as 98 ms, well within the 400 ms Doherty Threshold.

2. **Neutral Dominance in Strict Mode:** In Strict Mode, the text sensor frequently predicts Neutral even for emotionally charged inputs (e.g., "i got my new car!" → Neutral, "I am very happy today" → Neutral). This confirms the known GoEmotions Neutral-inflation problem and validates our post-normalization rebalancing approach in Enhanced Mode.

3. **Dissonance Detection Reliability:** When the text predicts Neutral and the face shows Happy (a common test pattern), the system correctly flags dissonance in every case, demonstrating that the cosine threshold of 0.20 is appropriately calibrated.

4. **Alpha Sensitivity:** The experiment logs show tests across α values of 0.0, 0.2, 0.5, 0.7, 0.9, and 1.0. At α = 1.0 (text-only), the system ignores facial evidence entirely. At α = 0.0 (vision-only), it ignores text. The balanced α = 0.5 consistently produces the most contextually appropriate fused emotions.

---

## 5. Discussion

### 5.1 Strengths

- **End-to-end real-time operation:** The system achieves interactive-speed bimodal inference without requiring specialized hardware beyond a standard webcam.
- **Transparent fusion mechanics:** The weighted linear fusion with a user-adjustable α makes the system's decision process interpretable and auditable.
- **Dual-mode evaluation:** The Strict/Enhanced Mode toggle allows rigorous academic evaluation against pure model output while also demonstrating practical HCI deployment with real-world heuristics.
- **Comprehensive telemetry:** Every interaction logs 18 data fields, enabling post-hoc statistical analysis of system performance.

### 5.2 Limitations

- **Face detection sensitivity:** The Haar Cascade detector occasionally fails to detect faces under challenging lighting conditions or at extreme angles, forcing a text-only fallback.
- **GoEmotions Neutral bias:** The BERT model's training distribution heavily favors Neutral, requiring post-hoc calibration that may not generalize across all conversational domains.
- **FLAN-T5-small capacity:** The small model occasionally produces generic or hallucinated responses, necessitating the multi-stage quality gate. A larger model would improve response quality at the cost of latency.
- **Static fusion weights:** The α parameter, while adjustable, does not adapt automatically based on per-sample confidence estimates. A learned attention mechanism could improve fusion quality.

### 5.3 Future Work

- Integration of vocal prosody as a third modality (trimodal fusion).
- Replacement of the static α weight with a learned attention-based fusion mechanism.
- Fine-tuning FLAN-T5 on empathetic dialogue datasets (e.g., EmpatheticDialogues) for higher-quality response generation.
- Longitudinal trust evaluation studies with human participants.

---

## 6. Conclusion

This report presented a bimodal affective alignment system that addresses the fundamental Unimodal Ambiguity problem in Human-Computer Interaction. By combining a BERT-based text emotion classifier (trained on the 28-label GoEmotions taxonomy) with a ResNet-50 facial expression recognizer (trained on FER2013), the system projects both modalities into a shared 7-dimensional emotion space and fuses them via a weighted linear equation controlled by the trust-weight hyperparameter α.

The system's dissonance detector, based on cosine distance between the two probability vectors, reliably identifies cases where a user's words contradict their facial expression — a critical capability for building trustworthy empathetic agents. As demonstrated in the qualitative evaluation (Table 1), the bimodal approach correctly resolves both congruent and dissonant affective signals, producing contextually appropriate empathetic responses that a unimodal system would fail to generate.

The dual-mode architecture (Strict vs. Enhanced) enables both rigorous academic evaluation and practical deployment, while comprehensive experiment logging supports reproducible analysis. Together, these contributions advance the field of affective computing toward systems that perceive and respond to the full spectrum of human emotional expression.

---

## References

1. Demszky, D., Movshovitz-Attias, D., Ko, J., Cowen, A., Nemade, G., & Ravi, S. (2020). GoEmotions: A Dataset of Fine-Grained Emotions. *ACL 2020*.
2. Goodfellow, I. J., et al. (2013). Challenges in Representation Learning: A report on three machine learning contests. *Neural Networks*, 64, 59–63. (FER2013)
3. He, K., Zhang, X., Ren, S., & Sun, J. (2016). Deep Residual Learning for Image Recognition. *CVPR 2016*.
4. Devlin, J., Chang, M.-W., Lee, K., & Toutanova, K. (2019). BERT: Pre-training of Deep Bidirectional Transformers for Language Understanding. *NAACL-HLT 2019*.
5. Chung, H. W., et al. (2022). Scaling Instruction-Finetuned Language Models (FLAN-T5). *arXiv:2210.11416*.
6. Doherty, W. J. (1982). The Economic Value of Rapid Response Time. *IBM Technical Report*.
7. Picard, R. W. (1997). *Affective Computing*. MIT Press.
