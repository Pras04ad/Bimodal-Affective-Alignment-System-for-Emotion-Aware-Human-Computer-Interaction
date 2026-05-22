from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
import re
import random

class EmpatheticSynthesizer:
    def __init__(self):
        # FLAN-T5 small is drastically faster for approaching the <400ms Doherty Threshold
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-small")
        self.model = AutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-small").to(self.device)
        
    def generate_response(self, user_text: str, fused_emotion: str, dissonance_flag: bool, strict_mode: bool = False):
        if not user_text.strip():
            return "", None
            
        user_text_lower = user_text.lower()
        if not strict_mode:
            # Basic Trust & Safety Filter for crisis/hostile inputs
            toxic_keywords = ['kill', 'suicide', 'die', 'murder', 'hurt', 'beat', 'attack', 'hit', 'punch', 'harm', 'violent', 'hunt', 'bury', 'burry', 'weapon', 'shoot', 'stab']
            safe_contexts = ['game', 'play', 'movie', 'book', 'video']
            
            if any(word in user_text_lower for word in toxic_keywords):
                if not any(ctx in user_text_lower for ctx in safe_contexts):
                    return "It sounds like you're really upset right now. Let's slow down for a moment—what's making you feel this way?", "Crisis / Threat Detected"
                    
            # Latency Fast-Path: Bypass slow LLM inference for basic emotional states
            if not dissonance_flag and fused_emotion in ['Neutral', 'Happy']:
                norm_text = user_text_lower
                
                # Sub-type routing for Happy
                if fused_emotion == 'Happy':
                    if any(w in norm_text for w in ["kiss", "love", "girlfriend", "boyfriend", "partner"]):
                        return random.choice([
                            "That's so sweet! I'm happy for you.",
                            "It sounds like you really care about them."
                        ]), "Fast-Path (Affection)"
                    elif any(w in norm_text for w in ["achieved", "got", "completed", "done", "won", "success"]):
                        return random.choice([
                            "Congratulations! That's a great achievement.",
                            "You should be proud of yourself!"
                        ]), "Fast-Path (Achievement)"
                    else:
                        return random.choice([
                            "That's great to hear! Keep enjoying the moment.",
                            "That sounds wonderful."
                        ]), "Fast-Path (Happy/General)"
                        
                # Sub-type routing for Neutral
                else:
                    if any(w in norm_text for w in ["sleep", "tired", "exhausted", "sick"]):
                        return random.choice([
                            "Make sure you get some rest. Take care of yourself.",
                            "Take it easy today. You deserve a break."
                        ]), "Fast-Path (Physical State)"
                    else:
                        return random.choice([
                            "I understand. Thanks for sharing.",
                            "That makes sense. Tell me more.",
                            "Got it. I'm listening."
                        ]), "Fast-Path (Neutral/Factual)"
            
        system_instruction = (
            "You are an empathetic assistant. "
            "Write a short conversational reply in 1 to 2 sentences. "
            f"The detected emotional state is '{fused_emotion}'. "
            "Respond only to the emotional state and the user's exact words. "
            "Do not invent facts, people, events, causes, or outcomes. "
            "Do not add details not present in the user input. "
        )
        
        if dissonance_flag:
            prompt_instruction = system_instruction + f"Dissonance exists: acknowledge gently that they seem to feel differently than what they say.\nUser: '{user_text}'\nResponse:"
        else:
            if fused_emotion in ['Happy', 'Surprise']:
                prompt_instruction = system_instruction + f"Congruent positive: respond supportively and share their excitement.\nUser: '{user_text}'\nResponse:"
            else:
                prompt_instruction = system_instruction + f"Congruent negative/neutral: respond supportively, calmly, and offer comfort.\nUser: '{user_text}'\nResponse:"
        
        inputs = self.tokenizer(prompt_instruction, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            # Reduced max_length and beams to massively speed up inference
            outputs = self.model.generate(
                **inputs,
                max_length=45,
                num_beams=3,
                do_sample=False,
                repetition_penalty=2.0  # Increased heavily to mathematically prohibit FLAN-T5 loops (FIX D)
            )
            
        response_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True).strip()
        
        # Quality Gate: Catch all extreme edge cases (Anti-Copying Gate & Fallback Templates)
        def normalize_text(t: str) -> str:
            t = t.lower()
            # Convert common contractions for math overlap comparisons
            t = t.replace("i'm", "i am").replace("it's", "it is").replace("don't", "do not")
            # Strip all punctuation
            t = re.sub(r'[^\w\s]', '', t)
            return t.strip()
            
        clean_response = normalize_text(response_text)
        clean_input = normalize_text(user_text)
        
        is_short = len(clean_response) <= 5
        is_just_emotion = (clean_response == fused_emotion.lower())
        
        # Fuzzy copycheck: Drop threshold to 0.35 and compare completely sanitized textual structures
        resp_words = set(clean_response.split())
        input_words = set(clean_input.split())
        shared_words = resp_words.intersection(input_words)
        
        is_copying = False
        if len(resp_words) > 0 and (len(shared_words) / len(resp_words)) > 0.35:
            is_copying = True
        
        # Detect repetition loops (e.g. "I'm sorry. I'm sorry. I'm sorry.")
        words = clean_response.split()
        is_looping = len(words) > 6 and len(set(words)) <= 3
        
        failed_quality_checks = is_short or "response" in clean_response or is_just_emotion or is_copying or is_looping
        
        if failed_quality_checks:
            if dissonance_flag:
                return random.choice([
                    "It seems like you might be feeling differently than what you just shared. Want to talk about it?",
                    "I'm sensing some mixed feelings here. Everything okay?",
                    "Your expressions seem to be telling a slightly different story. Let's talk about it."
                ]), None
            elif fused_emotion == 'Happy':
                return random.choice([
                    "That's great to hear! Sounds like you're feeling really good.",
                    "I'm so glad to hear that! Keep enjoying the moment.",
                    "That is wonderful! Celebrate it."
                ]), None
            elif fused_emotion == 'Surprise':
                return random.choice([
                    "Wow! That sounds amazing.",
                    "That is quite incredible! Tell me more.",
                    "I can see why you'd be amazed by that!"
                ]), None
            elif fused_emotion in ['Angry', 'Disgust']:
                return random.choice([
                    "That sounds incredibly frustrating. Let's take a moment and think through this.",
                    "I can understand why you're upset. Take a deep breath.",
                    "It's completely valid to feel frustrated about that."
                ]), None
            elif fused_emotion in ['Sad', 'Fear']:
                return random.choice([
                    "I'm so sorry you're feeling this way. Do you want to talk about it?",
                    "That sounds really tough. I'm here for you.",
                    "That must feel overwhelming. I am listening."
                ]), None
            else:
                return random.choice([
                    "I understand. I am here for you if you want to talk.",
                    "Thanks for sharing that with me.",
                    "I hear you. Let me know if you need anything.",
                    "That makes sense. Tell me more.",
                    "I am listening. Take your time.",
                    "Okay. I'm here however you need me to be."
                ]), None
                
        # FIX 4: LLM Consistency Enforcement (Preventing Toxic Positivity)
        if fused_emotion in ['Sad', 'Angry', 'Fear']:
            toxic_positivity = ["good idea", "great", "wonderful", "happy for you", "looks fun", "sounds fun", "excellent", "worth it"]
            if any(p in response_text.lower() for p in toxic_positivity):
                return random.choice([
                    "I am so sorry you are going through this. I am here to listen.",
                    "That sounds really difficult. Take all the time you need.",
                    "I hear you. I am here for you however you need me."
                ]), "Quality Gate: Blocked Toxic Positivity"
                
        return response_text, None
