import os
import json
from groq import Groq
from dotenv import load_dotenv

from voxmed.config import DEFAULT_AUDIO_LANGUAGE, GROQ_MODEL_NAME, WHISPER_MODEL_NAME
from voxmed.prompt_templates import build_extraction_prompt
from voxmed.schema_validation import normalize_extraction_payload, validate_extraction_payload

load_dotenv()

def transcribe_audio(audio_path: str) -> str:
    """
    Transcribes the given audio file using Groq's cloud-deployed Whisper model (whisper-large-v3) 
    for blazing-fast, high-accuracy inference compared to running locally on CPU.
    """
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("[!] GROQ_API_KEY is missing. Add it in Hugging Face Secrets or your local .env file.")

    print(f"[*] Processing '{audio_path}' with Groq ({WHISPER_MODEL_NAME})...")
    
    try:
        client = Groq()
        
        # We can also add an initial_prompt with medical terms to help Whisper recognize them.
        medical_prompt = "“This is a medical condition for a patient. Important words: blood pressure, sugar, heartbeat, x-ray, medication."

        with open(audio_path, "rb") as file:
            transcription = client.audio.transcriptions.create(
                file=(os.path.basename(audio_path), file.read()),
                model=WHISPER_MODEL_NAME,
                prompt=medical_prompt,       # optional, helps guide vocabulary
                response_format="json",
                language=DEFAULT_AUDIO_LANGUAGE,               # force Arabic
                temperature=0.0              # low temperature for accuracy
            )
        
        return transcription.text.strip()
        
    except Exception as e:
        print(f"[!] Error during transcription: {e}")
        return ""

def extract_clinical_data(raw_text: str) -> str:
    """
    Sends raw Arabic dictation to an LLM to extract structured clinical entities.
    Returns a strict JSON string.
    """
    if not raw_text:
        print("[!] No text provided for extraction.")
        return "{}"

    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("[!] GROQ_API_KEY is missing. Add it in Hugging Face Secrets or your local .env file.")

    print("[*] Extracting clinical entities via LLM...")
    client = Groq()
    
    prompt = build_extraction_prompt()

    try:
        response = client.chat.completions.create(
            messages=[
                {"role": "system", "content": prompt},
                {"role": "user", "content": raw_text}
            ],
            model=GROQ_MODEL_NAME,
            temperature=0.0, # Kept at 0 for deterministic extraction
            response_format={"type": "json_object"}
        )
        raw_content = response.choices[0].message.content.strip()
        parsed_content = json.loads(raw_content)
        normalized_payload = normalize_extraction_payload(parsed_content)
        validation_errors = validate_extraction_payload(parsed_content)
        if validation_errors:
            # strict fallback schema with English-only placeholders
            fallback_payload = {
                "patient_name": normalized_payload.get("patient_name") or "Unknown",
                "patient_age": normalized_payload.get("patient_age"),
                "symptoms": normalized_payload.get("symptoms", []),
                "medical_history": normalized_payload.get("medical_history", {"conditions": [], "denied_conditions": []}),
                "surgical_history": normalized_payload.get("surgical_history", []),
                "current_medications": normalized_payload.get("current_medications", []),
                "stopped_medications": normalized_payload.get("stopped_medications", []),
                "allergies": normalized_payload.get("allergies", []),
                "social_history": normalized_payload.get("social_history", []),
                "vitals": normalized_payload.get(
                    "vitals",
                    {
                        "blood_pressure": None,
                        "heart_rate": None,
                        "respiratory_rate": None,
                        "temperature": None,
                        "spo2": None,
                    },
                ),
            }
            return json.dumps(fallback_payload, ensure_ascii=False, indent=2)
        return json.dumps(normalized_payload, ensure_ascii=False, indent=2)
        
    except Exception as e:
        print(f"[!] API call failed: {e}")
        return "{}"

if __name__ == "__main__":
    path = r'C:\Users\omarw\Documents\Sound Recordings\Recording.mp3'
    target_audio = path
    
    # 1. Run transcription
    transcription = transcribe_audio(target_audio)
    
    if transcription:
        print(f"\n--- Raw Transcription ---\n{transcription}\n")
        
        # 2. Run extraction
        structured_data = extract_clinical_data(transcription)
        print("--- Structured Data (JSON) ---")
        print(structured_data)
    else:
        print("[!] Pipeline stopped due to transcription failure.")