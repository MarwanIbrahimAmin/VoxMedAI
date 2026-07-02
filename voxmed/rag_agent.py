import os
import json
import re
from langchain_pinecone import PineconeVectorStore
from langchain_huggingface import HuggingFaceEmbeddings
from groq import Groq
from dotenv import load_dotenv

from voxmed.config import (
    CARDIAC_CONTEXT_HINTS,
    CARDIOVASCULAR_RISK_FACTORS,
    EMBEDDING_MODEL_NAME,
    GROQ_MODEL_NAME,
    HIGH_RISK_SYMPTOMS,
    PINECONE_INDEX_NAME,
    PINECONE_TOP_K,
)
from voxmed.prompt_templates import REPORT_SYSTEM_PROMPT, build_report_prompt

load_dotenv()


def _format_structured_text(value, indent: int = 0) -> str:
    """Render nested dict/list values into readable plain text."""
    prefix = "  " * indent

    if isinstance(value, dict):
        lines = []
        for key, item in value.items():
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}{key}:")
                nested_text = _format_structured_text(item, indent + 1)
                if nested_text:
                    lines.append(nested_text)
            else:
                lines.append(f"{prefix}{key}: {item}")
        return "\n".join(lines)

    if isinstance(value, list):
        lines = []
        for item in value:
            if isinstance(item, (dict, list)):
                lines.append(f"{prefix}-")
                nested_text = _format_structured_text(item, indent + 1)
                if nested_text:
                    lines.append(nested_text)
            else:
                lines.append(f"{prefix}- {item}")
        return "\n".join(lines)

    return str(value).strip()


def _extract_first_json_object(text: str) -> dict:
    """Best-effort JSON extraction from a model response."""
    if not text:
        return {}

    stripped = text.strip()
    if stripped.startswith("{") and stripped.endswith("}"):
        try:
            return json.loads(stripped)
        except json.JSONDecodeError:
            pass

    match = re.search(r"\{.*\}", stripped, flags=re.DOTALL)
    if match:
        try:
            return json.loads(match.group(0))
        except json.JSONDecodeError:
            return {}

    return {}


def _normalize_report_payload(payload, raw_content: str) -> dict:
    """Normalize multiple possible model output shapes into the expected schema."""
    if not isinstance(payload, dict) or not payload:
        # Return empty strings so app.py falls back to a clean local draft report
        return {
            "clinical_report": "",
            "action_plan": "",
        }

    clinical_report = (
        payload.get("clinical_report")
        or payload.get("report")
        or payload.get("final_report")
        or payload.get("summary")
        or payload.get("assessment")
        or ""
    )
    action_plan = (
        payload.get("action_plan")
        or payload.get("plan")
        or payload.get("management_plan")
        or payload.get("recommendations")
        or ""
    )

    clinical_report = _format_structured_text(clinical_report).strip()
    action_plan = _format_structured_text(action_plan).strip()

    # Sanitize markdown code fences, JSON brackets, and braces if they leak
    def sanitize_narrative(text: str) -> str:
        text = re.sub(r"```[a-zA-Z]*\n?", "", text)
        text = text.replace("`", "")
        text = text.replace("{", "").replace("}", "").replace("[", "").replace("]", "")
        return text.strip()

    clinical_report = sanitize_narrative(clinical_report)
    action_plan = sanitize_narrative(action_plan)

    if not clinical_report and action_plan:
        clinical_report = action_plan
    if not action_plan and clinical_report:
        action_plan = clinical_report

    return {
        "clinical_report": clinical_report,
        "action_plan": action_plan,
    }


def build_retrieval_query(patient_data: dict, transcription: str = "", uploaded_reports_text: str = "") -> str:
    symptoms = patient_data.get("symptoms", []) if isinstance(patient_data, dict) else []
    medical_history = patient_data.get("medical_history", {}) if isinstance(patient_data, dict) else {}
    current_medications = patient_data.get("current_medications", []) if isinstance(patient_data, dict) else []
    stopped_medications = patient_data.get("stopped_medications", []) if isinstance(patient_data, dict) else []

    history_conditions = []
    if isinstance(medical_history, dict):
        history_conditions = medical_history.get("conditions", [])
    else:
        history_conditions = medical_history or []

    symptom_text = " ".join(str(item).lower() for item in symptoms)
    history_text = " ".join(str(item).lower() for item in history_conditions)
    medication_text = " ".join(json.dumps(item, ensure_ascii=False).lower() for item in current_medications + stopped_medications)

    has_high_risk_symptom = any(keyword in symptom_text for keyword in HIGH_RISK_SYMPTOMS)
    has_cardiovascular_risk = any(
        risk in symptom_text or risk in history_text or risk in medication_text
        for risk in CARDIOVASCULAR_RISK_FACTORS
    )

    query_parts = []
    if symptoms:
        query_parts.append("Symptoms: " + ", ".join(str(item) for item in symptoms))
    if history_conditions:
        query_parts.append("Medical history: " + ", ".join(str(item) for item in history_conditions))
    if stopped_medications:
        stopped_text = ", ".join(json.dumps(item, ensure_ascii=False) for item in stopped_medications)
        query_parts.append("Stopped medications: " + stopped_text)

    if has_high_risk_symptom and has_cardiovascular_risk:
        query_parts.append("Clinical objective: prioritize cardiovascular differential diagnosis and immediate risk management.")
        query_parts.append("Oxford guideline hints: " + ", ".join(CARDIAC_CONTEXT_HINTS))

    if transcription.strip():
        query_parts.append("Transcription context: " + transcription.strip()[:1200])
    if uploaded_reports_text.strip():
        query_parts.append("Prior reports: " + uploaded_reports_text.strip()[:1200])

    return "\n".join(part for part in query_parts if part).strip() or "General medical checkup"

def setup_retriever():
    """
    Sets up the Pinecone vector store retriever using the same embeddings
    used during indexing in the preprocessing notebook.
    """
    if not os.environ.get("PINECONE_API_KEY"):
        raise ValueError("[!] PINECONE_API_KEY is missing. Add it in Hugging Face Secrets or your local .env file.")
        
    print("[*] Initializing embedding model and Pinecone connection...")
    embeddings = HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL_NAME,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True}
    )
    
    vectorstore = PineconeVectorStore(
        index_name=PINECONE_INDEX_NAME,
        embedding=embeddings
    )
    
    return vectorstore.as_retriever(search_kwargs={"k": PINECONE_TOP_K})

def generate_clinical_report(
    patient_data: str,
    retrieved_docs: list,
    uploaded_reports_text: str = "",
) -> dict:
    """
    Takes the structured patient data (JSON), retrieved clinical context, and
    optional uploaded medical report text, then uses Groq LLM to synthesize a
    reviewable English clinical note plus action plan.
    """
    if not os.environ.get("GROQ_API_KEY"):
        raise ValueError("[!] GROQ_API_KEY is missing. Add it in Hugging Face Secrets or your local .env file.")

    client = Groq()
    
    context_text = "\n\n".join([doc.page_content for doc in retrieved_docs])
    uploaded_text_block = uploaded_reports_text.strip() or "No previous medical reports were provided."
    
    prompt = build_report_prompt(patient_data, context_text, uploaded_text_block)

    print("[*] Synthesizing final response with LLM...")
    response = client.chat.completions.create(
        messages=[
            {"role": "system", "content": REPORT_SYSTEM_PROMPT},
            {"role": "user", "content": prompt}
        ],
        model=GROQ_MODEL_NAME,
        temperature=0.2, # Slight variation allowed for conversational response
        response_format={"type": "json_object"},
    )
    
    raw_content = response.choices[0].message.content.strip()
    parsed_content = _extract_first_json_object(raw_content)
    return _normalize_report_payload(parsed_content, raw_content)


def generate_action_plan(patient_data: str, retrieved_docs: list) -> str:
    """
    Backward-compatible wrapper that returns the combined English note.
    """
    report = generate_clinical_report(patient_data, retrieved_docs)
    clinical_report = report.get("clinical_report", "")
    action_plan = report.get("action_plan", "")
    return f"{clinical_report}\n\nAction Plan:\n{action_plan}".strip()
