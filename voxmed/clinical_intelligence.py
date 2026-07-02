import json
import re
from typing import Any

from voxmed.config import (
    CARDIOVASCULAR_RISK_FACTORS,
    HIGH_CONFIDENCE_SCORE,
    HIGH_RISK_SYMPTOMS,
    HIGH_SEVERITY_SCORE,
    MEDIUM_SEVERITY_SCORE,
    MODERATE_CONFIDENCE_SCORE,
    URGENT_TRIAGE_SCORE,
)
from voxmed.enums import ConfidenceLevel, SeverityLevel, TriageLevel


MEDIUM_RISK_SYMPTOMS = [
    "fever",
    "persistent pain",
    "severe pain",
    "abdominal pain",
    "vomiting",
    "diarrhea",
    "dehydration",
    "dizziness",
    "palpitations",
    "cough",
]

RED_FLAG_PATTERNS = [
    ("chest pain", "Chest pain may indicate a time-sensitive cardiac or pulmonary condition."),
    ("shortness of breath", "Shortness of breath is a potential respiratory or cardiac red flag."),
    ("difficulty breathing", "Breathing difficulty requires urgent clinical assessment."),
    ("syncope", "Syncope can indicate cardiovascular or neurological instability."),
    ("fainting", "Fainting can indicate cardiovascular or neurological instability."),
    ("unilateral weakness", "Unilateral weakness may indicate possible stroke or focal neurological deficit."),
    ("slurred speech", "Slurred speech may indicate possible stroke or focal neurological deficit."),
    ("seizure", "Seizure activity requires prompt assessment."),
    ("severe bleeding", "Severe bleeding is an urgent safety concern."),
    ("suicidal", "Suicidal ideation requires immediate safeguarding and escalation."),
]


def _normalize_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        return value.lower().strip()
    if isinstance(value, list):
        return " ".join(_normalize_text(item) for item in value)
    if isinstance(value, dict):
        return " ".join(f"{key} {_normalize_text(item)}" for key, item in value.items())
    return str(value).lower().strip()


def _extract_numeric_values(text: str) -> list[float]:
    return [float(match) for match in re.findall(r"\d+(?:\.\d+)?", text)]


def _parse_blood_pressure(value: str) -> tuple[float | None, float | None]:
    match = re.search(r"(\d{2,3})\s*/\s*(\d{2,3})", value)
    if not match:
        return None, None
    return float(match.group(1)), float(match.group(2))


def _medications_text(value: Any) -> str:
    if not value:
        return ""
    if isinstance(value, list):
        return " ".join(_medications_text(item) for item in value)
    if isinstance(value, dict):
        return json.dumps(value, ensure_ascii=False).lower()
    return _normalize_text(value)


def detect_red_flags(patient_data: dict, transcription: str = "", uploaded_reports_text: str = "") -> dict:
    combined_text = " ".join(
        [
            _normalize_text(patient_data),
            _normalize_text(transcription),
            _normalize_text(uploaded_reports_text),
        ]
    )

    matched_flags = []
    for keyword, explanation in RED_FLAG_PATTERNS:
        if keyword in combined_text:
            matched_flags.append({"flag": keyword, "reason": explanation})

    vitals = patient_data.get("vitals", {}) if isinstance(patient_data, dict) else {}
    medical_history = patient_data.get("medical_history", {}) if isinstance(patient_data, dict) else {}
    patient_age = patient_data.get("patient_age") if isinstance(patient_data, dict) else None

    history_conditions = []
    if isinstance(medical_history, dict):
        history_conditions = medical_history.get("conditions", []) or []
    elif isinstance(medical_history, list):
        history_conditions = medical_history

    history_text = _normalize_text(history_conditions)
    symptom_text = _normalize_text(patient_data.get("symptoms", []))
    stopped_medications_text = _medications_text(patient_data.get("stopped_medications", []))

    blood_pressure = _normalize_text(vitals.get("blood_pressure"))
    heart_rate_text = _normalize_text(vitals.get("heart_rate"))
    respiratory_rate_text = _normalize_text(vitals.get("respiratory_rate"))
    temperature_text = _normalize_text(vitals.get("temperature"))
    spo2_text = _normalize_text(vitals.get("spo2"))

    systolic, diastolic = _parse_blood_pressure(blood_pressure)
    if systolic is not None and diastolic is not None and (systolic >= 180 or systolic <= 90 or diastolic <= 60):
        matched_flags.append({"flag": "blood_pressure", "reason": f"Blood pressure {int(systolic)}/{int(diastolic)} is outside a reassuring range."})

    heart_rate_values = _extract_numeric_values(heart_rate_text)
    if heart_rate_values:
        heart_rate = heart_rate_values[0]
        if heart_rate >= 120 or heart_rate <= 50:
            matched_flags.append({"flag": "heart_rate", "reason": f"Heart rate {int(heart_rate)} may need urgent review."})

    respiratory_rate_values = _extract_numeric_values(respiratory_rate_text)
    if respiratory_rate_values:
        respiratory_rate = respiratory_rate_values[0]
        if respiratory_rate >= 28 or respiratory_rate <= 10:
            matched_flags.append({"flag": "respiratory_rate", "reason": f"Respiratory rate {int(respiratory_rate)} may indicate instability."})

    temperature_values = _extract_numeric_values(temperature_text)
    if temperature_values:
        temperature = temperature_values[0]
        if temperature >= 39.0 or temperature <= 35.0:
            matched_flags.append({"flag": "temperature", "reason": f"Temperature {temperature:.1f}C is outside a reassuring range."})

    spo2_values = _extract_numeric_values(spo2_text)
    if spo2_values:
        spo2 = spo2_values[0]
        if spo2 <= 92:
            matched_flags.append({"flag": "spo2", "reason": f"SpO2 {int(spo2)}% may indicate hypoxemia and requires urgent review."})

    has_chest_pain = "chest pain" in symptom_text
    has_breathing_symptoms = any(item in symptom_text for item in ("shortness of breath", "difficulty breathing", "dyspnea"))

    has_age_risk = isinstance(patient_age, int) and patient_age > 50
    has_history_risk = any(risk in symptom_text or risk in history_text for risk in CARDIOVASCULAR_RISK_FACTORS)
    has_recent_aspirin_stop = "aspirin" in stopped_medications_text
    cardiovascular_risk_profile = has_age_risk or has_history_risk or has_recent_aspirin_stop

    has_hypertension_history = "hypertension" in history_text or "high blood pressure" in history_text
    forced_cardiac_urgent_case = has_age_risk and has_chest_pain and has_hypertension_history and has_recent_aspirin_stop

    if (has_chest_pain or has_breathing_symptoms) and cardiovascular_risk_profile:
        matched_flags.append(
            {
                "flag": "cardiovascular_high_risk_profile",
                "reason": "High-risk cardiopulmonary symptoms with cardiovascular risk factors detected; urgent escalation is required.",
            }
        )

    if forced_cardiac_urgent_case:
        matched_flags.append(
            {
                "flag": "forced_cardiac_urgent_case",
                "reason": "Age > 50 with chest pain, hypertension history, and recent aspirin discontinuation requires immediate high-priority escalation.",
            }
        )

    severity_score = 10
    if matched_flags:
        severity_score += 55

    if (has_chest_pain or has_breathing_symptoms) and cardiovascular_risk_profile:
        severity_score = max(severity_score, URGENT_TRIAGE_SCORE)

    if forced_cardiac_urgent_case:
        severity_score = max(severity_score, URGENT_TRIAGE_SCORE)

    if any(
        flag["flag"]
        in {
            "chest pain",
            "shortness of breath",
            "difficulty breathing",
            "syncope",
            "unilateral weakness",
            "slurred speech",
            "seizure",
            "severe bleeding",
            "suicidal",
            "cardiovascular_high_risk_profile",
            "forced_cardiac_urgent_case",
        }
        for flag in matched_flags
    ):
        severity_score = max(severity_score, URGENT_TRIAGE_SCORE)
    elif any(keyword in combined_text for keyword in MEDIUM_RISK_SYMPTOMS):
        severity_score = max(severity_score, 50)

    if any(symptom in symptom_text for symptom in HIGH_RISK_SYMPTOMS) and cardiovascular_risk_profile:
        severity_score = max(severity_score, URGENT_TRIAGE_SCORE)

    severity_score = min(severity_score, 100)
    if severity_score >= HIGH_SEVERITY_SCORE:
        label = SeverityLevel.HIGH
    elif severity_score >= MEDIUM_SEVERITY_SCORE:
        label = SeverityLevel.MEDIUM
    else:
        label = SeverityLevel.LOW

    recommendation = {
        SeverityLevel.HIGH: "Urgent clinician review recommended. Escalate if symptoms are ongoing or worsening.",
        SeverityLevel.MEDIUM: "Clinical review recommended. Check vitals, history, and follow-up details before final sign-off.",
        SeverityLevel.LOW: "Routine review appears reasonable based on the available information.",
    }[label]

    return {
        "severity_label": str(label),
        "severity_score": severity_score,
        "red_flags": matched_flags,
        "recommendation": recommendation,
    }


def estimate_confidence(patient_data: dict, transcription: str = "", uploaded_reports_text: str = "", report_payload: dict | None = None) -> dict:
    completeness = 0
    factors = []

    if _normalize_text(patient_data.get("patient_name")):
        completeness += 15
        factors.append("Patient name captured")
    else:
        factors.append("Patient name missing")

    if patient_data.get("patient_age") not in (None, "", []):
        completeness += 10
        factors.append("Patient age captured")
    else:
        factors.append("Patient age missing")

    symptoms = patient_data.get("symptoms", []) if isinstance(patient_data, dict) else []
    if symptoms:
        completeness += 20
        factors.append("Symptoms extracted")
    else:
        factors.append("Symptoms missing")

    if _normalize_text(patient_data.get("vitals")):
        completeness += 15
        factors.append("Vitals available")
    else:
        factors.append("Vitals missing")

    if _normalize_text(patient_data.get("medical_history")) or _normalize_text(patient_data.get("surgical_history")):
        completeness += 10
        factors.append("History captured")

    if len(_normalize_text(transcription)) > 100:
        completeness += 10
        factors.append("Transcription length sufficient")
    else:
        factors.append("Short transcription")

    if len(_normalize_text(uploaded_reports_text)) > 100:
        completeness += 5
        factors.append("Prior reports supplied")

    if report_payload:
        clinical_report = _normalize_text(report_payload.get("clinical_report"))
        action_plan = _normalize_text(report_payload.get("action_plan"))
        if clinical_report:
            completeness += 7
        if action_plan:
            completeness += 8

    if report_payload and (report_payload.get("clinical_report") or report_payload.get("action_plan")):
        completeness += 5
        factors.append("LLM summary returned")
    else:
        factors.append("LLM summary incomplete")

    completeness = min(completeness, 100)
    if completeness >= HIGH_CONFIDENCE_SCORE:
        label = ConfidenceLevel.HIGH
    elif completeness >= MODERATE_CONFIDENCE_SCORE:
        label = ConfidenceLevel.MODERATE
    else:
        label = ConfidenceLevel.LOW

    return {
        "confidence_label": str(label),
        "confidence_score": completeness,
        "signals": factors,
    }


def build_clinical_intelligence(patient_data: dict, transcription: str = "", uploaded_reports_text: str = "", report_payload: dict | None = None) -> dict:
    red_flags = detect_red_flags(patient_data, transcription, uploaded_reports_text)
    confidence = estimate_confidence(patient_data, transcription, uploaded_reports_text, report_payload)

    if red_flags["severity_label"] == str(SeverityLevel.HIGH):
        triage_label = TriageLevel.URGENT
    elif red_flags["severity_label"] == str(SeverityLevel.MEDIUM):
        triage_label = TriageLevel.NEEDS_REVIEW
    else:
        triage_label = TriageLevel.ROUTINE

    return {
        "triage_label": str(triage_label),
        "red_flags": red_flags,
        "confidence": confidence,
    }
