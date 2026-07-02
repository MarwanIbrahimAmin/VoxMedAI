from __future__ import annotations

import json
import re
from typing import Any


MEDICATION_FIELDS = (
    "trade_name",
    "generic_name",
    "dosage",
    "frequency",
    "route",
    "indication",
    "stopped_reason",
)

VITAL_FIELDS = (
    "blood_pressure",
    "heart_rate",
    "respiratory_rate",
    "temperature",
    "spo2",
)

ARABIC_CHAR_RE = re.compile(r"[\u0600-\u06FF]")

MEDICAL_TRANSLATION_MAP = {
    "هيبرتنشن": "Hypertension",
    "ارتفاع ضغط": "Hypertension",
    "ضغط عالي": "Hypertension",
    "الم في الصدر": "Chest pain",
    "قلم في الصدر": "Chest pain",
    "ضيق تنفس": "Shortness of breath",
    "نهجان": "Shortness of breath",
    "كونكور": "Concor",
    "اسبرين": "Aspirin",
    "أسبرين": "Aspirin",
}

GENERIC_NAME_MAP = {
    "concor": "Bisoprolol",
    "aspirin": "Acetylsalicylic acid",
}


def _as_text(value: Any) -> str | None:
    if value is None:
        return None
    if isinstance(value, str):
        value = value.strip()
        for raw_term, normalized_term in MEDICAL_TRANSLATION_MAP.items():
            value = value.replace(raw_term, normalized_term)
        return value or None
    return str(value).strip() or None


def _as_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return value
    if isinstance(value, str):
        text = value.strip()
        if text.isdigit():
            return int(text)
    return None


def _as_text_list(value: Any) -> list[str]:
    if value is None:
        return []
    if isinstance(value, list):
        result = []
        for item in value:
            text = _as_text(item)
            if text:
                result.append(text)
        return result
    text = _as_text(value)
    return [text] if text else []


def _as_medication_entry(value: Any) -> dict[str, str | None]:
    entry = {field: None for field in MEDICATION_FIELDS}

    if isinstance(value, str):
        entry["trade_name"] = _as_text(value)
        if entry["trade_name"]:
            generic_name = GENERIC_NAME_MAP.get(entry["trade_name"].strip().lower())
            if generic_name:
                entry["generic_name"] = generic_name
        return entry

    if isinstance(value, dict):
        for field in MEDICATION_FIELDS:
            entry[field] = _as_text(value.get(field))
        trade_name = (entry.get("trade_name") or "").strip().lower()
        if not entry.get("generic_name") and trade_name in GENERIC_NAME_MAP:
            entry["generic_name"] = GENERIC_NAME_MAP[trade_name]
        return entry

    text = _as_text(value)
    if text:
        entry["trade_name"] = text
        generic_name = GENERIC_NAME_MAP.get(text.strip().lower())
        if generic_name:
            entry["generic_name"] = generic_name
    return entry


def _contains_non_english_text(value: Any) -> bool:
    if value is None:
        return False
    if isinstance(value, str):
        return bool(ARABIC_CHAR_RE.search(value))
    if isinstance(value, list):
        return any(_contains_non_english_text(item) for item in value)
    if isinstance(value, dict):
        return any(_contains_non_english_text(key) or _contains_non_english_text(item) for key, item in value.items())
    return False


def _as_medication_list(value: Any) -> list[dict[str, str | None]]:
    if not value:
        return []
    if isinstance(value, list):
        return [_as_medication_entry(item) for item in value]
    return [_as_medication_entry(value)]


def _as_medical_history(value: Any) -> dict[str, list[str]]:
    if isinstance(value, dict):
        conditions = _as_text_list(value.get("conditions"))
        denied_conditions = _as_text_list(value.get("denied_conditions"))
        return {
            "conditions": conditions,
            "denied_conditions": denied_conditions,
        }

    return {
        "conditions": _as_text_list(value),
        "denied_conditions": [],
    }


def _as_vitals(value: Any) -> dict[str, str | None]:
    if not isinstance(value, dict):
        value = {}

    return {
        "blood_pressure": _as_text(value.get("blood_pressure")),
        "heart_rate": _as_text(value.get("heart_rate")),
        "respiratory_rate": _as_text(value.get("respiratory_rate")),
        "temperature": _as_text(value.get("temperature")),
        "spo2": _as_text(value.get("spo2")),
    }


def normalize_extraction_payload(payload: Any) -> dict:
    if not isinstance(payload, dict):
        payload = {}

    return {
        "patient_name": _as_text(payload.get("patient_name")),
        "patient_age": _as_int(payload.get("patient_age")),
        "symptoms": _as_text_list(payload.get("symptoms")),
        "medical_history": _as_medical_history(payload.get("medical_history")),
        "surgical_history": _as_text_list(payload.get("surgical_history")),
        "current_medications": _as_medication_list(payload.get("current_medications")),
        "stopped_medications": _as_medication_list(payload.get("stopped_medications")),
        "allergies": _as_text_list(payload.get("allergies")),
        "social_history": _as_text_list(payload.get("social_history")),
        "vitals": _as_vitals(payload.get("vitals")),
    }


def validate_extraction_payload(payload: Any) -> list[str]:
    errors = []
    normalized = normalize_extraction_payload(payload)

    if normalized["patient_name"] is None:
        errors.append("patient_name is missing")
    vitals = normalized["vitals"]
    if set(vitals.keys()) != set(VITAL_FIELDS):
        errors.append("vitals must contain only quantitative fields")

    medical_history = normalized["medical_history"]
    if set(medical_history.keys()) != {"conditions", "denied_conditions"}:
        errors.append("medical_history must contain conditions and denied_conditions")

    if _contains_non_english_text(normalized):
        errors.append("payload contains non-English text; output must be standardized medical English")

    return errors


def dump_normalized_payload(payload: Any) -> str:
    return json.dumps(normalize_extraction_payload(payload), ensure_ascii=False, indent=2)
