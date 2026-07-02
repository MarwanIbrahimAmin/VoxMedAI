import streamlit as st
import os
import json
import tempfile
import html
from io import BytesIO
from datetime import datetime
from dotenv import load_dotenv
from reportlab.lib import colors
from reportlab.lib.enums import TA_LEFT
from reportlab.lib.pagesizes import A4
from reportlab.lib.styles import ParagraphStyle, getSampleStyleSheet
from reportlab.platypus import Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

from config import PDF_SOURCE_TEXT, PDF_TITLE, SUPPORTED_AUDIO_TYPES, SUPPORTED_REPORT_TYPES

# Load Environment Variables
load_dotenv()

# Import our custom pipeline modules
from voxmed.intake_module import transcribe_audio, extract_clinical_data
from voxmed.rag_agent import setup_retriever, generate_clinical_report, build_retrieval_query
from voxmed.clinical_intelligence import build_clinical_intelligence

# Page Configuration
st.set_page_config(page_title="VoxMed: AI Medical Copilot", page_icon="🩺", layout="wide")

st.title("🩺 VoxMed: AI Medical Copilot")
st.markdown("""
Welcome to **VoxMed**, your AI-powered clinical assistant.  
Upload a patient's dictation recording and any previous medical reports below to automatically transcribe the symptoms, extract a structured clinical record, search the *Oxford Handbook of Clinical Medicine*, and prepare an English medical report that you can review and edit before exporting as PDF.
""")

# ---- Helpers ----
def _file_to_text(uploaded_file) -> str:
    file_name = uploaded_file.name.lower()
    file_bytes = uploaded_file.getvalue() if hasattr(uploaded_file, "getvalue") else uploaded_file.read()

    if file_name.endswith(".pdf"):
        try:
            from pypdf import PdfReader
        except ImportError:
            st.warning(f"PDF extraction is unavailable for {uploaded_file.name}. Install pypdf to enable it.")
            return ""

        reader = PdfReader(BytesIO(file_bytes))
        extracted_pages = []
        for page in reader.pages:
            page_text = page.extract_text() or ""
            if page_text.strip():
                extracted_pages.append(page_text.strip())
        return "\n\n".join(extracted_pages).strip()

    return file_bytes.decode("utf-8", errors="ignore").strip()


def _combine_uploaded_reports(uploaded_files) -> tuple[str, list[str]]:
    if not uploaded_files:
        return "", []

    report_chunks = []
    report_names = []
    for uploaded_file in uploaded_files:
        extracted_text = _file_to_text(uploaded_file)
        report_names.append(uploaded_file.name)
        if extracted_text:
            report_chunks.append(f"### {uploaded_file.name}\n{extracted_text}")

    return "\n\n".join(report_chunks).strip(), report_names


def _clean_text(value) -> str:
    if value in (None, "", []):
        return "Not specified"
    if isinstance(value, str):
        cleaned = value.replace("{", "").replace("}", "").replace("[", "").replace("]", "")
        cleaned = cleaned.replace('"', "").replace("'", "")
        return " ".join(cleaned.split()) or "Not specified"
    if isinstance(value, list):
        items = [
            _clean_text(item)
            for item in value
            if _clean_text(item) != "Not specified"
        ]
        return "; ".join(items) if items else "Not specified"
    if isinstance(value, dict):
        items = []
        for key, item in value.items():
            item_text = _clean_text(item)
            if item_text != "Not specified":
                items.append(f"{key}: {item_text}")
        return "; ".join(items) if items else "Not specified"
    return " ".join(str(value).split()) or "Not specified"


def _format_medication_entry(entry: dict, stopped: bool = False) -> str:
    if not isinstance(entry, dict):
        return _clean_text(entry)

    trade_name = _clean_text(entry.get("trade_name"))
    generic_name = _clean_text(entry.get("generic_name"))
    dosage = _clean_text(entry.get("dosage"))
    frequency = _clean_text(entry.get("frequency"))
    route = _clean_text(entry.get("route"))
    indication = _clean_text(entry.get("indication"))
    stopped_reason = _clean_text(entry.get("stopped_reason"))

    display_name = trade_name
    if generic_name and generic_name != "Not specified":
        display_name = f"{display_name} ({generic_name})" if display_name != "Not specified" else generic_name

    parts = [display_name]
    if dosage != "Not specified":
        parts.append(dosage)
    if frequency != "Not specified":
        parts.append(frequency)
    if route != "Not specified":
        parts.append(route)
    if indication != "Not specified":
        parts.append(f"for {indication}")
    if stopped:
        stop_note = f"Stopped: {stopped_reason}" if stopped_reason != "Not specified" else "Stopped; duration/dose unspecified"
        parts.append(stop_note)

    return ", ".join(part for part in parts if part and part != "Not specified") or "Not specified"


def _format_medication_profile(title: str, medications) -> str:
    if not medications:
        return "Not specified"
    lines = [title]
    for medication in medications:
        lines.append(f"- {_format_medication_entry(medication)}")
    return "\n".join(lines)


def _format_medical_history(history) -> str:
    if not isinstance(history, dict):
        items = history if isinstance(history, list) else [history]
        filtered = [_clean_text(item) for item in items if _clean_text(item) != "Not specified"]
        return "\n".join(["Medical History"] + [f"- {item}" for item in filtered]) if filtered else "Not specified"

    conditions = history.get("conditions", []) or []
    denied_conditions = history.get("denied_conditions", []) or []
    lines = ["Medical History"]
    if conditions:
        lines.append("Current / relevant conditions:")
        for condition in conditions:
            lines.append(f"- {_clean_text(condition)}")
    if denied_conditions:
        lines.append("Denied conditions:")
        for condition in denied_conditions:
            lines.append(f"- {_clean_text(condition)}")
    return "\n".join(lines) if len(lines) > 1 else "Not specified"


def _format_vitals(vitals) -> str:
    if not isinstance(vitals, dict):
        return _clean_text(vitals)

    labels = [
        ("Blood Pressure", vitals.get("blood_pressure")),
        ("Heart Rate", vitals.get("heart_rate")),
        ("Respiratory Rate", vitals.get("respiratory_rate")),
        ("Temperature", vitals.get("temperature")),
        ("SpO2", vitals.get("spo2")),
    ]
    lines = ["Vital Signs"]
    for label, value in labels:
        clean_value = _clean_text(value)
        if clean_value != "Not specified":
            lines.append(f"- {label}: {clean_value}")
    return "\n".join(lines) if len(lines) > 1 else "Not specified"


def _format_section_value(title: str, value) -> str:
    if title == "Medical History":
        return _format_medical_history(value)
    if title == "Current Medications":
        return _format_medication_profile("Current Medications", value)
    if title == "Stopped Medications":
        if not value:
            return "Not specified"
        lines = ["Stopped Medications"]
        for medication in value:
            lines.append(f"- {_format_medication_entry(medication, stopped=True)}")
        return "\n".join(lines)
    if title == "Vitals":
        return _format_vitals(value)
    if title in {"Allergies", "Social History", "Surgical History"}:
        if not value:
            return "Not specified"
        lines = [title]
        for item in value if isinstance(value, list) else [value]:
            clean_item = _clean_text(item)
            if clean_item != "Not specified":
                lines.append(f"- {clean_item}")
        return "\n".join(lines) if len(lines) > 1 else "Not specified"
    if title == "Symptoms":
        if not value:
            return "Not specified"
        lines = [title]
        for item in value if isinstance(value, list) else [value]:
            clean_item = _clean_text(item)
            if clean_item != "Not specified":
                lines.append(f"- {clean_item}")
        return "\n".join(lines) if len(lines) > 1 else "Not specified"
    if title in {"Triage Level", "Severity Score", "Confidence Level", "Confidence Score"}:
        return _clean_text(value)
    return _clean_text(value)


def _make_pdf_bytes(patient_data: dict, clinical_report_text: str, action_plan_text: str, intelligence: dict | None = None) -> bytes:
    buffer = BytesIO()
    document = SimpleDocTemplate(
        buffer,
        pagesize=A4,
        rightMargin=36,
        leftMargin=36,
        topMargin=42,
        bottomMargin=36,
    )

    styles = getSampleStyleSheet()
    title_style = ParagraphStyle(
        "VoxMedTitle",
        parent=styles["Title"],
        fontName="Helvetica-Bold",
        fontSize=18,
        leading=22,
        textColor=colors.HexColor("#17324d"),
        alignment=TA_LEFT,
        spaceAfter=8,
    )
    subtitle_style = ParagraphStyle(
        "VoxMedSubtitle",
        parent=styles["Normal"],
        fontName="Helvetica",
        fontSize=9,
        leading=11,
        textColor=colors.HexColor("#52616b"),
        spaceAfter=10,
    )
    section_style = ParagraphStyle(
        "VoxMedSection",
        parent=styles["Heading2"],
        fontName="Helvetica-Bold",
        fontSize=12,
        leading=14,
        textColor=colors.HexColor("#17324d"),
        spaceBefore=10,
        spaceAfter=6,
    )
    body_style = ParagraphStyle(
        "VoxMedBody",
        parent=styles["BodyText"],
        fontName="Helvetica",
        fontSize=10,
        leading=13,
        textColor=colors.black,
    )

    def append_markdown_paragraphs(text: str) -> None:
        subheading_style = ParagraphStyle(
            "VoxMedSubheading",
            parent=body_style,
            fontName="Helvetica-Bold",
            fontSize=11,
            leading=14,
            spaceBefore=6,
            spaceAfter=3,
        )
        bullet_style = ParagraphStyle(
            "VoxMedBullet",
            parent=body_style,
            leftIndent=15,
            firstLineIndent=-10,
            spaceAfter=3,
        )

        lines = text.split("\n")
        for line in lines:
            line_stripped = line.strip()
            if not line_stripped:
                story.append(Spacer(1, 4))
                continue

            # Headers: #, ##, ###, etc.
            if line_stripped.startswith("#"):
                depth = 0
                for char in line_stripped:
                    if char == "#":
                        depth += 1
                    else:
                        break
                header_text = line_stripped[depth:].strip()
                if header_text:
                    safe_header_text = html.escape(header_text)
                    style = section_style if depth <= 2 else subheading_style
                    story.append(Paragraph(safe_header_text, style))
                continue

            # Bullet points: - or *
            if line_stripped.startswith("-") or line_stripped.startswith("*"):
                bullet_content = line_stripped[1:].strip()
                if bullet_content:
                    safe_bullet_content = html.escape(bullet_content)
                    story.append(Paragraph(f"&bull; {safe_bullet_content}", bullet_style))
                continue

            # Plain text fallback
            safe_line = html.escape(line_stripped)
            story.append(Paragraph(safe_line, body_style))

    story = []
    story.append(Paragraph(PDF_TITLE, title_style))
    story.append(Paragraph(f"Generated on {datetime.now().strftime('%Y-%m-%d %H:%M')}", subtitle_style))

    # 1. Patient Metadata
    patient_table_data = [
        [Paragraph("Patient Name", body_style), Paragraph(_clean_text(patient_data.get("patient_name")), body_style)],
        [Paragraph("Patient Age", body_style), Paragraph(_clean_text(patient_data.get("patient_age")), body_style)],
        [Paragraph("Source", body_style), Paragraph(PDF_SOURCE_TEXT, body_style)],
    ]

    patient_table = Table(patient_table_data, colWidths=[120, 360])
    patient_table.setStyle(
        TableStyle(
            [
                ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#eef3f7")),
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#9bb0bd")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c7d3db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
                ("TOPPADDING", (0, 0), (-1, -1), 6),
                ("BOTTOMPADDING", (0, 0), (-1, -1), 6),
            ]
        )
    )
    story.append(patient_table)
    story.append(Spacer(1, 10))

    # 2. Clinical Intelligence Summary
    if intelligence:
        red_flags = intelligence.get("red_flags", {})
        intel_table_data = [
            [Paragraph("Triage Level", body_style), Paragraph(_clean_text(intelligence.get("triage_label")), body_style)],
            [Paragraph("Severity Score", body_style), Paragraph(f"{red_flags.get('severity_score', '0')}/100", body_style)],
            [Paragraph("Confidence Level", body_style), Paragraph(_clean_text(intelligence.get("confidence", {}).get("confidence_label")), body_style)],
        ]
        intel_table = Table(intel_table_data, colWidths=[120, 360])
        intel_table.setStyle(
            TableStyle([
                ("BOX", (0, 0), (-1, -1), 0.6, colors.HexColor("#c7d3db")),
                ("INNERGRID", (0, 0), (-1, -1), 0.4, colors.HexColor("#c7d3db")),
                ("VALIGN", (0, 0), (-1, -1), "TOP"),
                ("LEFTPADDING", (0, 0), (-1, -1), 8),
                ("RIGHTPADDING", (0, 0), (-1, -1), 8),
            ])
        )
        story.append(Paragraph("Clinical Intelligence Summary", section_style))
        story.append(intel_table)
        story.append(Spacer(1, 10))

    # 3. Final Clinical Assessment
    story.append(Paragraph("Final Clinical Assessment", section_style))
    append_markdown_paragraphs(clinical_report_text)
    story.append(Spacer(1, 10))

    # 4. Action Plan
    story.append(Paragraph("Action Plan", section_style))
    append_markdown_paragraphs(action_plan_text)

    document.build(story)
    return buffer.getvalue()


def _safe_text(value: str) -> str:
    return value.strip() if isinstance(value, str) else ""


def _build_fallback_report(patient_data: dict, transcription: str, uploaded_reports_text: str) -> tuple[str, str]:
    patient_name = _clean_text(patient_data.get("patient_name"))
    patient_age = _clean_text(patient_data.get("patient_age"))
    medical_history = _format_section_value("Medical History", patient_data.get("medical_history"))
    surgical_history = _format_section_value("Surgical History", patient_data.get("surgical_history"))
    current_medications = _format_section_value("Current Medications", patient_data.get("current_medications"))
    stopped_medications = _format_section_value("Stopped Medications", patient_data.get("stopped_medications"))
    allergies = _format_section_value("Allergies", patient_data.get("allergies"))
    social_history = _format_section_value("Social History", patient_data.get("social_history"))
    vitals = _format_section_value("Vitals", patient_data.get("vitals"))
    symptoms = _format_section_value("Symptoms", patient_data.get("symptoms"))
    report_sources = []
    if transcription.strip():
        report_sources.append("Audio dictation")
    if uploaded_reports_text.strip():
        report_sources.append("Uploaded medical reports")

    clinical_report = (
        f"Patient Details\n"
        f"- Name: {patient_name}\n"
        f"- Age: {patient_age}\n\n"
        f"Clinical Summary\n"
        f"- Presenting symptoms: {symptoms}\n"
        f"- Medical history: {medical_history}\n"
        f"- Surgical history: {surgical_history}\n"
        f"- Current medications: {current_medications}\n"
        f"- Stopped medications: {stopped_medications}\n"
        f"- Allergies: {allergies}\n"
        f"- Social history: {social_history}\n"
        f"- Vitals: {vitals}\n\n"
        f"Assessment\n"
        f"- A structured AI summary was not returned, so this draft is built from the available clinical intake and uploaded records.\n"
        f"- Sources reviewed: {', '.join(report_sources) if report_sources else 'None'}"
    )

    action_plan = (
        "1. Review the patient history, symptoms, and vitals manually before sign-off.\n"
        "2. Correlate the dictation with any uploaded prior reports for missing longitudinal details.\n"
        "3. Escalate urgently if the chest pain, shortness of breath, or abnormal vitals suggest a time-sensitive condition.\n"
        "4. Add any physician-specific recommendations or medication changes before exporting the final PDF."
    )

    return clinical_report, action_plan


# Initialize state
st.session_state.setdefault("report_ready", False)
st.session_state.setdefault("generated_clinical_report", "")
st.session_state.setdefault("generated_action_plan", "")
st.session_state.setdefault("pdf_bytes", None)
st.session_state.setdefault("clinical_intelligence", {})

# ---- UI: Audio Input ----
st.markdown("### 1. Patient Audio Input")

# Create tabs for different input methods
tab1, tab2 = st.tabs(["📁 Upload Audio File", "🎙️ Record Live Audio"])

with tab1:
    uploaded_file = st.file_uploader("Upload Audio Dictation", type=list(SUPPORTED_AUDIO_TYPES))

with tab2:
    recorded_file = st.audio_input("Record Patient Dictation")

# ---- UI: Previous Records ----
st.markdown("### 2. Previous Medical Reports (Optional)")
uploaded_reports = st.file_uploader(
    "Upload prior medical reports or text files",
    type=list(SUPPORTED_REPORT_TYPES),
    accept_multiple_files=True,
    help="These files will be added to the search and report-generation context.",
)

# Determine which audio source to use priority to recorded if both exist, or uploaded if selected
audio_data = recorded_file if recorded_file else uploaded_file

if audio_data is not None:
    st.audio(audio_data)
    
    if st.button("▶️ Process Patient Record", type="primary"):
        uploaded_reports_text, uploaded_report_names = _combine_uploaded_reports(uploaded_reports)
        temp_audio_path = None
        
        # Save audio data temporarily because Whisper needs a file path
        audio_suffix = os.path.splitext(getattr(audio_data, "name", "recording.wav"))[1] or ".wav"
        audio_bytes = audio_data.getvalue() if hasattr(audio_data, "getvalue") else audio_data.read()
        with tempfile.NamedTemporaryFile(delete=False, suffix=audio_suffix) as temp_audio_file:
            temp_audio_file.write(audio_bytes)
            temp_audio_path = temp_audio_file.name

        try:
            # ---- STEP 1: Transcription ----
            with st.spinner("🎙️ Step 1: Transcribing Audio using Whisper (Groq)..."):
                transcription = transcribe_audio(temp_audio_path)
            
            if not transcription:
                st.error("Failed to transcribe audio. Please try again.")
                st.stop()

            with st.expander("📝 View Raw Transcription", expanded=False):
                st.info(transcription)

            # ---- STEP 2: Clinical Data Extraction ----
            with st.spinner("🧠 Step 2: Extracting Clinical Entities..."):
                patient_json_string = extract_clinical_data(transcription)
                
                # Try to parse into a python dictionary for pretty UI display
                try:
                    patient_data = json.loads(patient_json_string)
                except json.JSONDecodeError:
                    patient_data = {"error": "Could not parse JSON", "raw_output": patient_json_string}
            
            # Display Extracted Data as nicely formatted JSON
            st.markdown("### 🗂️ Structured Patient Record")
            col1, col2 = st.columns([1, 1])
            with col1:
                st.json(patient_data)
            
            # ---- STEP 3: RAG Search ----
            with st.spinner("📚 Step 3: Searching Medical Guidelines (Pinecone)..."):
                search_query = build_retrieval_query(patient_data, transcription, uploaded_reports_text)

                retriever = setup_retriever()
                retrieved_docs = retriever.invoke(search_query)

            with col2:
                with st.expander("🔍 Retrieved Medical Context (Oxford Handbook)", expanded=False):
                    for i, doc in enumerate(retrieved_docs, start=1):
                        st.markdown(f"**Document {i}:**")
                        st.caption(f"{doc.page_content[:400]}...")
                        st.divider()

                if uploaded_report_names:
                    with st.expander("🗂️ Uploaded Medical Reports", expanded=False):
                        st.caption(", ".join(uploaded_report_names))
                        if uploaded_reports_text:
                            st.text_area("Combined uploaded report text", value=uploaded_reports_text, height=220, disabled=True)

            # ---- STEP 4: Clinical Report Synthesis ----
            with st.spinner("⚕️ Step 4: Synthesizing Final Clinical Report..."):
                report_payload = generate_clinical_report(patient_json_string, retrieved_docs, uploaded_reports_text)

            clinical_report_text = _safe_text(report_payload.get("clinical_report", ""))
            action_plan_text = _safe_text(report_payload.get("action_plan", ""))

            if not clinical_report_text and not action_plan_text:
                clinical_report_text, action_plan_text = _build_fallback_report(
                    patient_data,
                    transcription,
                    uploaded_reports_text,
                )
                st.info("The model response was incomplete, so a structured draft was generated from the extracted clinical data instead.")

            clinical_intelligence = build_clinical_intelligence(
                patient_data,
                transcription,
                uploaded_reports_text,
                report_payload,
            )

            st.markdown("### 🚦 Clinical Intelligence")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.metric("Triage", clinical_intelligence.get("triage_label", "Routine"))
            with c2:
                st.metric("Severity", f"{clinical_intelligence.get('red_flags', {}).get('severity_score', 0)}/100")
            with c3:
                st.metric("Confidence", f"{clinical_intelligence.get('confidence', {}).get('confidence_score', 0)}/100")

            red_flags = clinical_intelligence.get("red_flags", {})
            if red_flags.get("red_flags"):
                st.warning("Red flags detected")
                for flag in red_flags.get("red_flags", []):
                    st.caption(f"• {flag.get('reason', '')}")
            else:
                st.success("No explicit red flags detected from the available text and vitals.")

            st.caption(red_flags.get("recommendation", ""))

            with st.expander("Confidence signals", expanded=False):
                for signal in clinical_intelligence.get("confidence", {}).get("signals", []):
                    st.caption(f"• {signal}")

            st.session_state["report_ready"] = True
            st.session_state["generated_clinical_report"] = clinical_report_text or "No clinical report was returned."
            st.session_state["generated_action_plan"] = action_plan_text or "No action plan was returned."
            st.session_state["pdf_bytes"] = None
            st.session_state["latest_patient_data"] = patient_data
            st.session_state["latest_uploaded_report_names"] = uploaded_report_names
            st.session_state["clinical_intelligence"] = clinical_intelligence

            st.success("Clinical report generated. Review and edit it below before exporting to PDF.")

        except Exception as e:
            st.error(f"An error occurred during processing: {e}")
        
        finally:
            # Clean up the temporary audio file
            if temp_audio_path and os.path.exists(temp_audio_path):
                os.remove(temp_audio_path)

if st.session_state.get("report_ready"):
    st.markdown("### 3. Review and Edit Report")
    st.caption("Edit the English clinical summary and action plan before generating the final PDF.")

    st.session_state.setdefault("generated_clinical_report_editor", st.session_state.get("generated_clinical_report", ""))
    st.session_state.setdefault("generated_action_plan_editor", st.session_state.get("generated_action_plan", ""))

    st.text_area(
        "Clinical Report",
        key="generated_clinical_report_editor",
        height=280,
    )
    st.text_area(
        "Action Plan",
        key="generated_action_plan_editor",
        height=240,
    )

    st.session_state["generated_clinical_report"] = st.session_state.get("generated_clinical_report_editor", "")
    st.session_state["generated_action_plan"] = st.session_state.get("generated_action_plan_editor", "")

    if st.button("📄 Build Final PDF"):
        try:
            pdf_bytes = _make_pdf_bytes(
                st.session_state.get("latest_patient_data", {}),
                st.session_state.get("generated_clinical_report", ""),
                st.session_state.get("generated_action_plan", ""),
                st.session_state.get("clinical_intelligence", {}),
            )
            st.session_state["pdf_bytes"] = pdf_bytes
        except Exception as pdf_error:
            st.error(f"Failed to generate PDF: {pdf_error}")

    if st.session_state.get("pdf_bytes"):
        patient_name = st.session_state.get("latest_patient_data", {}).get("patient_name", "patient")
        safe_patient_name = "".join(character for character in str(patient_name) if character.isalnum() or character in (" ", "_", "-"))
        safe_patient_name = safe_patient_name.strip().replace(" ", "_") or "patient"
        st.download_button(
            "⬇️ Download Final PDF",
            data=st.session_state["pdf_bytes"],
            file_name=f"VoxMed_Report_{safe_patient_name}.pdf",
            mime="application/pdf",
        )
