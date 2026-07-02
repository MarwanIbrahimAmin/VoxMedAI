# 🩺 VoxMedAI -- An Intelligent Voice-Driven Medical Assistant and Data Analysis System

**VoxMedAI** is a production-grade, voice-driven clinical intelligence platform that transforms physician dictation audio into structured medical diagnoses, evidence-based clinical reports, and automated healthcare artifacts. The system leverages OpenAI Whisper for high-fidelity medical transcription, performs semantic retrieval across indexed medical literature via a RAG (Retrieval-Augmented Generation) pipeline, and orchestrates a Multi-Agent AI architecture to produce critically reviewed diagnoses, automated prescriptions (PDF), QR codes, and ICD-10 classification codes -- all surfaced through a unified Streamlit dashboard.

> **VoxMedAI is designed as a clinical decision-support tool.** It augments physician workflows and reduces administrative burden; it does not replace clinical judgment.

---

## 1. Project Planning & Management

**Methodology:** The project follows Agile development practices with iterative sprint cycles, enabling continuous delivery and rapid integration of feedback at every stage.

**Core Objective:** Build an automated medical assistant that minimizes clinical reporting time while maximizing diagnostic accuracy through evidence-based AI augmentation.

### Sprint Breakdown

| Sprint | Focus Area | Deliverable |
|---|---|---|
| Sprint 1 | Speech-to-Text Integration | Robust medical audio transcription pipeline with vocabulary priming |
| Sprint 2 | Vector Database & RAG Pipeline Engineering | Indexed medical literature with semantic retrieval capabilities |
| Sprint 3 | Multi-Agent AI System Implementation | Retriever, Generator, and Critic agent orchestration with safety gates |
| Sprint 4 | Streamlit UI Development & Deployment | Production dashboard with PDF export, QR code, and ICD-10 display |

### Risk Management -- LLM Hallucination Mitigation

Large Language Models can produce clinically plausible but medically incorrect outputs. VoxMedAI addresses this through a dedicated **Critic Agent** that operates as a strict Safety Guard within the multi-agent pipeline. Every generated diagnosis is passed through this validation gate before reaching the clinician. The Critic Agent:

- Cross-references the generated diagnosis against retrieved evidence chunks from trusted medical sources.
- Flags unsupported claims, missing differential considerations, and potential contraindications.
- Returns structured feedback to the Generator Agent for revision if the draft fails validation criteria.
- Only grants an **Approved Diagnosis** status when the output meets evidence-alignment and safety thresholds.

---

## 2. Literature Review & Technology Stack

### 🎙️ Speech Processing

The transcription layer uses **OpenAI Whisper** for robust speech-to-text conversion optimized for medical terminology. Whisper's multilingual architecture handles Arabic and mixed-language dictations, while medical vocabulary priming ensures accurate recognition of drug names, anatomical terms, and clinical abbreviations.

### 📚 Retrieval-Augmented Generation (RAG)

To ensure evidence-based outputs and eliminate unchecked generative hallucinations, VoxMedAI implements a full RAG pipeline:

- **Vector Database:** ChromaDB stores dense embeddings of trusted medical PDFs (including clinical handbooks and pharmacological references).
- **Embedding Model:** Domain-specific biomedical embeddings ensure high semantic fidelity for medical queries.
- **Retrieval Strategy:** Top-K relevant chunks are retrieved at inference time and injected into the generation context, grounding all outputs in verified medical literature.

### 🤖 Multi-Agent Architecture

VoxMedAI employs a role-based delegation pattern across three specialized agents:

| Agent | Role | Responsibility |
|---|---|---|
| **Retriever Agent** | Evidence Sourcing | Queries the ChromaDB vector store and returns the most semantically relevant medical context for the current patient case. |
| **Generator Agent** | Diagnosis Drafting | Consumes structured patient data and retrieved evidence to synthesize a draft clinical diagnosis, report, and action plan. |
| **Critic Agent** | Safety Guard | Evaluates the draft diagnosis for clinical accuracy, evidence alignment, and safety. Sends revision feedback or grants approval. |

This architecture enforces a separation of concerns where no single agent controls the full output path, reducing the risk of unvalidated content reaching the clinician.

---

## 3. Requirements Gathering

### Functional Requirements

| ID | Requirement | Description |
|---|---|---|
| FR-01 | Voice Capture & Processing | Accept audio recordings in common formats (MP3, WAV, M4A, OGG) from physician dictation sessions. |
| FR-02 | High-Accuracy Transcription | Transcribe medical audio with domain-specific vocabulary priming to minimize terminology errors. |
| FR-03 | Structured Data Extraction | Parse raw transcription text into a strict JSON schema extracting symptoms, vitals, medications, allergies, medical history, and patient demographics. |
| FR-04 | Semantic Reference Search | Query the vector database to retrieve relevant clinical evidence from indexed medical PDFs. |
| FR-05 | Draft Diagnosis Generation | Synthesize a clinically coherent draft diagnosis grounded in patient data and retrieved evidence. |
| FR-06 | Critical Review Gate | Subject every draft diagnosis to Critic Agent validation before presenting to the clinician. |
| FR-07 | Prescription Generation | Automatically generate a formatted Prescription document (PDF) from the approved diagnosis. |
| FR-08 | QR Code Generation | Produce a scannable QR code encoding key diagnostic and prescription metadata. |
| FR-09 | ICD-10 Classification | Assign the appropriate ICD-10 classification code to the approved diagnosis and display it on the dashboard. |
| FR-10 | Unified Dashboard | Surface all outputs (transcription, structured data, diagnosis, prescription, QR code, ICD-10 code) in a single Streamlit interface. |

### Non-Functional Requirements

| Category | Requirement | Detail |
|---|---|---|
| **Reliability & Safety** | Critic Agent Validation Gate | No diagnosis reaches the user without passing the Critic Agent's evidence-alignment and safety checks. |
| **Performance** | Real-Time Output Generation | End-to-end pipeline (audio to dashboard) executes within acceptable clinical workflow timeframes. |
| **Usability** | Professional Interface | Distraction-free, clean dashboard design that integrates naturally into physician workflows without cognitive overload. |
| **Maintainability** | Modular Package Architecture | Core logic is encapsulated in an importable Python package (`voxmed/`) with clean separation from UI and tooling layers. |

---

## 4. System Analysis & Design

### 📊 System Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────┐
│                        INPUT PIPELINE                               │
│                                                                     │
│   Voice Recording ──▶ OpenAI Whisper ──▶ Raw Transcription Text    │
│                              │                                      │
│                              ▼                                      │
│                    LLM API (Structured Extraction)                   │
│                              │                                      │
│                              ▼                                      │
│                   Structured JSON (Symptoms,                        │
│                   Vitals, Medications, History)                      │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     RAG & MULTI-AGENT FLOW                          │
│                                                                     │
│   ┌──────────────────┐    Relevant     ┌──────────────────┐        │
│   │ Retriever Agent  │───Chunks───────▶│ Generator Agent  │        │
│   │ (ChromaDB Query) │                 │ (Draft Diagnosis)│        │
│   └──────────────────┘                 └────────┬─────────┘        │
│                                                  │                  │
│                                                  ▼                  │
│                                        ┌──────────────────┐        │
│                                        │  Critic Agent    │        │
│                                        │  (Safety Guard)  │        │
│                                        └────────┬─────────┘        │
│                                                  │                  │
│                                    ┌─────────────┼─────────────┐   │
│                                    │             │             │    │
│                              Feedback Loop   Approved     Rejected │
│                              (Revise Draft)  Diagnosis    (Halt)   │
└─────────────────────────────────────┬───────────────────────────────┘
                                      │
                                      ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     AUTOMATION LOGIC                                │
│                                                                     │
│   Approved Diagnosis triggers parallel processes:                   │
│                                                                     │
│   ├──▶ LLM API ──────────────▶ ICD-10 Classification Code         │
│   ├──▶ Python QR Library ────▶ QR Code (Scannable)                 │
│   └──▶ PDF Library ──────────▶ Prescription Document (PDF)         │
│                                                                     │
└──────────────────────────────┬──────────────────────────────────────┘
                               │
                               ▼
┌─────────────────────────────────────────────────────────────────────┐
│                     USER INTERFACE                                   │
│                                                                     │
│              Unified Streamlit Dashboard                             │
│   ┌──────────┬──────────┬──────────┬──────────┬──────────┐         │
│   │Transcript│ Symptoms │Diagnosis │   Rx PDF │ QR Code  │         │
│   │  Panel   │  (JSON)  │+ ICD-10  │ Download │ Display  │         │
│   └──────────┴──────────┴──────────┴──────────┴──────────┘         │
│                                                                     │
└─────────────────────────────────────────────────────────────────────┘
```

### Data Flow Summary

| Stage | Input | Process | Output |
|---|---|---|---|
| Transcription | Audio file | OpenAI Whisper with medical vocabulary priming | Raw text |
| Extraction | Raw text | LLM-driven structured parsing | JSON (symptoms, vitals, meds, history) |
| Retrieval | Structured symptoms | Retriever Agent queries ChromaDB | Relevant medical evidence chunks |
| Generation | Symptoms + Evidence | Generator Agent synthesizes diagnosis | Draft diagnosis |
| Validation | Draft diagnosis | Critic Agent safety review | Approved or revision-flagged diagnosis |
| Automation | Approved diagnosis | Parallel LLM + QR + PDF generation | ICD-10 code, QR code, prescription PDF |
| Presentation | All artifacts | Streamlit rendering | Unified clinical dashboard |

---

## 5. 📂 Production Codebase Architecture

The codebase has been refactored into a production-grade Python package layout that cleanly separates core library logic, standalone tooling, data science artifacts, and test fixtures.

```
VoxMedAI/
├── app.py                        # Main Streamlit entrypoint (Hugging Face Spaces deployment)
├── main.py                       # CLI entrypoint for pipeline dry runs
├── requirements.txt              # Pinned dependency manifest with version bounds
├── runtime.txt                   # Python version lock (python-3.10.20)
├── .env.example                  # Environment variable template (API keys)
├── .gitignore                    # VCS exclusion rules
├── README.md                     # This document
│
├── voxmed/                       # Core package — all importable library logic
│   ├── __init__.py               # Package marker
│   ├── config.py                 # Centralized constants, model names, clinical thresholds
│   ├── enums.py                  # Triage, Severity, and Confidence enumerations
│   ├── intake_module.py          # Audio transcription and clinical entity extraction
│   ├── schema_validation.py      # Payload normalization, validation, Arabic-to-English translation
│   ├── clinical_intelligence.py  # Red-flag detection, severity scoring, triage classification
│   ├── rag_agent.py              # Vector retrieval, report synthesis, query construction
│   └── prompt_templates.py       # LLM prompt definitions and builder functions
│
├── scripts/                      # Standalone CLI tools (not imported by core package)
│   └── validate_schema.py        # Offline schema validation and normalization utility
│
├── notebooks/                    # Data science and prototyping
│   └── voxmed_preprocessing.ipynb  # Vector index preparation and medical PDF ingestion
│
└── tests_data/                   # Standardized test assets
    └── test_audio.mp3            # Reference audio fixture for pipeline smoke tests
```

### Design Rationale

| Directory | Purpose | Boundary Rule |
|---|---|---|
| `voxmed/` | All importable production logic | Uses intra-package imports only (`from voxmed.config import ...`). Never references `scripts/` or `notebooks/`. |
| `scripts/` | Standalone CLI wrappers | Imports from `voxmed/` but is never imported by it. Prevents circular dependencies. |
| `notebooks/` | Exploratory and preprocessing work | Isolated from production code paths. Used for one-time tasks like vector index creation. |
| `tests_data/` | Test fixtures | Single version-controlled location for mock assets. No test data lives alongside source code. |

---

## 6. 🚀 Installation & Verification Flow

### Prerequisites

- Python 3.10+
- Active API keys for the LLM provider and vector database service

### Clone & Install

```bash
git clone https://github.com/MarwanIbrahimAmin/VoxMedAI.git
cd VoxMedAI

python3 -m venv .venv
source .venv/bin/activate

pip install -r requirements.txt
```

### Configure Environment

```bash
cp .env.example .env
# Edit .env and insert your API keys
```

For Hugging Face Spaces deployment, set the keys under **Settings > Secrets** in the Space dashboard instead.

### Local Syntax Verification

Validate that all modules compile cleanly without executing any API calls:

```bash
python3 -m py_compile app.py main.py voxmed/*.py
```

A zero-exit-code result confirms clean compilation across the entire codebase.

### Run the Application

```bash
streamlit run app.py
```

The dashboard opens at `http://localhost:8501` by default.

---

## Team & Acknowledgments

This project is developed as part of the **DEPI Round 4 -- Microsoft Machine Learning Engineering Track**. Refer to the repository license file for usage terms.
