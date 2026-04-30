# VoxMedAI 🩺🎙️
**An Intelligent Voice-Driven Medical Assistant and Data Analysis System**

[![GitHub Repo](https://img.shields.io/badge/GitHub-Repository-blue.svg)](https://github.com/MarwanIbrahimAmin/VoxMedAI)

## About The Project
**VoxMedAI** is a comprehensive AI-powered system designed to assist doctors by transforming voice notes into accurate medical diagnoses, backed by semantic search across trusted medical literature. The system automatically generates medical prescriptions, QR codes, and ICD-10 classifications to streamline healthcare workflows and enhance efficiency.

---

## 📑 Documentation

This document contains the comprehensive documentation for the graduation project, categorized into four main phases:

### 1. Project Planning & Management
This project was managed using Agile software development methodologies to ensure flexibility and continuous delivery.
* **Objectives:** Build an automated medical assistant that reduces the time spent on writing medical reports and increases diagnostic accuracy.
* **Task Management (Sprints):** The project was divided into key phases:
  1. Integrating Speech-to-Text models.
  2. Building the Vector Database and RAG (Retrieval-Augmented Generation) pipeline.
  3. Designing and implementing the Multi-Agent AI system.
  4. Developing the user interface using Streamlit.
* **Risk Management:** The risk of LLM hallucinations was heavily considered and mitigated by implementing a "Critic Agent" to act as a Safety Guard for all generated diagnoses.

### 2. Literature Review
The project relies on state-of-the-art research and technologies in the field of Medical AI:
* **Speech Processing Models:** Utilization of the **OpenAI Whisper** model, which has proven highly efficient in transcribing complex medical terminology compared to traditional models.
* **Retrieval-Augmented Generation (RAG):** Using vector databases like **ChromaDB** for semantic search within Medical PDFs, ensuring that the language model relies on trusted, evidence-based medical information.
* **Multi-Agent Architecture:** Designing a system that delegates specific roles to specialized AI agents—a Retriever, a Generator, and a Critic—to ensure the highest standards of medical accuracy and patient safety.

### 3. Requirements Gathering
**Functional Requirements:**
* Ability to capture and process voice recordings from the doctor.
* High-accuracy conversion of audio to raw text.
* Extraction of structured data (e.g., Symptoms) in JSON format.
* Semantic search in medical references using the extracted symptoms.
* Generation of a draft medical diagnosis with automated critical review.
* Automatic generation of a Prescription (PDF), a QR Code, and an ICD-10 code.
* Displaying all outputs in an interactive User Interface (Dashboard).

**Non-Functional Requirements:**
* **Reliability & Safety:** No diagnosis can be finalized without passing the validation of the Critic Agent.
* **Performance:** Fast response times for audio processing and real-time generation of outputs.
* **Usability:** A simple, intuitive, and distraction-free interface tailored for medical professionals.

### 4. System Analysis & Design
The system architecture was designed based on a logical and secure data flow. It consists of the following core components:

1. **Input & Audio Processing:**
   * **Doctor Voice Input:** Receives the audio file from the user.
   * **OpenAI Whisper Model:** Transcribes the audio file into Raw Text.

2. **Data Structuring:**
   * **LLM API (e.g., Llama 3):** Takes the raw text and converts it into Structured Data (JSON) to clearly isolate the "Symptoms".

3. **RAG & Agent System:**
   * **Retriever Agent:** Takes the symptoms and performs a semantic search.
   * **Vector Database (ChromaDB):** Contains Medical PDFs and returns the "Relevant chunks" matching the symptoms.
   * **Generator Agent:** Combines the relevant chunks with the extracted symptoms to create a "Draft diagnosis".

4. **Review & Safety (Safety Guard):**
   * **Critic Agent:** Acts as a safety guard by evaluating the draft diagnosis. If it requires correction, it sends "Feedback" back to the Generator. If it is accurate and safe, it outputs an "Approved diagnosis".

5. **Automation Logic & Outputs:**
   * The Automation Logic receives the approved diagnosis and triggers three parallel processes:
     * **LLM for ICD-10:** Makes an API call to determine the correct international disease classification code.
     * **QR Code Generator:** Uses a Python library to generate a scannable QR code.
     * **Prescription Generator:** Uses a PDF library to format and create the final medical prescription document.

6. **User Interface:**
   * **Streamlit Dashboard:** A unified dashboard that collects and displays all final outputs (ICD-10 code, QR Code, and Prescription PDF) to the doctor.

---

## 🚀 How to Run the Project

```bash
# 1. Clone the repository
git clone [https://github.com/MarwanIbrahimAmin/VoxMedAI.git](https://github.com/MarwanIbrahimAmin/VoxMedAI.git)

# 2. Navigate to the directory
cd VoxMedAI

# 3. Install the required dependencies
pip install -r requirements.txt

# 4. Run the Streamlit application
streamlit run app.py
