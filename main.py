import os
import json
from dotenv import load_dotenv

# Import functions from our modules
from voxmed.intake_module import transcribe_audio, extract_clinical_data
from voxmed.rag_agent import setup_retriever, generate_action_plan, build_retrieval_query

def main():
    load_dotenv()
    
    # 1. AUDIO INPUT
    # Update this path to an actual audio file you have
    audio_file_path = "tests_data/test_audio.mp3"
    
    if not os.path.exists(audio_file_path):
        print(f"[!] Please place a test audio file named '{audio_file_path}' in the tests_data folder.")
        return

    print("==================================================")
    print(" 🎙️ STEP 1: VOICE TO TEXT (Intake Module)")
    print("==================================================")
    transcription = transcribe_audio(audio_file_path)
    print(f"\n[Transcription Result]:\n{transcription}\n")
    if not transcription:
        print("[!] Flow stopped: Could not transcribe audio.")
        return

    print("==================================================")
    print(" 🧠 STEP 2: EXTRACT CLINICAL DATA (NLP)")
    print("==================================================")
    patient_json = extract_clinical_data(transcription)
    print(f"\n[Structured Patient Data]:\n{patient_json}\n")

    print("==================================================")
    print(" 📚 STEP 3: RAG RESEARCH (Pinecone Search)")
    print("==================================================")
    
    # Extract symptoms or main keywords to search Pinecone
    try:
        parsed_data = json.loads(patient_json)
        search_query = build_retrieval_query(parsed_data, transcription, "")
    except json.JSONDecodeError:
        search_query = transcription

    if not search_query.strip():
        search_query = "General medical checkup"

    print(f"[*] Querying Pinecone for: '{search_query}'")
    retriever = setup_retriever()
    retrieved_docs = retriever.invoke(search_query)
    
    print("\n[Retrieved Context (Top 3)]:")
    for i, doc in enumerate(retrieved_docs, start=1):
        print(f"--- Doc {i} ---\n{doc.page_content[:200]}...\n")

    print("==================================================")
    print(" ⚕️ STEP 4: SYNTHESIZE & ACTION PLAN (Agentic Output)")
    print("==================================================")
    action_plan = generate_action_plan(patient_json, retrieved_docs)
    
    print("\n[Final Output]:\n")
    print(action_plan)


if __name__ == "__main__":
    main()
