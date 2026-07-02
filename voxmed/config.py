GROQ_MODEL_NAME = "llama-3.3-70b-versatile"
WHISPER_MODEL_NAME = "whisper-large-v3"
EMBEDDING_MODEL_NAME = "NeuML/pubmedbert-base-embeddings"

PINECONE_INDEX_NAME = "voxmed-ai"
PINECONE_TOP_K = 3

DEFAULT_SYSTEM_PROMPT = "You are a clinical decision support system."
DEFAULT_AUDIO_LANGUAGE = "ar"

SUPPORTED_AUDIO_TYPES = ("mp3", "wav", "m4a", "ogg")
SUPPORTED_REPORT_TYPES = ("txt", "md", "csv", "json", "pdf")

PDF_TITLE = "VoxMed Medical Report"
PDF_SOURCE_TEXT = "Audio dictation reviewed with optional uploaded records"

HIGH_RISK_SYMPTOMS = (
	"chest pain",
	"shortness of breath",
	"difficulty breathing",
	"dyspnea",
	"syncope",
	"fainting",
	"unilateral weakness",
	"facial droop",
	"slurred speech",
	"confusion",
	"altered mental status",
	"seizure",
	"severe bleeding",
	"suicidal",
)

CARDIOVASCULAR_RISK_FACTORS = (
	"hypertension",
	"high blood pressure",
	"coronary artery disease",
	"ischemic heart disease",
	"myocardial infarction",
	"stroke",
	"smoking",
	"diabetes",
	"aspirin",
	"antiplatelet",
)

CARDIAC_CONTEXT_HINTS = (
	"acute coronary syndrome",
	"myocardial infarction",
	"unstable angina",
	"ECG",
	"troponin",
	"risk stratification",
	"differential diagnosis",
	"emergency chest pain evaluation",
)

URGENT_TRIAGE_SCORE = 85
HIGH_SEVERITY_SCORE = 80
MEDIUM_SEVERITY_SCORE = 40
HIGH_CONFIDENCE_SCORE = 80
MODERATE_CONFIDENCE_SCORE = 50
