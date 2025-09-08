import os
import io
import time
import torch
import numpy as np
import sounddevice as sd
import soundfile as sf
import whisperx
from playsound3 import playsound
from TTS.api import TTS
import pickle
from sentence_transformers import SentenceTransformer
import faiss
from llama_cpp import Llama

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True

# --- UTILS ---
def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}")

# --- LLM INIT ---
start = time.time()
log("Loading LLM model...")
llm = Llama(model_path="./Qwen3-1.7B-Q4_K_M.gguf", n_ctx=2048, n_threads=8, n_gpu_layers=40)
log(f"LLM loaded in {time.time()-start:.2f}s")

start = time.time()
warmup = llm.create_chat_completion(
    messages=[
        {"role": "system", "content": "You are an assistant who perfectly describes images."},
        {"role": "user", "content": "Describe this image in detail please."}
    ]
)
log(f"Warmup completed in {time.time()-start:.2f}s")

if warmup:
    log("LLM model warmed up successfully.")
else:
    log("Failed to load LLM model.")
    exit(1)

# --- GLOBAL STATE ---
LAST_IMAGE = None
MISSION_PARAMETERS = ""
TRANSCRIPTION_FILE = "transcription.txt"
RAG_FOLDER = "RAG"
INDEX_FILE = "faiss_index.bin"
DOCS_FILE = "documents.pkl"
CHUNK_SIZE = 500

# --- MODEL LOADING ---
start = time.time()
device = "cuda" if torch.cuda.is_available() else "cpu"
log(f"Using device: {device}")

log("Loading TTS model...")
tts = TTS(model_name="tts_models/en/vctk/vits").to(device)
log("TTS loaded.")

log("Loading SentenceTransformer retriever...")
retriever = SentenceTransformer('all-MiniLM-L6-v2', device=device)
log("Retriever loaded.")

log("Loading WhisperX STT model...")
STT = whisperx.load_model("large-v3-turbo", device=device, language="en")
log(f"All models loaded in {time.time()-start:.2f}s")

# --- RAG SETUP ---
def load_and_index_documents():
    global documents, faiss_index
    start = time.time()
    documents = []
    if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
        log("Loading existing FAISS index and documents...")
        faiss_index = faiss.read_index(INDEX_FILE)
        with open(DOCS_FILE, 'rb') as f:
            documents = pickle.load(f)
        log(f"Loaded {len(documents)} documents in {time.time()-start:.2f}s")
        return
    log("Generating FAISS index from scratch...")
    os.makedirs(RAG_FOLDER, exist_ok=True)
    for filename in os.listdir(RAG_FOLDER):
        if filename.endswith(".txt"):
            with open(os.path.join(RAG_FOLDER, filename), 'r', encoding='utf-8') as f:
                text = f.read().strip()
                chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
                documents.extend(chunks)
    if not documents:
        log(f"No .txt files found in {RAG_FOLDER}. Using empty knowledge base.")
        documents = ["No relevant information available."]
    embeddings = retriever.encode(documents, convert_to_numpy=True, show_progress_bar=True)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings.astype(np.float32))
    faiss.write_index(faiss_index, INDEX_FILE)
    with open(DOCS_FILE, 'wb') as f:
        pickle.dump(documents, f)
    log(f"Indexed {len(documents)} documents in {time.time()-start:.2f}s")

documents = []
faiss_index = None
load_and_index_documents()

# --- RAG RETRIEVAL ---
def RAG_retrieval(query, top_k=2):
    log(f"RAG retrieval for query: {query}")
    start = time.time()
    query_embedding = retriever.encode([query], convert_to_numpy=True)[0]
    _, indices = faiss_index.search(np.array([query_embedding]).astype(np.float32), k=min(top_k, len(documents)))
    retrieved_docs = [documents[idx] for idx in indices[0] if idx < len(documents)]
    log(f"RAG retrieved {len(retrieved_docs)} docs in {time.time()-start:.2f}s")
    return retrieved_docs

def put_chat_in_RAG(user_msg, ai_reply):
    global documents, faiss_index
    log("Adding chat to RAG store...")
    chat_text = f"User: {user_msg}\nAI: {ai_reply}"
    documents.append(chat_text)
    embedding = retriever.encode([chat_text], convert_to_numpy=True)
    embedding = np.array(embedding).astype(np.float32)
    if embedding.ndim == 1:
        embedding = embedding.reshape(1, -1)
    faiss_index.add(embedding)
    faiss.write_index(faiss_index, INDEX_FILE)
    with open(DOCS_FILE, "wb") as f:
        pickle.dump(documents, f)
    log("Chat added to RAG.")

# --- SYSTEM PROMPT ---
def get_system_prompt():
    base_path = os.path.join(os.path.dirname(__file__), 'SystemPrompt.txt')
    log(f"Loading system prompt from {base_path}")
    with open(base_path) as f:
        return f.read().strip() + "\n\nMission Parameters:\n" + MISSION_PARAMETERS

# --- AUDIO RECORDING ---
def record_audio(threshold=0.02, silence_duration=1.0, sample_rate=16000, max_duration=10):
    log("Recording audio...")
    start = time.time()
    recording = []
    silence_start = None
    started = False

    def callback(indata, *_):
        nonlocal started, silence_start
        volume = np.linalg.norm(indata)
        if not started and volume > threshold:
            started = True
            log("Recording started (threshold passed).")
        if started:
            recording.append(indata.copy())
            if volume < threshold:
                if silence_start is None:
                    silence_start = time.time()
                elif time.time() - silence_start > silence_duration:
                    raise sd.CallbackStop
            else:
                silence_start = None

    with sd.InputStream(callback=callback, channels=1, samplerate=sample_rate):
        try:
            sd.sleep(int(max_duration * 1000))
        except sd.CallbackStop:
            log("Recording stopped due to silence.")
    
    if not recording:
        log("No audio recorded.")
        return None

    audio = np.concatenate(recording, axis=0)
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    buf.seek(0)
    log(f"Audio recorded in {time.time()-start:.2f}s")
    return buf

# --- WHISPER TRANSCRIPTION ---
def transcribe(buf):
    log("Starting transcription...")
    start = time.time()
    buf.seek(0)
    audio, sr = sf.read(buf)
    audio = audio.astype(np.float32)
    transcription = ""
    result = STT.transcribe(audio, batch_size=16)
    transcription = " ".join([seg["text"].strip() for seg in result["segments"]])
    log(f"Transcription done in {time.time()-start:.2f}s: {transcription}")
    return transcription

# --- TEXT TO SPEECH ---
def play_tts(text):
    log(f"Generating TTS for: {text}")
    start = time.time()
    tts.tts_to_file(text=text, speaker="p298", file_path="output.wav")
    playsound("output.wav")
    os.remove("output.wav")
    log(f"TTS played in {time.time()-start:.2f}s")

# --- AI RESPONSE WITH RAG USING vLLM ---
def ai_response(prompt):
    log(f"Generating AI response for: {prompt}")
    start = time.time()
    system_prompt = get_system_prompt()
    retrieved_docs = RAG_retrieval(prompt, top_k=2)
    prompt_text = "\n\nRelevant Information:\n" + "\n".join(retrieved_docs) + "\n\nUser: " + prompt + " /no_think"
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text}
    ]
    response = llm.create_chat_completion(messages, max_tokens=512, temperature=0.7)
    play_tts(response['choices'][0]['message']['content'])
    log(f"AI response generated in {time.time()-start:.2f}s")
    return response

# --- MAIN CHAT LOOP ---
def main_loop():
    os.system("clear")
    play_tts("Atlas Ready.")
    log("Atlas Ready.")
    while True:
        try:
            audio_buf = record_audio()
            if not audio_buf:
                continue
            user_input = transcribe(audio_buf)
            if not user_input:
                continue
            log(f"User: {user_input}")
            response = ai_response(user_input)
            print(response)
            os.system("clear")
        except Exception as e:
            log(f"[Error]: {e}")

# --- MAIN ENTRY POINT ---
if __name__ == '__main__':
    main_loop()
