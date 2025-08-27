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
from ollama import chat
from ollama import ChatResponse

messages = [
        {"role": "system", "content": "test"},
        {"role": "user", "content": "this is a test"}
    ]

output = chat(model="qwen3:1.7b", messages=messages)

# --- GLOBAL STATE ---
LAST_IMAGE = None
MISSION_PARAMETERS = ""
TRANSCRIPTION_FILE = "transcription.txt"
RAG_FOLDER = "RAG"
INDEX_FILE = "faiss_index.bin"
DOCS_FILE = "documents.pkl"
CHUNK_SIZE = 500

# --- MODEL LOADING ---
device = "cuda" if torch.cuda.is_available() else "cpu"
tts = TTS(model_name="tts_models/en/vctk/vits").to(device)
retriever = SentenceTransformer('all-MiniLM-L6-v2', device=device)
STT = whisperx.load_model("large-v3-turbo", device=device, language="en")

# --- RAG SETUP ---
def load_and_index_documents():
    global documents, faiss_index
    documents = []
    if os.path.exists(INDEX_FILE) and os.path.exists(DOCS_FILE):
        faiss_index = faiss.read_index(INDEX_FILE)
        with open(DOCS_FILE, 'rb') as f:
            documents = pickle.load(f)
        print("Loaded existing FAISS index and documents.")
        return
    os.makedirs(RAG_FOLDER, exist_ok=True)
    for filename in os.listdir(RAG_FOLDER):
        if filename.endswith(".txt"):
            with open(os.path.join(RAG_FOLDER, filename), 'r', encoding='utf-8') as f:
                text = f.read().strip()
                chunks = [text[i:i+CHUNK_SIZE] for i in range(0, len(text), CHUNK_SIZE)]
                documents.extend(chunks)
    if not documents:
        print(f"No .txt files found in {RAG_FOLDER}. Using empty knowledge base.")
        documents = ["No relevant information available."]
    embeddings = retriever.encode(documents, convert_to_numpy=True, show_progress_bar=True)
    dimension = embeddings.shape[1]
    faiss_index = faiss.IndexFlatL2(dimension)
    faiss_index.add(embeddings.astype(np.float32))
    faiss.write_index(faiss_index, INDEX_FILE)
    with open(DOCS_FILE, 'wb') as f:
        pickle.dump(documents, f)
    print("Generated and saved FAISS index and documents.")

documents = []
faiss_index = None
load_and_index_documents()

# --- RAG RETRIEVAL ---
def RAG_retrieval(query, top_k=2):
    query_embedding = retriever.encode([query], convert_to_numpy=True)[0]
    _, indices = faiss_index.search(np.array([query_embedding]).astype(np.float32), k=min(top_k, len(documents)))
    retrieved_docs = [documents[idx] for idx in indices[0] if idx < len(documents)]
    return retrieved_docs

def put_chat_in_RAG(user_msg, ai_reply):
    global documents, faiss_index
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

# --- SYSTEM PROMPT ---
def get_system_prompt():
    base_path = os.path.join(os.path.dirname(__file__), 'SystemPrompt.txt')
    with open(base_path) as f:
        return f.read().strip() + "\n\nMission Parameters:\n" + MISSION_PARAMETERS

# --- AUDIO RECORDING ---
def record_audio(threshold=0.02, silence_duration=1.0, sample_rate=16000, max_duration=10):
    recording = []
    silence_start = None
    started = False

    def callback(indata, *_):
        nonlocal started, silence_start
        volume = np.linalg.norm(indata)
        if not started and volume > threshold:
            started = True
            print("Recording started.")
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
            pass

    if not recording:  # nothing captured
        return None

    audio = np.concatenate(recording, axis=0)

    # Save to proper WAV in BytesIO
    buf = io.BytesIO()
    sf.write(buf, audio, sample_rate, format="WAV")
    buf.seek(0)
    return buf

# --- WHISPER TRANSCRIPTION ---
def transcribe(buf):
    buf.seek(0)
    audio, sr = sf.read(buf)
    result = STT.transcribe(audio, batch_size=16)
    transcription = result["text"].strip()
    with open(TRANSCRIPTION_FILE, "w") as f:
        f.write(transcription)
    return transcription

# --- TEXT TO SPEECH ---
def play_tts(text):
    tts.tts_to_file(text=text, speaker="p298", file_path="output.wav")
    playsound("output.wav")
    os.remove("output.wav")

# --- AI RESPONSE WITH RAG USING vLLM ---
def ai_response(prompt):
    system_prompt = get_system_prompt()
    retrieved_docs = RAG_retrieval(prompt, top_k=2)
    prompt_text = "\n\nRelevant Information:\n" + "\n".join(retrieved_docs) + "\n\nUser: " + prompt
    messages = [
        {"role": "system", "content": system_prompt},
        {"role": "user", "content": prompt_text}
    ]
    try:
        output = chat(model="qwen3:1.7b", messages=messages, max_tokens=512, temperature=0.7)
        reply = (output['message']['content'])
        print("Retrieved Context:\n", retrieved_docs)
        print("AI Response:", reply)
        play_tts(reply)
        put_chat_in_RAG(prompt, reply)
        return reply
    except Exception as e:
        print(f"Error generating response: {e}")
        return None

# --- MAIN CHAT LOOP ---
def main_loop():
    play_tts("Atlas Ready.")
    print("Atlas Ready.")
    while True:
        try:
            audio_buf = record_audio()
            if not audio_buf:
                continue
            user_input = transcribe(audio_buf)
            if not user_input:
                continue
            print(f"User: {user_input}")
            response = ai_response(user_input)
            if response:
                print(response)
        except Exception as e:
            print(f"[Error]: {e}")

# --- MAIN ENTRY POINT ---
if __name__ == '__main__':
    main_loop()
