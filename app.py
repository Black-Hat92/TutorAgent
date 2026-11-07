import os
import json
import io
import re
import logging
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from gtts import gTTS

# 1. SETUP PAGE CONFIG & MUTE WARNINGS
st.set_page_config(page_title="AI Tutor Agent", page_icon="🎓", layout="wide")
logging.getLogger("pypdf").setLevel(logging.ERROR)

# 2. INITIALIZE GROQ CLIENT
GROQ_API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
client = Groq(api_key=GROQ_API_KEY)

# 3. DIRECTORY & SMART LOADER
BASE_NOTES_DIR = "notes"
if not os.path.exists(BASE_NOTES_DIR):
    os.makedirs(BASE_NOTES_DIR)

def get_subjects():
    dirs = [d for d in os.listdir(BASE_NOTES_DIR) if os.path.isdir(os.path.join(BASE_NOTES_DIR, d))]
    direct_files = [f for f in os.listdir(BASE_NOTES_DIR) if f.endswith(('.pdf', '.txt'))]
    if direct_files:
        dirs.insert(0, "Main Notes")
    if not dirs:
        os.makedirs(os.path.join(BASE_NOTES_DIR, "General_Study"), exist_ok=True)
        dirs = ["General_Study"]
    return dirs

@st.cache_data(show_spinner="Indexing all chapters and textbooks...")
def load_all_course_files(subject):
    """Loads all PDF files into a dictionary: {filename: text}"""
    folder = BASE_NOTES_DIR if subject == "Main Notes" else os.path.join(BASE_NOTES_DIR, subject)
    docs = {}
    
    if os.path.exists(folder):
        for file in sorted(os.listdir(folder)):
            p = os.path.join(folder, file)
            text = ""
            if file.endswith(".pdf"):
                try:
                    reader = PdfReader(p)
                    # Read up to 80 pages per file
                    total_pages = min(len(reader.pages), 80)
                    for i in range(total_pages):
                        page_text = reader.pages[i].extract_text()
                        if page_text:
                            text += f"\n[File: {file} - Page {i+1}]\n" + page_text
                except Exception as e:
                    print(f"Error reading {file}: {e}")
            elif file.endswith(".txt"):
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    text = f.read()
                    
            if text.strip():
                docs[file] = text
    return docs

def get_relevant_content(docs_dict, user_message):
    """
    Intelligently routes to the exact chapter file requested.
    Never lets a large textbook push out specific chapter slides.
    """
    targeted_content = ""
    msg = user_message.lower()

    # 1. Match specific chapter requests
    for filename, content in docs_dict.items():
        clean_name = filename.lower().replace("_", " ").replace("-", " ")
        if any(term in msg for term in ["chapter 1", "ch 1", "intro"]) and "chapter 1" in clean_name:
            targeted_content += f"\n=== {filename} ===\n" + content[:12000]
        elif any(term in msg for term in ["chapter 2", "ch 2", "key concept"]) and "chapter 2" in clean_name:
            targeted_content += f"\n=== {filename} ===\n" + content[:12000]
        elif any(term in msg for term in ["chapter 3", "ch 3", "dimension"]) and "chapter 3" in clean_name:
            targeted_content += f"\n=== {filename} ===\n" + content[:12000]
        elif any(term in msg for term in ["chapter 5", "ch 5", "practice", "management"]) and "chapter 5" in clean_name:
            targeted_content += f"\n=== {filename} ===\n" + content[:12000]

    # 2. If no specific chapter matched, include excerpts from all slide decks first
    if not targeted_content:
        for filename, content in docs_dict.items():
            if "chapter" in filename.lower():
                targeted_content += f"\n=== SLIDES: {filename} ===\n" + content[:5000]

    # 3. Supplement with textbook context
    for filename, content in docs_dict.items():
        if "axelos" in filename.lower() or "edition" in filename.lower():
            targeted_content += f"\n=== TEXTBOOK: {filename} ===\n" + content[4000:12000]
            break

    return targeted_content[:20000]

def get_memory_file(subject):
    return f"memory_{subject.lower().replace(' ', '_')}.json"

def load_memory(subject):
    mem_file = get_memory_file(subject)
    if os.path.exists(mem_file):
        with open(mem_file, "r") as f:
            return json.load(f)
    return {"mastered_topics": [], "weak_topics": []}

def save_memory(subject, memory):
    with open(get_memory_file(subject), "w") as f:
        json.dump(memory, f, indent=2)

def get_clean_models():
    """Fetches chat models, filtering out restricted/audio models."""
    try:
        raw_models = client.models.list().data
        valid = [
            m.id for m in raw_models 
            if not any(bad in m.id.lower() for bad in ["whisper", "guard", "canopylabs", "vision"])
        ]
        return valid if valid else ["llama-3.1-8b-instant", "mixtral-8x7b-32768"]
    except Exception:
        return ["llama-3.1-8b-instant", "mixtral-8x7b-32768"]

def ask_ai(prompt, model_name):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model=model_name,
        temperature=0.6,
    )
    return chat_completion.choices[0].message.content.strip()

# 4. CLEAN AUDIO ENGINE (No dashes, pipes, or asterisks)
def clean_text_for_speech(text):
    """Removes all Markdown symbols, emojis, and tags so TTS sounds human."""
    cleaned = re.sub(r'[*#_~`>|\-]', ' ', text)
    cleaned = re.sub(r'[^\w\s.,?!]', ' ', cleaned)
    cleaned = re.sub(r'\[.*?\]', '', cleaned)
    cleaned = re.sub(r'\s+', ' ', cleaned).strip()
    return cleaned

def text_to_speech(text):
    try:
        spoken_text = clean_text_for_speech(text)
        tts = gTTS(text=spoken_text[:3500], lang='en', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        return fp.getvalue()
    except Exception as e:
        print(f"Audio Error: {e}")
        return None

# 5. SIDEBAR CONTROLS
st.sidebar.title("Study Dashboard")

available_models = get_clean_models()
selected_model = st.sidebar.selectbox("Choose AI Model", available_models)

subjects = get_subjects()
selected_subject = st.sidebar.selectbox("Select Subject", subjects)

# Load course documents into memory
docs_dict = load_all_course_files(selected_subject)
total_files = len(docs_dict)

if total_files == 0:
    st.sidebar.warning("⚠️ No materials found in this folder!")
else:
    st.sidebar.success(f"📚 {total_files} Course Files Active in Memory")

# Language Style
language_mode = st.sidebar.selectbox(
    "Tutor Language Style",
    [
        "English with Ghanaian Real-Life Context", 
        "Twi (Asante / Akuapem) + English Mix",
        "Ghanaian Pidgin (Informal & Fun)",
        "Ga + English Mix",
        "Ewe + English Mix"
    ]
)

# Upload File Directly via UI
st.sidebar.subheader("Upload Material")
uploaded_file = st.sidebar.file_uploader(f"Add PDF/TXT", type=["pdf", "txt"])
if uploaded_file is not None:
    dest_dir = BASE_NOTES_DIR if selected_subject == "Main Notes" else os.path.join(BASE_NOTES_DIR, selected_subject)
    with open(os.path.join(dest_dir, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.cache_data.clear()
    st.sidebar.success(f"Saved {uploaded_file.name}! Please refresh.")

# Voice Toggle
enable_voice = st.sidebar.checkbox("🔊 Read answers out loud", value=True)

# Clear Chat Button
if st.sidebar.button("Clear Chat / New Session", use_container_width=True):
    st.session_state.messages = [
        {"role": "assistant", "content": f"Session cleared! What would you like to study next in **{selected_subject}**?"}
    ]
    st.rerun()

# Memory & Progress Tracking
memory = load_memory(selected_subject)
st.sidebar.divider()
st.sidebar.subheader("Mastery Progress")
col1, col2 = st.sidebar.columns(2)
col1.metric("Mastered", len(memory["mastered_topics"]))
col2.metric("Weak Spots", len(memory["weak_topics"]))

# 6. MAIN CHAT INTERFACE
st.title(f"🎓 AI Tutor: {selected_subject}")
st.caption(f"Connected to `{selected_model}` | Style: `{language_mode}`")

# Session State Setup
if "messages" not in st.session_state or st.session_state.get("last_subject") != selected_subject:
    st.session_state.last_subject = selected_subject
    st.session_state.messages = [
        {"role": "assistant", "content": f"Hello! Ready to study **{selected_subject}**. Which chapter should we tackle?"}
    ]

# Render Message History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if "audio" in msg and msg["audio"]:
            st.audio(msg["audio"], format="audio/mp3")

# Chat Input Field
user_input = st.chat_input("Type your response (or press Win + H to speak)...")

if user_input:
    # Show user message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Route specifically to the requested chapter
    relevant_notes = get_relevant_content(docs_dict, user_input)

    if not relevant_notes.strip():
        bot_response = "⚠️ No notes found for this subject. Please upload your PDFs in the sidebar."
        with st.chat_message("assistant"):
            st.markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
    else:
        with st.spinner("Tutor is preparing the lesson..."):
            # Deep Chapter Instruction + Synoptic Bridge Prompt
            combined_prompt = f"""
            You are a master academic tutor teaching {selected_subject}.
            Language/Style: {language_mode}

            LECTURE NOTES (PRIMARY TRUTH):
            {relevant_notes}

            STUDENT TRACKER:
            Mastered: {memory['mastered_topics']}
            Weak Spots: {memory['weak_topics']}

            STUDENT MESSAGE: "{user_input}"

            =======================================
            CORE TEACHING MANDATE:
            =======================================
            1. TEACH THOROUGHLY FIRST (80% FOCUS):
               - Your primary job is to TEACH the material from the notes deeply and accurately.
               - Explain the mechanisms, terms, definitions, and principles completely so the student understands the full concept.
               - Do not give surface-level summaries; give actual substance from the slides.

            2. THE CHAPTER BRIDGE (20% FOCUS):
               - After explaining the concept, show the natural link: How does this build upon previous foundations, and how does it set up what comes next?

            3. REAL-WORLD GHANAIAN SCENARIO:
               - Ground the concept in everyday Ghanaian life (e.g., MoMo transactions, trotro transport, chop bar operations, dumsor management, Makola/Kejetia commerce, local enterprise IT).

            4. STRUCTURE YOUR LESSON INTO THESE 4 CLEAN PARTS:
               ### Detailed Concept Breakdown
               (Deep, step-by-step teaching grounded in the lecture notes)

               ### The Chapter Connection
               (1 or 2 clear sentences showing how this ties back to previous chapters or foundations)

               ### Real-Life Scenario
               (A vivid Ghanaian example making the concept intuitive)

               ### Quick Application Challenge
               (ONE scenario-based question to verify comprehension before moving on)
            """

            raw_reply = ask_ai(combined_prompt, selected_model)

            # Auto-update memory from the response tag if present
            if "[RESULT:" in raw_reply:
                try:
                    tag_line = raw_reply.split("]")[0].replace("[RESULT:", "").strip()
                    status, concept = [x.strip() for x in tag_line.split("-", 1)]
                    if "CORRECT" in status:
                        if concept not in memory["mastered_topics"]: memory["mastered_topics"].append(concept)
                        if concept in memory["weak_topics"]: memory["weak_topics"].remove(concept)
                    else:
                        if concept not in memory["weak_topics"]: memory["weak_topics"].append(concept)
                    save_memory(selected_subject, memory)
                except Exception:
                    pass
                bot_response = raw_reply.split("]", 1)[-1].strip()
            else:
                bot_response = raw_reply

            # Audio generation (Clean speech without dashes/pipes)
            audio_fp = text_to_speech(bot_response) if enable_voice else None

            # Display response
            with st.chat_message("assistant"):
                st.markdown(bot_response)
                if audio_fp:
                    st.audio(audio_fp, format="audio/mp3", autoplay=True)

            # Append to session history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": bot_response,
                "audio": audio_fp
            })