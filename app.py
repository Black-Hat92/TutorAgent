import os
import json
import io
import logging
import streamlit as st
from groq import Groq
from pypdf import PdfReader
from gtts import gTTS

# 1. SETUP PAGE CONFIG & MUTE WARNINGS
st.set_page_config(page_title="AI Tutor Agent", page_icon="🎓", layout="wide")
logging.getLogger("pypdf").setLevel(logging.ERROR)

# ✅ SAFE (Never hardcode keys in public code):
API_KEY = st.secrets.get("GROQ_API_KEY", os.environ.get("GROQ_API_KEY", ""))
client = Groq(api_key=API_KEY)

# 3. DIRECTORY & CACHED LOADER
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

@st.cache_data(show_spinner="📖 Loading notes into memory...")
def load_subject_notes(subject):
    """Caches the text so it loads instantly without freezing."""
    folder = BASE_NOTES_DIR if subject == "Main Notes" else os.path.join(BASE_NOTES_DIR, subject)
    combined = ""
    if os.path.exists(folder):
        for file in os.listdir(folder):
            p = os.path.join(folder, file)
            if file.endswith(".pdf"):
                try:
                    reader = PdfReader(p)
                    # Limit to first 60 pages so it never freezes
                    total_pages = min(len(reader.pages), 60)
                    for i in range(total_pages):
                        page_text = reader.pages[i].extract_text()
                        if page_text:
                            combined += f"\n[File: {file} - Page {i+1}]\n" + page_text
                except Exception as e:
                    print(f"Error reading {file}: {e}")
            elif file.endswith(".txt"):
                with open(p, "r", encoding="utf-8", errors="ignore") as f:
                    combined += f"\n[File: {file}]\n" + f.read()
    return combined

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
    """Fetches general chat models, filtering out restricted/audio models."""
    try:
        raw_models = client.models.list().data
        valid = [
            m.id for m in raw_models 
            if not any(bad in m.id.lower() for bad in ["whisper", "guard", "canopylabs", "vision"])
        ]
        return valid if valid else ["mixtral-8x7b-32768", "gemma2-9b-it"]
    except Exception:
        return ["mixtral-8x7b-32768", "gemma2-9b-it"]

def ask_ai(prompt, model_name):
    chat_completion = client.chat.completions.create(
        messages=[
            {"role": "user", "content": prompt}
        ],
        model=model_name,
        temperature=0.6,
    )
    return chat_completion.choices[0].message.content.strip()

def text_to_speech(text):
    try:
        clean_text = text.replace("*", "").replace("#", "")
        # Change [:350] to [:2500] so it reads the full answer!
        tts = gTTS(text=clean_text[:2500], lang='en', slow=False)
        fp = io.BytesIO()
        tts.write_to_fp(fp)
        return fp.getvalue()
    except Exception as e:
        print(f"Audio Error: {e}")
        return None

# 4. SIDEBAR DASHBOARD
st.sidebar.title("📚 Study Dashboard")

# Model Selector (Filters out restricted models)
available_models = get_clean_models()
selected_model = st.sidebar.selectbox("🤖 Choose AI Model", available_models)

# Add New Subject Field directly in UI
new_sub = st.sidebar.text_input("➕ Create New Subject")
if st.sidebar.button("Add Subject") and new_sub:
    os.makedirs(os.path.join(BASE_NOTES_DIR, new_sub.strip().replace(" ", "_")), exist_ok=True)
    st.sidebar.success(f"Created {new_sub}!")
    st.rerun()

# Subject Selector
subjects = get_subjects()
selected_subject = st.sidebar.selectbox("Select Subject", subjects)

study_phase = st.sidebar.radio(
    "🎯 Current Study Phase",
    ["1. Chapter Lecture (Full Overview)", "2. Comprehensive Chapter Exam", "3. Quick Review / Q&A"],
    index=0
)

# Load cached notes
notes_content = load_subject_notes(selected_subject)
notes_len = len(notes_content)

if notes_len == 0:
    st.sidebar.warning("⚠️ No materials found in this folder!")
else:
    st.sidebar.success(f"📖 Loaded {notes_len:,} characters of notes.")

# Upload File Directly via Web UI
st.sidebar.subheader("Upload Material")
uploaded_file = st.sidebar.file_uploader(f"Add PDF/TXT", type=["pdf", "txt"])
if uploaded_file is not None:
    dest_dir = BASE_NOTES_DIR if selected_subject == "Main Notes" else os.path.join(BASE_NOTES_DIR, selected_subject)
    with open(os.path.join(dest_dir, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.getbuffer())
    st.cache_data.clear()
    st.sidebar.success(f"Saved {uploaded_file.name}! Please refresh.")

# Voice Toggle
enable_voice = st.sidebar.checkbox("🔊 Read answers out loud", value=False)
# Clear Chat Button
if st.sidebar.button("🗑️ Clear Chat / New Session", use_container_width=True):
    st.session_state.messages = [
        {"role": "assistant", "content": f"Session cleared! What would you like to study next in **{selected_subject}**?"}
    ]
    st.rerun()

# Memory & Progress Tracking
memory = load_memory(selected_subject)
st.sidebar.divider()
st.sidebar.subheader("📈 Mastery Progress")
col1, col2 = st.sidebar.columns(2)
col1.metric("Mastered", len(memory["mastered_topics"]))
col2.metric("Weak Spots", len(memory["weak_topics"]))

with st.sidebar.expander("View Weak Topics"):
    for t in memory["weak_topics"]:
        st.write(f"- ⚠️ {t}")

with st.sidebar.expander("View Mastered Topics"):
    for t in memory["mastered_topics"]:
        st.write(f"- ✅ {t}")

# 5. MAIN CHAT INTERFACE
st.title(f"🎓 AI Tutor: {selected_subject}")
st.caption(f"Powered by `{selected_model}`")

# Session State Setup
if "messages" not in st.session_state or st.session_state.get("last_subject") != selected_subject:
    st.session_state.last_subject = selected_subject
    st.session_state.messages = [
        {"role": "assistant", "content": f"Hello! Ready to study **{selected_subject}**. What topic or chapter should we tackle today?"}
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
    # 1. Display User Message
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    if not notes_content.strip():
        bot_response = "⚠️ No notes found in this subject folder yet! Please upload a PDF in the sidebar."
        with st.chat_message("assistant"):
            st.markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
    else:
        with st.spinner("Tutor is thinking..."):
            combined_prompt = f"""
            You are a master personal tutor for {selected_subject}.
            STUDY NOTES:
            {notes_content[:15000]}

            STUDENT INPUT: "{user_input}"
            CURRENT PHASE: {study_phase}
            WEAK SPOTS: {memory['weak_topics']}

            ===========================
            BEHAVIOR RULES FOR CURRENT PHASE:
            ===========================

            IF PHASE IS "1. Chapter Lecture (Full Overview)":
            - DO NOT ask quiz questions.
            - Extract the 3 to 4 core Learning Outcomes from the chapter notes.
            - Teach the full chapter cohesively from start to finish.
            - Explain every major theory using a vivid, everyday REAL-LIFE SCENARIO (cooking, cars, tech, sports, business).
            - End your response with: "We have covered all outcomes for this chapter! When you are ready to test your knowledge, switch the sidebar to '2. Comprehensive Chapter Exam'."

            IF PHASE IS "2. Comprehensive Chapter Exam":
            - Review the chapter's learning outcomes and generate a 3-question Scenario Exam.
            - Do not ask definition questions. Put the student inside realistic dilemmas where they must apply the concepts.
            - If the student just submitted answers, evaluate each one, mark concepts as Mastered or Weak, and explain any errors simply.

            IF PHASE IS "3. Quick Review / Q&A":
            - Answer the student's specific question directly with a practical analogy.
            """

            raw_reply = ask_ai(combined_prompt, selected_model)

            # Auto-update memory from the single response tag
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
                # Clean up the tag so the student sees clean formatting
                bot_response = raw_reply.split("]", 1)[-1].strip()
            else:
                bot_response = raw_reply

            # Audio generation
            audio_fp = text_to_speech(bot_response) if enable_voice else None

            # Display response
            with st.chat_message("assistant"):
                st.markdown(bot_response)
                if audio_fp:
                    st.audio(audio_fp, format="audio/mp3", autoplay=True)

            # Save to chat history
            st.session_state.messages.append({
                "role": "assistant", 
                "content": bot_response,
                "audio": audio_fp
            })
