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

# 6. MAIN WORKSPACE & CHAT
st.title(f"🎓 AI Tutor: {selected_subject}")
st.caption(f"Model: `{selected_model}` | Language: `{language_mode}`")

# Track active quiz questions in session state
if "active_question" not in st.session_state:
    st.session_state.active_question = None
if "active_concept" not in st.session_state:
    st.session_state.active_concept = None
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

# ==============================================================================
# 🎯 DEDICATED QUIZ & GRADING WORKSPACE (Separate from Chat)
# ==============================================================================
if st.session_state.active_question:
    st.divider()
    with st.container():
        st.markdown(f"### 🎯 Challenge Question:\n**{st.session_state.active_question}**")
        
        with st.form("quiz_answer_form"):
            quiz_answer = st.text_area(
                "Write your answer here (apply the concept using real scenarios):",
                placeholder="Type your explanation or press Win + H to speak...",
                key="quiz_input_box"
            )
            submit_quiz = st.form_submit_button("Submit Answer for Grading 📝", use_container_width=True)

        if submit_quiz and quiz_answer.strip():
            with st.spinner("Grading your answer against the lecture notes..."):
                grading_prompt = f"""
                You are an objective, encouraging academic evaluator for {selected_subject}.
                Language style: {language_mode}

                QUESTION ASKED:
                "{st.session_state.active_question}"

                CONCEPT TESTED:
                "{st.session_state.active_concept}"

                STUDENT'S ANSWER:
                "{quiz_answer}"

                Task:
                1. Explicitly state whether the answer is:
                   ✅ FULLY CORRECT
                   ⚠️ PARTIALLY CORRECT
                   ❌ INCORRECT
                2. Explain WHY based strictly on the lecture notes principles.
                3. If they missed something, give a quick, memorable Ghanaian scenario analogy to fix the mental model.
                4. Conclude with 1 sentence telling them what to study next.
                """
                grade_feedback = ask_ai(grading_prompt, selected_model)
                grade_audio = text_to_speech(grade_feedback) if enable_voice else None

                # Update memory tracker
                concept = st.session_state.active_concept or "Recent Concept"
                if "FULLY CORRECT" in grade_feedback.upper():
                    if concept not in memory["mastered_topics"]: memory["mastered_topics"].append(concept)
                    if concept in memory["weak_topics"]: memory["weak_topics"].remove(concept)
                elif "INCORRECT" in grade_feedback.upper():
                    if concept not in memory["weak_topics"]: memory["weak_topics"].append(concept)
                save_memory(selected_subject, memory)

                # Append user answer and tutor grade to chat log
                st.session_state.messages.append({"role": "user", "content": f"**My Answer to Challenge:**\n{quiz_answer}"})
                st.session_state.messages.append({"role": "assistant", "content": f"### 📊 Evaluation Result\n\n{grade_feedback}", "audio": grade_audio})
                
                # Clear active question so student can proceed
                st.session_state.active_question = None
                st.session_state.active_concept = None
                st.rerun()

# ==============================================================================
# 💬 INQUIRY & LESSON CHAT (For asking questions or requesting chapters)
# ==============================================================================
user_input = st.chat_input("Ask a question, request a chapter (e.g. 'Chapter 2')...")

if user_input:
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    relevant_notes = get_relevant_content(docs_dict, user_input)

    if not relevant_notes.strip():
        bot_response = "⚠️ No notes found for this subject. Please upload your PDFs in the sidebar."
        with st.chat_message("assistant"):
            st.markdown(bot_response)
        st.session_state.messages.append({"role": "assistant", "content": bot_response})
    else:
        with st.spinner("Tutor is preparing the lesson..."):
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
            TEACHING RULES:
            =======================================
            1. TEACH THOROUGHLY:
               - Explain the mechanisms, terms, and definitions completely from the notes.
            2. REAL-WORLD GHANAIAN SCENARIO:
               - Ground the concept in everyday Ghanaian life (MoMo, trotros, chop bar, dumsor, Makola trade).
            3. CRUCIAL CHALLENGE TAG:
               - At the very end of your response, write the challenge question using this EXACT format on its own line:
               [CHALLENGE: Concept Name | Your scenario question here]

            Structure your reply cleanly:
            ### 📖 Detailed Concept Breakdown
            (Deep, step-by-step teaching)

            ### 🔗 The Chapter Connection
            (How this links to earlier foundations)

            ### 🇬🇭 Real-Life Scenario
            (Intuitive Ghanaian example)

            [CHALLENGE: Concept Name | Your scenario question here]
            """

            raw_reply = ask_ai(combined_prompt, selected_model)

            # Extract challenge question to populate dedicated quiz box
            if "[CHALLENGE:" in raw_reply:
                try:
                    challenge_part = raw_reply.split("[CHALLENGE:")[1].split("]")[0]
                    c_name, c_q = [x.strip() for x in challenge_part.split("|", 1)]
                    st.session_state.active_concept = c_name
                    st.session_state.active_question = c_q
                    # Clean out the tag so chat looks pristine
                    clean_reply = raw_reply.split("[CHALLENGE:")[0].strip()
                except Exception:
                    clean_reply = raw_reply
            else:
                clean_reply = raw_reply

            audio_fp = text_to_speech(clean_reply) if enable_voice else None

            st.session_state.messages.append({
                "role": "assistant", 
                "content": clean_reply,
                "audio": audio_fp
            })
            st.rerun()