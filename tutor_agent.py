import os
import json
from google import genai
from pypdf import PdfReader
import pyttsx3

# 1. INITIALIZE GEMINI
client = genai.Client(api_key=os.environ.get("AQ.Ab8RN6L084KMOMm_UuGXWdsdPkEICT7IPSbLdznclK7h-R_09g"))

# 2. VOICE ENGINE
engine = pyttsx3.init()
engine.setProperty('rate', 170)

def speak(text):
    print(f"\n[Agent]: {text}\n")
    engine.say(text)
    engine.runAndWait()

# 3. SUBJECT-SPECIFIC MEMORY
def get_memory_file(subject):
    return f"memory_{subject.lower().replace(' ', '_')}.json"

def load_memory(subject):
    mem_file = get_memory_file(subject)
    if os.path.exists(mem_file):
        with open(mem_file, "r") as f:
            return json.load(f)
    return {"mastered_topics": [], "weak_topics": []}

def save_memory(subject, memory):
    mem_file = get_memory_file(subject)
    with open(mem_file, "w") as f:
        json.dump(memory, f, indent=2)

# 4. LOAD NOTES FOR A SPECIFIC SUBJECT
def get_available_subjects(base_folder="notes"):
    if not os.path.exists(base_folder):
        os.makedirs(base_folder)
    # Find all subdirectories
    subjects = [d for d in os.listdir(base_folder) if os.path.isdir(os.path.join(base_folder, d))]
    return subjects

def load_subject_notes(subject, base_folder="notes"):
    subject_folder = os.path.join(base_folder, subject)
    combined_notes = ""

    for filename in os.listdir(subject_folder):
        file_path = os.path.join(subject_folder, filename)
        if filename.endswith(".pdf"):
            reader = PdfReader(file_path)
            for page in reader.pages:
                combined_notes += f"\n[File: {filename}]\n" + (page.extract_text() or "")
        elif filename.endswith(".txt"):
            with open(file_path, "r", encoding="utf-8") as f:
                combined_notes += f"\n[File: {filename}]\n" + f.read()

    return combined_notes

def ask_gemini(prompt):
    response = client.models.generate_content(
        model="gemini-3.8-flash",
        contents=prompt
    )
    return response.text.strip()

# 5. AGENT RUN LOOP
def run_tutor_agent():
    subjects = get_available_subjects()

    if not subjects:
        print("⚠️ No subject folders found inside 'notes/'!")
        print("Example: Create a folder 'notes/Biology' and add your PDFs there.")
        input("Press Enter to exit...")
        return

    # Choose subject
    print("====================================================")
    print("📚 AVAILABLE SUBJECTS:")
    for idx, subj in enumerate(subjects, 1):
        print(f"  [{idx}] {subj}")
    print("====================================================")
    
    choice = input("Select a subject number (e.g. 1): ")
    try:
        selected_subject = subjects[int(choice) - 1]
    except Exception:
        selected_subject = subjects[0]

    notes_content = load_subject_notes(selected_subject)
    if not notes_content.strip():
        print(f"⚠️ No files found in 'notes/{selected_subject}'! Put some PDFs there.")
        input("Press Enter to exit...")
        return

    memory = load_memory(selected_subject)
    
    speak(f"Welcome back to {selected_subject}. You have {len(memory['mastered_topics'])} topics mastered.")
    speak("What topic would you like to study, or should I quiz your weak spots?")
    topic = input("[You] (or Win+H to speak): ")

    while True:
        agent_prompt = f"""
        You are a voice tutor for the subject: {selected_subject}.
        STUDY NOTES:
        {notes_content}

        STUDENT RECORD:
        Weak spots: {memory['weak_topics']}
        Mastered: {memory['mastered_topics']}

        CURRENT FOCUS: {topic}

        Instructions:
        1. Keep responses concise and conversational (spoken aloud).
        2. Explain 1 concept from the notes simply, then ask ONE question.
        3. Do NOT use markdown asterisks (*).
        """
        agent_response = ask_gemini(agent_prompt)
        speak(agent_response)

        user_answer = input("[You] (or Win+H): ")
        if user_answer.lower() in ["exit", "quit", "stop"]:
            speak(f"Great work on {selected_subject}! Progress saved.")
            save_memory(selected_subject, memory)
            break

        # Evaluator
        evaluator_prompt = f"""
        Evaluate student response based on the notes:
        NOTES: {notes_content}
        QUESTION: {agent_response}
        ANSWER: {user_answer}

        Output ONLY JSON:
        {{
            "is_correct": true or false,
            "concept_tested": "name of concept",
            "feedback": "1 sentence feedback"
        }}
        """
        eval_raw = ask_gemini(evaluator_prompt).replace("```json", "").replace("```", "").strip()
        
        try:
            eval_data = json.loads(eval_raw)
            speak(eval_data["feedback"])

            concept = eval_data["concept_tested"]
            if eval_data["is_correct"]:
                if concept not in memory["mastered_topics"]:
                    memory["mastered_topics"].append(concept)
                if concept in memory["weak_topics"]:
                    memory["weak_topics"].remove(concept)
            else:
                if concept not in memory["weak_topics"]:
                    memory["weak_topics"].append(concept)
            
            save_memory(selected_subject, memory)
        except Exception:
            pass

if __name__ == "__main__":
    run_tutor_agent()