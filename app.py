import streamlit as st
import requests
import torch
import json
from html import escape
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
 

# ---------------- CONFIG ---------------- #
st.set_page_config(page_title="Multilingual Gemini Chatbot", layout="wide")

# Global styles for a fuller UI
st.markdown(
    """
<style>
/* Layout */
.main > div { padding-top: 1rem; }

/* Title */
h1, .stTitle { letter-spacing: 0.2px; }

/* Bigger inputs */
.stTextArea textarea { min-height: 130px; font-size: 1.05rem; }

/* Bigger buttons */
.stButton > button {
  padding: 0.9rem 1.2rem;
  font-size: 1.05rem;
  border-radius: 12px;
}

/* HTML Speak button */
.speak-btn {
  padding: 0.9rem 1.2rem;
  font-size: 1.05rem;
  border-radius: 12px;
  border: 1px solid #e5e7eb;
  background: linear-gradient(180deg, #ffffff, #f6f8fa);
  cursor: pointer;
}
.speak-btn:hover { filter: brightness(0.98); }

/* Chat bubbles */
.chat-bubble { max-width: 78%; padding: 12px 16px; margin: 10px 0; border-radius: 16px; box-shadow: 0 2px 6px rgba(0,0,0,0.06); }
.chat-bubble .label { font-weight: 600; margin-bottom: 4px; opacity: 0.9; }
.chat-bubble .text { font-size: 1.05rem; line-height: 1.55; color: inherit; }
.chat-bubble.user { margin-left: auto; color: #ffffff; background: linear-gradient(135deg, #2563eb, #7c3aed); }
.chat-bubble.bot { background: #f1f5f9; border: 1px solid #cbd5e1; color: #0f172a; }
.chat-bubble.bot .label { color: #0f172a; opacity: 0.9; }

/* Section cards */
.info-card {
  padding: 14px 18px;
  border-radius: 14px;
  background: #fffbeb;            /* soft amber for contrast on light/dark */
  border: 1px solid #f59e0b55;    /* amber border */
  color: #111827;                 /* ensure readable text */
  box-shadow: 0 2px 6px rgba(0,0,0,0.04);
}

/* Chips */
.chips { display: flex; flex-wrap: wrap; gap: 8px; margin-top: 6px; }
.chip { padding: 6px 10px; border-radius: 999px; background: #eef2ff; border: 1px solid #c7d2fe; color: #1f2937; font-size: 0.95rem; }

/* Exercise content styling */
.exercise-section { 
  background: #f8fafc; 
  border: 1px solid #e2e8f0; 
  border-radius: 16px; 
  padding: 20px; 
  margin: 20px 0; 
  box-shadow: 0 2px 8px rgba(0,0,0,0.06);
}
.exercise-title { 
  color: #1e40af; 
  font-size: 1.5rem; 
  font-weight: 700; 
  margin-bottom: 16px; 
  text-align: center;
}
.lesson-card { 
  background: white; 
  border: 1px solid #e5e7eb; 
  border-radius: 12px; 
  padding: 16px; 
  margin: 12px 0; 
  box-shadow: 0 1px 3px rgba(0,0,0,0.1);
}
.lesson-header { 
  color: #059669; 
  font-weight: 600; 
  font-size: 1.1rem; 
  margin-bottom: 8px;
}
.exercise-item { 
  background: #fef3c7; 
  border-left: 4px solid #f59e0b; 
  padding: 12px; 
  margin: 8px 0; 
  border-radius: 0 8px 8px 0;
}
.exercise-question { 
  background: #dbeafe; 
  border-left: 4px solid #3b82f6; 
  padding: 12px; 
  margin: 8px 0; 
  border-radius: 0 8px 8px 0;
}
.exercise-answer { 
  background: #dcfce7; 
  border-left: 4px solid #16a34a; 
  padding: 12px; 
  margin: 8px 0; 
  border-radius: 0 8px 8px 0;
}
</style>
""",
    unsafe_allow_html=True,
)

# Gemini API key
GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
GEMINI_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash:generateContent"

# Device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Supported languages for dropdown (lang_code: display_name)
LANGUAGES = {
    "hin_Deva": "Hindi",
    "pan_Guru": "Punjabi",
    "guj_Gujr": "Gujarati",
    "tam_Taml": "Tamil",
    "tel_Telu": "Telugu",
    "mal_Mlym": "Malayalam",
    "ben_Beng": "Bengali",
    "mar_Deva": "Marathi",
    "kan_Knda": "Kannada",
    "eng_Latn": "English"
}

# Map IndicTrans2 language codes to browser TTS/STT BCP-47 tags
LANG_TO_TTS_TAG = {
    "eng_Latn": "en-US",
    "hin_Deva": "hi-IN",
    "ben_Beng": "bn-IN",
    "guj_Gujr": "gu-IN",
    "mar_Deva": "mr-IN",
    "pan_Guru": "pa-IN",
    "tam_Taml": "ta-IN",
    "tel_Telu": "te-IN",
    "mal_Mlym": "ml-IN",
    "kan_Knda": "kn-IN",
}


@st.cache_data(show_spinner=False)
def get_ui_texts(lang_code: str):
    base = {
        "title": "🌐 Multilingual Chatbot (Indic + Gemini 2.0 Flash)",
        "language_label": "Select Language",
        "message_label": "Type your message (or use 🎤 button below)",
        "send_button": "Send",
        "speak_button": "🎤 Speak",
        "speak_last_button": "🔊 Speak Last Bot Reply",
        "you": "You",
        "bot": "Bot",
    }
    if lang_code == "eng_Latn":
        return base
    # Ask Gemini to translate UI labels in one go, pipe-separated for easy parsing
    target_name = LANGUAGES.get(lang_code, lang_code)
    prompt = (
        f"Translate these UI labels from English to {target_name} ({lang_code}). "
        "Return only the translated values joined by ' || ' in this exact order: "
        "title | language_label | message_label | send_button | speak_button | speak_last_button | you | bot.\n\n"
        f"title: {base['title']}\n"
        f"language_label: {base['language_label']}\n"
        f"message_label: {base['message_label']}\n"
        f"send_button: {base['send_button']}\n"
        f"speak_button: {base['speak_button']}\n"
        f"speak_last_button: {base['speak_last_button']}\n"
        f"you: {base['you']}\n"
        f"bot: {base['bot']}\n"
    )
    result = gemini_chat(prompt)
    parts = [p.strip() for p in str(result).split("||")]
    if len(parts) >= 8:
        return {
            "title": parts[0],
            "language_label": parts[1],
            "message_label": parts[2],
            "send_button": parts[3],
            "speak_button": parts[4],
            "speak_last_button": parts[5],
            "you": parts[6],
            "bot": parts[7],
        }
    return base

# ---------------- TRANSLATION FUNCTIONS ---------------- #
def translate(text, src_lang, tgt_lang):
    """Translate using Gemini only (no local IndicTrans2)."""
    src_name = LANGUAGES.get(src_lang, src_lang)
    tgt_name = LANGUAGES.get(tgt_lang, tgt_lang)
    prompt = (
        f"Translate the following text from {src_name} ({src_lang}) to {tgt_name} ({tgt_lang}).\n"
        "- Output only the translated text.\n"
        "- Do not add quotes or explanations.\n\n"
        f"Text: {text}"
    )
    return gemini_chat(prompt)

# ---------------- GEMINI API CALL ---------------- #
def gemini_chat(prompt):
    payload = {
        "contents": [{"parts": [{"text": prompt}]}]
    }
    response = requests.post(
        f"{GEMINI_URL}?key={GEMINI_API_KEY}",
        json=payload
    )
    if response.status_code == 200:
        try:
            return response.json()["candidates"][0]["content"]["parts"][0]["text"]
        except:
            return "Error: Unexpected Gemini response format."
    else:
        return f"Error {response.status_code}: {response.text}"

# ---------------- DESCRIPTIVE COPY (intro/purpose/tips) ---------------- #
@st.cache_data(show_spinner=False)
def get_copy_texts(lang_code: str):
    base = {
        "hero_subtitle": "Conversational AI that adapts to your language.",
        "intro_paragraph": (
            "This chatbot helps you converse in your preferred Indic language. "
            "Type your message, and we will translate, talk to Gemini, and reply back in the same language."
        ),
        "purpose_title": "Purpose",
        "purpose_text": (
            "Enable smooth multilingual conversations for learning, support, and daily assistance across Indic languages."
        ),
        "how_title": "How it works",
        "how_points": [
            "Choose your language from the dropdown.",
            "Type a message or use voice input (🎤).",
            "We send it to Gemini and return a localized reply.",
            "Use the speaker button to listen to the last response.",
        ],
        "tips_title": "Tips",
        "tips_points": [
            "Ask for translations, explanations, or summaries.",
            "Be clear and concise for best results.",
            "Try different languages to compare outputs.",
        ],
        "privacy_title": "Privacy",
        "privacy_points": [
            "Your inputs are sent to the Gemini API for processing.",
            "Avoid sharing sensitive personal information.",
        ],
        "langs_title": "Supported languages",
    }
    if lang_code == "eng_Latn":
        return base
    target_name = LANGUAGES.get(lang_code, lang_code)
    prompt = (
        f"Translate the following UI copy from English to {target_name} ({lang_code}). Return lines in 'key: value' form. "
        "For list values (how_points, tips_points, privacy_points) join items with ' | '. Do not add extra lines.\n\n"
        f"hero_subtitle: {base['hero_subtitle']}\n"
        f"intro_paragraph: {base['intro_paragraph']}\n"
        f"purpose_title: {base['purpose_title']}\n"
        f"purpose_text: {base['purpose_text']}\n"
        f"how_title: {base['how_title']}\n"
        f"how_points: {' | '.join(base['how_points'])}\n"
        f"tips_title: {base['tips_title']}\n"
        f"tips_points: {' | '.join(base['tips_points'])}\n"
        f"privacy_title: {base['privacy_title']}\n"
        f"privacy_points: {' | '.join(base['privacy_points'])}\n"
        f"langs_title: {base['langs_title']}\n"
    )
    raw = str(gemini_chat(prompt))
    parsed = dict(base)
    for line in raw.splitlines():
        if ":" not in line:
            continue
        k, v = line.split(":", 1)
        key = k.strip()
        val = v.strip()
        if key in ("how_points", "tips_points", "privacy_points"):
            parsed[key] = [p.strip() for p in val.split("|") if p.strip()]
        elif key in parsed:
            parsed[key] = val
    return parsed

# ---------------- UI ---------------- #
# Persist the selected language so the label itself can be localized
_options = list(LANGUAGES.keys())
current_lang_code = st.session_state.get("selected_lang_code", "eng_Latn")
_ui_for_label = get_ui_texts(current_lang_code)
selected_lang_code = st.selectbox(
    _ui_for_label["language_label"],
    options=_options,
    format_func=lambda code: LANGUAGES[code],
    index=_options.index(current_lang_code) if current_lang_code in _options else 0,
)
st.session_state["selected_lang_code"] = selected_lang_code

# Refresh UI strings for the newly selected language
ui = get_ui_texts(selected_lang_code)

st.title(ui["title"])

# Introductory sections
copy = get_copy_texts(selected_lang_code)
st.caption(copy["hero_subtitle"])  # small subtitle under the title
st.write(copy["intro_paragraph"])  # intro paragraph

colA, colB = st.columns(2)
with colA:
    st.subheader(copy["how_title"])
    st.markdown("\n".join([f"- {escape(item)}" for item in copy["how_points"]]))
with colB:
    st.subheader(copy["tips_title"])
    st.markdown("\n".join([f"- {escape(item)}" for item in copy["tips_points"]]))

st.subheader(copy["purpose_title"])
st.markdown(escape(copy["purpose_text"]))

st.subheader(copy["privacy_title"])
st.markdown("\n".join([f"- {escape(item)}" for item in copy["privacy_points"]]))

st.subheader(copy["langs_title"])
chips_html = "<div class='chips'>" + "".join([
    f"<div class='chip'>{escape(LANGUAGES[c])}</div>" for c in LANGUAGES
]) + "</div>"
st.markdown(chips_html, unsafe_allow_html=True)

# ---------------- EXERCISE CONTENT SECTION ---------------- #
st.markdown("## 📚 SpeakGenie English Learning Exercises")

# Initialize session state for exercise tracking
if "exercise_scores" not in st.session_state:
    st.session_state.exercise_scores = {"lesson1": 0, "lesson2": 0}
if "current_lesson" not in st.session_state:
    st.session_state.current_lesson = "lesson1"

# Lesson Navigation Tabs
lesson_tab1, lesson_tab2, progress_tab = st.tabs(["👋 Lesson 1: Greetings", "🙋 Lesson 2: Introduction", "📊 Progress"])

# Lesson 1: Greetings and Hello
with lesson_tab1:
    st.markdown("### 👋 Lesson 1: Greetings and Hello")
    
    # Welcome Section with Interactive Button
    with st.container():
        st.markdown("**🌟 Welcome to SpeakGenie!**")
        col1, col2 = st.columns([3, 1])
        with col1:
            st.markdown("👋 Hi! I'm Genie — your English buddy!")
            st.markdown("📚 Welcome to SpeakGenie — a fun way to learn English!")
            st.markdown("🧠 We'll start from the basics: speaking, reading, grammar & more.")
            st.markdown("🚀 Step by step, you'll get better every day!")
        with col2:
            if st.button("🎯 Start Lesson", key="start_lesson1"):
                st.session_state.current_lesson = "lesson1"
                st.success("Lesson 1 started! Let's begin learning greetings!")

    # Learning Section
    with st.expander("🔤 Learn Greetings", expanded=True):
        st.markdown("**👋 Let's Learn to Say Hello!**")
        st.markdown("We say 'Hello', 'Hi', 'Good morning' when we meet someone. It's polite and friendly!")
        
        # Interactive greeting buttons
        col1, col2, col3 = st.columns(3)
        with col1:
            if st.button("👋 Hello", key="hello_btn"):
                st.info("Hello! How are you today?")
        with col2:
            if st.button("🌅 Good Morning", key="morning_btn"):
                st.info("Good morning! Have a wonderful day!")
        with col3:
            if st.button("👋 Hi", key="hi_btn"):
                st.info("Hi there! Nice to meet you!")

    # Practice Section
    with st.expander("🎯 Practice Exercises", expanded=True):
        st.markdown("**🔠 Build the Greeting!**")
        st.markdown("👉 Sentence: Good morning, teacher.")
        st.markdown("Words: Good / morning / teacher")
        
        # Word building exercise
        st.markdown("**Build your own greeting:**")
        greeting_parts = st.multiselect(
            "Choose greeting parts:",
            ["Good", "Morning", "Hello", "Hi", "Afternoon", "Evening", "Night"],
            default=["Good", "Morning"]
        )
        if greeting_parts:
            st.success(f"Your greeting: {' '.join(greeting_parts)}!")

    # MCQ Section 1
    with st.expander("🧠 MCQ Quiz 1: Spot the Right Greeting", expanded=True):
        st.markdown("**Question 1:** Which picture shows two people shaking hands?")
        
        # Radio button for MCQ
        answer1 = st.radio(
            "Select the correct answer:",
            ["A. Waving goodbye", "B. Shaking hands ✅", "C. Sleeping", "D. Eating food"],
            key="mcq1"
        )
        
        if st.button("Submit Answer 1", key="submit1"):
            if answer1 == "B. Shaking hands ✅":
                st.session_state.exercise_scores["lesson1"] += 1
                st.success("🎉 Correct! Shaking hands is a friendly greeting!")
            else:
                st.error("❌ Try again! Think about what people do when they meet.")

    # MCQ Section 2
    with st.expander("🧠 MCQ Quiz 2: Complete the Sentence", expanded=True):
        st.markdown("**Question 2:** I say ______ in the morning.")
        
        answer2 = st.radio(
            "Select the correct answer:",
            ["A. Good night", "B. Good morning ✅", "C. Bye", "D. Thanks"],
            key="mcq2"
        )
        
        if st.button("Submit Answer 2", key="submit2"):
            if answer2 == "B. Good morning ✅":
                st.session_state.exercise_scores["lesson1"] += 1
                st.success("🎉 Perfect! 'Good morning' is the right greeting for mornings!")
            else:
                st.error("❌ Not quite right. Think about what time of day it is.")

    # Reading Practice
    with st.expander("📖 Reading Practice", expanded=True):
        st.markdown("**📖 Read and Repeat**")
        st.markdown("Hi! I am Rahul.")
        
        # Practice button
        if st.button("🎤 Practice Speaking", key="speak_practice1"):
            st.info("🎤 Say: 'Hi! I am Rahul.' Practice makes perfect!")

# Lesson 2: Introducing Yourself
with lesson_tab2:
    st.markdown("### 🙋 Lesson 2: Introducing Yourself")
    
    # Introduction Section
    with st.expander("🙋 Learn to Introduce Yourself", expanded=True):
        st.markdown("**🙋 Tell Me About You!**")
        st.markdown("We use 'My name is...', 'I am...' to introduce ourselves to others.")
        
        # Interactive introduction form
        st.markdown("**Practice your introduction:**")
        name = st.text_input("What's your name?", placeholder="Enter your name")
        age = st.number_input("How old are you?", min_value=1, max_value=100, value=25)
        city = st.text_input("Where do you live?", placeholder="Enter your city")
        
        if name and age and city:
            st.success(f"👋 Hi! My name is {name}. I am {age} years old. I live in {city}.")

    # MCQ Section 3
    with st.expander("🧠 MCQ Quiz 3: Pick the Right Introduction", expanded=True):
        st.markdown("**Question 3:** Which picture shows a girl saying her name?")
        
        answer3 = st.radio(
            "Select the correct answer:",
            ["A. Writing on board", "B. Sleeping", "C. Saying hello ✅", "D. Running"],
            key="mcq3"
        )
        
        if st.button("Submit Answer 3", key="submit3"):
            if answer3 == "C. Saying hello ✅":
                st.session_state.exercise_scores["lesson2"] += 1
                st.success("🎉 Excellent! Saying hello is a great way to introduce yourself!")
            else:
                st.error("❌ Think about what people do when they first meet.")

    # Fill the Gap Exercise
    with st.expander("✍️ Fill the Gap Exercise", expanded=True):
        st.markdown("**Question 4:** My name ______ Tina.")
        
        answer4 = st.radio(
            "Select the correct answer:",
            ["A. are", "B. is ✅", "C. am", "D. be"],
            key="mcq4"
        )
        
        if st.button("Submit Answer 4", key="submit4"):
            if answer4 == "B. is ✅":
                st.session_state.exercise_scores["lesson2"] += 1
                st.success("🎉 Perfect! 'My name is Tina' is grammatically correct!")
            else:
                st.error("❌ Remember: 'My name is...' uses 'is' not 'are' or 'am'.")

    # Matching Exercise
    with st.expander("🔗 Matching Exercise", expanded=True):
        st.markdown("**Match the following:**")
        
        # Create a matching game
        col1, col2 = st.columns(2)
        with col1:
            st.markdown("**Sentences:**")
            st.markdown("• I am Tina")
            st.markdown("• I am 6 years old")
            st.markdown("• I study in Class 2")
            st.markdown("• I live in Delhi")
        
        with col2:
            st.markdown("**Types:**")
            st.markdown("• Name")
            st.markdown("• Age")
            st.markdown("• School class")
            st.markdown("• Location")
        
        # Interactive matching
        st.markdown("**Practice matching:**")
        sentence_type = st.selectbox(
            "What type is 'I am Tina'?",
            ["Select...", "Name", "Age", "School class", "Location"]
        )
        if sentence_type == "Name":
            st.success("🎉 Correct! 'I am Tina' tells us the person's name.")
        elif sentence_type != "Select...":
            st.error("❌ Try again! Think about what information 'I am Tina' gives us.")

# Progress Tab
with progress_tab:
    st.markdown("### 📊 Your Learning Progress")
    
    # Progress bars
    lesson1_score = st.session_state.exercise_scores["lesson1"]
    lesson2_score = st.session_state.exercise_scores["lesson2"]
    
    st.markdown(f"**Lesson 1: Greetings** - Score: {lesson1_score}/2")
    st.progress(lesson1_score / 2)
    
    st.markdown(f"**Lesson 2: Introduction** - Score: {lesson2_score}/2")
    st.progress(lesson2_score / 2)
    
    total_score = lesson1_score + lesson2_score
    st.markdown(f"**Total Score: {total_score}/4**")
    
    # Achievement system
    if total_score == 4:
        st.balloons()
        st.success("🏆 Congratulations! You've completed all exercises perfectly!")
    elif total_score >= 3:
        st.success("🌟 Great job! You're doing really well!")
    elif total_score >= 1:
        st.info("📚 Keep practicing! You're making progress!")
    else:
        st.info("🎯 Ready to start learning? Begin with Lesson 1!")
    
    # Reset button
    if st.button("🔄 Reset Progress", key="reset_progress"):
        st.session_state.exercise_scores = {"lesson1": 0, "lesson2": 0}
        st.success("Progress reset! Start fresh with your learning journey!")

st.markdown(
    f"<div id=\"selected_lang_code\" style=\"display:none\">{selected_lang_code}</div>",
    unsafe_allow_html=True,
)
st.markdown(
    f"<script>window.langMap = {json.dumps(LANG_TO_TTS_TAG)}</script>",
    unsafe_allow_html=True,
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Subtitle / helper
st.markdown(
    "<div class='info-card'>💡 Tip: Type in your preferred language. We'll translate, chat with Gemini, and reply back in the same language.</div>",
    unsafe_allow_html=True,
)

# User input (larger)
user_text = st.text_area(ui["message_label"], placeholder=ui["message_label"], key="input_text")

# Action buttons row
col_send, col_speak_last, col_stt = st.columns([1, 1, 1])
send_clicked = col_send.button(ui["send_button"], use_container_width=True)
speak_last_clicked = col_speak_last.button(ui["speak_last_button"], use_container_width=True)
col_stt.markdown(f"<button class='speak-btn' onclick=\"startSTT()\">{ui['speak_button']}</button>", unsafe_allow_html=True)

# Chat processing
if send_clicked:
    if user_text.strip():
        # Step 1: Translate user text to English if not already
        if selected_lang_code != "eng_Latn":
            prompt_en = translate(user_text, selected_lang_code, "eng_Latn")
        else:
            prompt_en = user_text

        # Step 2: Send to Gemini
        gemini_response_en = gemini_chat(prompt_en)

        # Step 3: Translate response back
        if selected_lang_code != "eng_Latn":
            gemini_response_local = translate(gemini_response_en, "eng_Latn", selected_lang_code)
        else:
            gemini_response_local = gemini_response_en

        # Step 4: Store in history (store roles; localize on display)
        st.session_state.chat_history.append(("user", user_text))
        st.session_state.chat_history.append(("bot", gemini_response_local))

# Display chat as bubbles
for speaker, msg in st.session_state.chat_history:
    label = ui["you"] if speaker in ("user", "You") else ui["bot"] if speaker in ("bot", "Bot") else str(speaker)
    role_class = "user" if speaker in ("user", "You") else "bot"
    st.markdown(
        f"<div class='chat-bubble {role_class}'>"
        f"<div class='label'>{escape(label)}</div>"
        f"<div class='text'>{escape(str(msg))}</div>"
        f"</div>",
        unsafe_allow_html=True,
    )

# ---------------- Voice Input (STT) ---------------- #
st.markdown(
    f"""
<script>
function getSelectedLangCode() {{
  var el = document.getElementById('selected_lang_code');
  return el ? el.textContent : 'eng_Latn';
}}
function langTag() {{
  var code = getSelectedLangCode();
  var map = window.langMap || {{}};
  return map[code] || 'en-US';
}}
function startSTT() {{
  var SpeechRecognition = window.SpeechRecognition || window.webkitSpeechRecognition;
  if (!SpeechRecognition) {{ alert('SpeechRecognition not supported in this browser.'); return; }}
  var recognition = new SpeechRecognition();
  recognition.lang = langTag();
  recognition.onresult = function(event) {{
    var text = event.results[0][0].transcript;
    window.parent.postMessage({{type: 'stt_result', text: text}}, '*');
  }}
  recognition.start();
}}
</script>
""",
    unsafe_allow_html=True,
)

# ---------------- Voice Output (TTS) ---------------- #
st.markdown(
    """
<script>
function getSelectedLangCode() {
  var el = document.getElementById('selected_lang_code');
  return el ? el.textContent : 'eng_Latn';
}
function langTag() {
  var code = getSelectedLangCode();
  var map = window.langMap || {};
  return map[code] || 'en-US';
}
function speakText(text){
  var tag = langTag();
  var utterance = new SpeechSynthesisUtterance(text);
  utterance.lang = tag;
  function chooseAndSpeak(){
    var voices = speechSynthesis.getVoices();
    var v = voices.find(v => v.lang && v.lang.toLowerCase() === tag.toLowerCase());
    if (!v) {
      var base = tag.split('-')[0].toLowerCase();
      v = voices.find(v => (v.lang||'').toLowerCase().startsWith(base));
    }
    utterance.voice = v || voices[0] || null;
    speechSynthesis.speak(utterance);
  }
  if (speechSynthesis.getVoices().length === 0) {
    speechSynthesis.onvoiceschanged = chooseAndSpeak;
  } else {
    chooseAndSpeak();
  }
}
</script>
""",
    unsafe_allow_html=True,
)

# Button to speak last bot message
if st.session_state.chat_history and speak_last_clicked:
    last_bot_msg = next((msg for speaker, msg in reversed(st.session_state.chat_history) if speaker in ("bot", "Bot")), None)
    if last_bot_msg:
        st.markdown(f"<script>speakText({repr(last_bot_msg)})</script>", unsafe_allow_html=True)
