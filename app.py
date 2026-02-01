import os
import random
import time
from azure.cosmos import CosmosClient, PartitionKey, exceptions
import azure.cognitiveservices.speech as speechsdk
from openai import AzureOpenAI
from dotenv import load_dotenv
from flask import Flask, render_template, jsonify, request



load_dotenv()
app = Flask(__name__)

# Credentials
SPEECH_KEY = os.getenv('SPEECH_KEY')
SPEECH_REGION = os.getenv('SPEECH_REGION')
AZURE_OPENAI_KEY = os.getenv("AZURE_OPENAI_KEY")
AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME")
COSMOS_ENDPOINT = os.getenv("COSMOS_ENDPOINT")
COSMOS_KEY = os.getenv("COSMOS_KEY")

client = AzureOpenAI(
    azure_endpoint=AZURE_OPENAI_ENDPOINT,
    api_key=AZURE_OPENAI_KEY,
    api_version="2024-02-15-preview"
)
# --- 2. COSMOS DB INITIALIZATION ---
client_db = CosmosClient(COSMOS_ENDPOINT, COSMOS_KEY)
database = client_db.create_database_if_not_exists(id="AssistantDB")
container = database.create_container_if_not_exists(
    id="ChatHistory", 
    partition_key=PartitionKey(path="/userId"),
    
)
USER_ID = "annu_kumar" 

# --- 3. STORAGE FUNCTIONS ---
def save_to_cosmos(history):
    container.upsert_item({
        "id": USER_ID,
        "userId": USER_ID,
        "history": history
    })

def load_from_cosmos():
    try:
        return container.read_item(item=USER_ID, partition_key=USER_ID).get("history", [])
    except exceptions.CosmosResourceNotFoundError:
        return []
conversation_history = load_from_cosmos()
agent_active = False
WAKE_WORD = "agent"
last_activity = "Waiting for wake word..."

def get_ride_details():
    """Generates random but realistic ride data."""
    distance = random.randint(2, 25)
    price = 50 + (distance * 15)
    otp = random.randint(1000, 9999)
    return distance, price, otp

def get_ai_response(user_input, language_context):
    global conversation_history
    dist, fare, otp = get_ride_details()
    conversation_history.append({"role": "user", "content": user_input})
    
    system_instructions = (
        f"You are a helpful voice assistant. User language: {language_context}. "
        f"CURRENT RIDE DATA: Distance is {dist}km, Fare is {fare} rupees, OTP is {otp}. "
        "CONSTRAINTS: "
        "1. If the user mentions a destination or asks to book, use the CURRENT RIDE DATA provided above. "
        "2. If user says 'Yes', 'Confirm', or 'Book it' AFTER you mentioned the price, you MUST say: "
        "'Confirmed! Your ride is on the way. The driver name is Ramesh and the vehicle number is T S 0 8 F I 8 9 7 6.' "
        f"Your security O T P is {otp}. I repeat, your O T P is {otp}. Once again, your O T P is {otp}.'"
        "3. If they change the destination, acknowledge it and state the NEW distance and fare provided."
        "4. For other chat, be brief like Siri."
    )
    
    messages = [{"role": "system", "content": system_instructions}] + conversation_history

    try:
        response = client.chat.completions.create(
            model=DEPLOYMENT_NAME,
            messages=messages,
            temperature=0.3
        )
        ai_reply = response.choices[0].message.content
        conversation_history.append({"role": "assistant", "content": ai_reply})
        if len(conversation_history) > 10:
            conversation_history.pop(0)
        save_to_cosmos(conversation_history) 
        return ai_reply
    except Exception as e:
        return f"Azure OpenAI Error: {str(e)}"

def speak_human_reply(text):
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    speech_config.speech_synthesis_voice_name = "en-US-AvaNeural"
    synthesizer = speechsdk.SpeechSynthesizer(speech_config=speech_config)
    synthesizer.speak_text_async(text).get()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/set_wake_word', methods=['POST'])
def set_wake_word():
    global WAKE_WORD
    WAKE_WORD = request.json.get('wake_word', '').lower().strip()
    return jsonify({"status": "success", "message": f"Wake word set to {WAKE_WORD}"})

@app.route('/run_assistant')
def run_assistant():
    global agent_active, WAKE_WORD
    
    speech_config = speechsdk.SpeechConfig(subscription=SPEECH_KEY, region=SPEECH_REGION)
    audio_config = speechsdk.audio.AudioConfig(use_default_microphone=True)
    languages = ["en-US", "hi-IN", "te-IN", "ta-IN"]
    auto_detect_config = speechsdk.languageconfig.AutoDetectSourceLanguageConfig(languages=languages)
    
    recognizer = speechsdk.SpeechRecognizer(
        speech_config=speech_config, 
        auto_detect_source_language_config=auto_detect_config, 
        audio_config=audio_config
    )

    # Listen once (triggered by button click)
    result = recognizer.recognize_once_async().get()

    if result.reason == speechsdk.ResultReason.RecognizedSpeech:
        user_text = result.text.lower().replace(".", "").strip()
        lang_result = speechsdk.AutoDetectSourceLanguageResult(result)
        detected_lang = lang_result.language

        # WAKE WORD LOGIC (Still required to activate the agent for the first time)
        if not agent_active:
            if WAKE_WORD in user_text:
                agent_active = True
                reply = "Yes, I am listening."
                speak_human_reply(reply)
                return jsonify({"user": user_text, "assistant": reply})
            return jsonify({"user": user_text, "assistant": f"Please say '{WAKE_WORD}' first."})

        # AI CONVERSATION
        ai_text = get_ai_response(user_text, detected_lang)
        speak_human_reply(ai_text)
        return jsonify({"user": user_text, "assistant": ai_text})

    return jsonify({"error": "No speech detected"})

if __name__ == '__main__':
    app.run(debug=True)