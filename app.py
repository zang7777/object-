import time
import os
import cv2
import numpy as np
import base64
import torch
from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
from torchvision import models
from PIL import Image
import google.generativeai as genai

app = Flask(__name__)
# Threading mode is best for handling simultaneous video and AI processing
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins='*')

# --- Gemini Setup ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
if not GEMINI_API_KEY and os.path.exists('key.txt'):
    with open('key.txt', 'r') as f:
        GEMINI_API_KEY = f.read().strip()

# Initialize Gemini 1.5 Flash (Faster response for Voice)
genai.configure(api_key=GEMINI_API_KEY)
model_gemini = genai.GenerativeModel('gemini-1.5-flash')
# Global chat session to maintain conversation history
chat_session = model_gemini.start_chat(history=[])

# --- Vision Model Setup ---
print("Loading MobileNetV3...")
weights = models.MobileNet_V3_Large_Weights.DEFAULT
model_net = models.mobilenet_v3_large(weights=weights)
model_net.eval()
preprocess = weights.transforms()
CLASSES = weights.meta["categories"]

# --- Global State ---
last_detected_label = "nothing yet"
last_spoken_label = None
last_spoken_time = 0
AUTO_SPEAK_COOLDOWN = 5.0 # Seconds between auto-announcements

@app.route('/')
def home(): return render_template('dashboard.html')

@app.route('/mobile')
def mobile(): return render_template('mobile.html')

# --- Voice & Chat Interaction ---

@socketio.on('chat_message')
def handle_chat_message(data):
    """
    This handles both typed text and voice transcripts sent from the client.
    """
    global last_detected_label, chat_session
    user_input = data.get('message', '')
    
    if not user_input:
        return

    # Provide Gemini with 'Visual Context' so it knows what the camera sees
    prompt = f"(Visual Context: You are looking through a camera and see a {last_detected_label}).\nUser says: {user_input}"
    
    try:
        response = chat_session.send_message(prompt)
        bot_reply = response.text
        
        # 1. Update the chat history in the UI
        emit('chat_reply', {'text': bot_reply}, broadcast=True)
        
        # 2. Trigger the Speak command so the user HEARS the bot
        emit('voice_command', {'label': bot_reply}, broadcast=True)
        
    except Exception as e:
        print(f"Gemini Error: {e}")
        emit('chat_reply', {'text': "Sorry, I'm having trouble processing that right now."})

@socketio.on('identify_request')
def handle_manual_identify():
    """Triggered by a button or voice command to explain the object."""
    global last_detected_label
    prompt = f"Explain what a {last_detected_label} is and its purpose in 2 short sentences."
    response = chat_session.send_message(prompt)
    emit('voice_command', {'label': response.text}, broadcast=True)

# --- Video Stream & Detection ---

@socketio.on('video_frame')
def handle_video(data):
    global last_detected_label, last_spoken_label, last_spoken_time
    try:
        header, encoded = data.split(",", 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Preprocess and Infer
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        input_tensor = preprocess(pil_image).unsqueeze(0)

        with torch.no_grad():
            output = model_net(input_tensor)
        
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, cat_id = torch.topk(probabilities, 1)
        
        if confidence.item() > 0.6:
            current_label = CLASSES[cat_id]
            last_detected_label = current_label
            
            # --- Auto-Announce Logic ---
            now = time.time()
            if (current_label != last_spoken_label) or (now - last_spoken_time > AUTO_SPEAK_COOLDOWN):
                emit('voice_command', {'label': f"I see a {current_label}"}, broadcast=True)
                last_spoken_label = current_label
                last_spoken_time = now

            cv2.putText(frame, f"{current_label}", (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

        # Emit frame back to dashboard
        _, buffer = cv2.imencode('.jpg', frame)
        processed_data = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')
        emit('dashboard_video', processed_data, broadcast=True)
        
    except Exception as e:
        print(f"Frame Error: {e}")

if __name__ == '__main__':
    socketio.run(app, host='0.0.0.0', port=8000, ssl_context=('cert.pem', 'key.pem'), allow_unsafe_werkzeug=True)