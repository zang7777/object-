# import eventlet
# eventlet.monkey_patch()

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import ssl
import google.generativeai as genai
import os

app = Flask(__name__)
# Switch to threading for stability on Python 3.12 w/ SSL
socketio = SocketIO(app, async_mode='threading', cors_allowed_origins='*')

# --- Gemini Setup ---
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
model = None
chat_session = None

# Try loading from key.txt if not in env
if not GEMINI_API_KEY:
    try:
        if os.path.exists('key.txt'):
            with open('key.txt', 'r') as f:
                GEMINI_API_KEY = f.read().strip()
    except Exception:
        pass

if GEMINI_API_KEY:
    try:
        genai.configure(api_key=GEMINI_API_KEY)
        model = genai.GenerativeModel('gemini-pro')
        chat_session = model.start_chat(history=[])
        print("✅ Gemini AI Configured")
    except Exception as e:
         print(f"❌ Gemini Configuration Error: {e}")
else:
    print("⚠️ GEMINI_API_KEY not found. Chatbot will not work.")

# --- Global State ---
identification_requested = False

@app.route('/')
def home():
    return render_template('dashboard.html')

@app.route('/mobile')
def mobile():
    return render_template('mobile.html')

@socketio.on('connect')
def handle_connect():
    print(f"Client connected: {request.sid}")

@socketio.on('disconnect')
def handle_disconnect():
    print(f"Client disconnected: {request.sid}")

# --- IMU Data Handling ---
@socketio.on('imu_data')
def handle_imu(data):
    # Simply forward the data to the dashboard
    emit('dashboard_imu', data, broadcast=True)

# --- Voice / Chat Events ---

@socketio.on('identify_request')
def handle_identify_request():
    global identification_requested
    print("🎤 Voice Command: Identify object requested")
    identification_requested = True

@socketio.on('chat_message')
def handle_chat_message(data):
    global chat_session, last_detected_label
    msg = data.get('message')
    print(f"💬 Chat User: {msg}")
    
    if not chat_session:
        emit('chat_reply', {'text': "Error: Gemini API Key not configured."}, broadcast=True)
        return

    try:
        # Inject context if available
        context_msg = msg
        if last_detected_label:
            context_msg = f"[Context: The user is looking at a {last_detected_label}] {msg}"
            
        response = chat_session.send_message(context_msg)
        reply = response.text
        print(f"🤖 Chat Gemini: {reply}")
        emit('chat_reply', {'text': reply}, broadcast=True)
    except Exception as e:
        emit('chat_reply', {'text': f"Error: {str(e)}"}, broadcast=True)

# --- Video Frame Handling ---

# Load PyTorch Model
import torch
from torchvision import models, transforms
from PIL import Image

print("Loading PyTorch MobileNetV3-Large...")
weights = models.MobileNet_V3_Large_Weights.DEFAULT
model_net = models.mobilenet_v3_large(weights=weights)
model_net.eval()

# Preprocessing transforms
preprocess = weights.transforms()
CLASSES = weights.meta["categories"]

# Global variable to store last detected object for chat context
last_detected_label = None

@socketio.on('video_frame')
def handle_video(data):
    global identification_requested, last_detected_label
    try:
        # 1. Decode base64 image
        header, encoded = data.split(",", 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        # 2. Object Detection (PyTorch)
        # Convert BGR (OpenCV) to RGB (PIL)
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pil_image = Image.fromarray(rgb_frame)
        
        # Preprocess and infer
        input_tensor = preprocess(pil_image)
        input_batch = input_tensor.unsqueeze(0) # Create mini-batch

        with torch.no_grad():
            output = model_net(input_batch)
        
        # Get top prediction
        probabilities = torch.nn.functional.softmax(output[0], dim=0)
        confidence, cat_id = torch.topk(probabilities, 1)
        
        confidence = confidence.item()
        d_label = CLASSES[cat_id]
        
        # Threshold
        if confidence > 0.5:
            label_text = f"{d_label}: {confidence*100:.1f}%"
            # Draw on frame (simple full-frame annotation since classification model doesn't return boxes like SSD)
            # Make it look cool
            cv2.putText(frame, label_text, (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Store for chat context
            last_detected_label = d_label
            
            # Voice Trigger Logic
            if identification_requested:
                print(f"🗣️ Speaking: {d_label}")
                
                # Get detailed info from Gemini if key is valid
                gemini_summary = f"Identified {d_label}."
                if chat_session:
                    try:
                        prompt_text = f"Tell me a fun fact about a {d_label} in 20 words or less."
                        resp = chat_session.send_message(prompt_text)
                        gemini_summary = resp.text
                    except Exception as e:
                        print(f"Gemini Error: {e}")
                
                emit('voice_command', {'label': gemini_summary}, broadcast=True)
                identification_requested = False

        # 3. Encode back to base64
        _, buffer = cv2.imencode('.jpg', frame)
        processed_data = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

        # 4. Emit results
        emit('dashboard_video', processed_data, broadcast=True)
        
    except Exception as e:
        print(f"Error processing frame: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    print("------------------------------------------------")
    print("🚀 Flask SocketIO Server Running (PyTorch CPU | Threading Mode)")
    print("------------------------------------------------")
    # Using threading mode, pass ssl_context as a tuple ('cert', 'key')
    socketio.run(app, host='0.0.0.0', port=8000, ssl_context=('cert.pem', 'key.pem'), allow_unsafe_werkzeug=True)
