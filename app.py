import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request
from flask_socketio import SocketIO, emit
import cv2
import numpy as np
import base64
import ssl

app = Flask(__name__)
socketio = SocketIO(app, async_mode='eventlet', cors_allowed_origins='*')

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

# --- Video Frame Handling ---

# Load Model
print("Loading MobileNet SSD...")
net = cv2.dnn.readNetFromCaffe('MobileNetSSD_deploy.prototxt', 'MobileNetSSD_deploy.caffemodel')
CLASSES = ["background", "aeroplane", "bicycle", "bird", "boat",
           "bottle", "bus", "car", "cat", "chair", "cow", "diningtable",
           "dog", "horse", "motorbike", "person", "pottedplant", "sheep",
           "sofa", "train", "tvmonitor"]
COLORS = np.random.uniform(0, 255, size=(len(CLASSES), 3))

@socketio.on('video_frame')
def handle_video(data):
    try:
        # 1. Decode base64 image
        header, encoded = data.split(",", 1)
        nparr = np.frombuffer(base64.b64decode(encoded), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if frame is None:
            return

        # 2. Object Detection
        (h, w) = frame.shape[:2]
        blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 0.007843, (300, 300), 127.5)
        net.setInput(blob)
        detections = net.forward()

        label_to_speak = ""
        max_conf = 0

        # Loop over detections
        for i in range(detections.shape[2]):
            confidence = detections[0, 0, i, 2]

            if confidence > 0.5:
                idx = int(detections[0, 0, i, 1])
                label = CLASSES[idx]
                
                # Draw box
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                
                color = COLORS[idx]
                cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)
                text = "{}: {:.2f}%".format(label, confidence * 100)
                y = startY - 15 if startY - 15 > 15 else startY + 15
                cv2.putText(frame, text, (startX, y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                # Prioritize high confidence for speech
                if confidence > max_conf:
                    max_conf = confidence
                    label_to_speak = label

        # 3. Encode back to base64
        _, buffer = cv2.imencode('.jpg', frame)
        processed_data = "data:image/jpeg;base64," + base64.b64encode(buffer).decode('utf-8')

        # 4. Emit results
        emit('dashboard_video', processed_data, broadcast=True)
        if label_to_speak:
            emit('voice_command', {'label': label_to_speak}, broadcast=True)
        
    except Exception as e:
        print(f"Error processing frame: {e}")

if __name__ == '__main__':
    print("------------------------------------------------")
    print("🚀 Flask SocketIO Server Running")
    print("------------------------------------------------")
    # For eventlet, pass keyfile and certfile directly
    socketio.run(app, host='0.0.0.0', port=8000, keyfile='key.pem', certfile='cert.pem')
