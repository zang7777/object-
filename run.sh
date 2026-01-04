#!/bin/bash

# 1. Get Local IP Address
# Try hostname -I first, taking the first IP (usually the LAN IP)
HOST_IP=$(hostname -I | cut -d' ' -f1)

if [ -z "$HOST_IP" ]; then
    # Fallback if hostname -I fails
    HOST_IP=$(ip -4 addr show | grep -oP '(?<=inet\s)\d+(\.\d+){3}' | grep -v "127.0.0.1" | head -n 1)
fi

echo "=================================================="
echo "   MOBILE SENSOR BRIDGE - SECURE SERVER SETUP"
echo "=================================================="
echo "Detected Local IP: $HOST_IP"

# 2. Generate SSL Certificates (Self-Signed)
echo "Generating SSL Certificates..."
openssl req -new -x509 -keyout key.pem -out cert.pem -days 365 -nodes \
    -subj "/C=US/ST=State/L=City/O=SensorBridge/CN=$HOST_IP" \
    2>/dev/null

if [ $? -eq 0 ]; then
    echo "✅ Certificates generated (cert.pem, key.pem)"
else
    echo "❌ Failed to generate certificates. Do you have openssl installed?"
    exit 1
fi

# 3. Print Instructions
echo ""
echo "🚀 Server Starting..."
echo "--------------------------------------------------"
echo "📲  ON YOUR MOBILE DEVICE, GO TO:"
echo ""
echo "    https://$HOST_IP:8000/mobile"
echo ""
echo "--------------------------------------------------"
echo "⚠️  NOTE: You will see a security warning."
echo "    This is normal. Click 'Advanced' -> 'Proceed'."
echo "=================================================="

# 4. Cleanup & Run Python Secure Server
# Kill any process on port 8000 to avoid "Address already in use"
PID=$(lsof -t -i:8000)
if [ ! -z "$PID" ]; then
    echo "⚠️  Port 8000 in use. Killing process $PID..."
    kill -9 $PID 2>/dev/null
fi

echo "Activating virtual environment..."
source .venv/bin/activate

echo "Starting Flask-SocketIO Server (OpenCV Enabled)..."
python3 app.py
