from scipy.io.wavfile import write
from transformers import pipeline
import os
import torch
from pydub import AudioSegment
from flask import Flask, request, jsonify
import shutil

app = Flask(__name__)

# Kiểm tra thiết bị sử dụng GPU hoặc CPU
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

# Initialize the ASR (Automatic Speech Recognition) pipeline
pipe_stt = pipeline("automatic-speech-recognition",
                    model="vinai/PhoWhisper-medium", device=0 if device == "cuda" else -1)

# Hàm chuyển đổi giọng nói sang văn bản
def speech_to_text(audio_path, pipe):
    try:
        # Ensure file exists
        if os.path.exists(audio_path):
            # Convert audio to text
            result = pipe(audio_path)
            return result['text']
        else:
            return f"Audio file {audio_path} does not exist."
    except Exception as e:
        return str(e)

@app.route('/transcribe', methods=['POST'])
def transcribe_audio():
    if 'file' not in request.files:
        return jsonify({"error": "No file part in the request."}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "No selected file."}), 400

    # Save the uploaded file temporarily
    os.makedirs("uploads", exist_ok=True)
    file_path = os.path.join("uploads", file.filename)
    file.save(file_path)

    try:
        # Gọi hàm chuyển đổi giọng nói sang văn bản
        transcription = speech_to_text(file_path, pipe_stt)
        print(transcription)
        # Trả về kết quả dưới dạng JSON
        return jsonify({"transcription": transcription}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500
    finally:
        # Clean up: remove the uploaded file after processing
        if os.path.exists(file_path):
            os.remove(file_path)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
