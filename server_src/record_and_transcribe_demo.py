import sounddevice as sd
from scipy.io.wavfile import write
import numpy as np
from transformers import WhisperProcessor, WhisperForConditionalGeneration

print("This is a self-contained test of your microphone and a small Whisper model.")

def record_audio(duration=5, sample_rate=16000):
    """
    Record audio from the microphone for a given duration and sample rate.
    """
    recording = sd.rec(int(duration * sample_rate),
                       samplerate=sample_rate, channels=1, dtype='float32')
    sd.wait()  # Wait until recording is finished
    return recording.flatten(), sample_rate

# Save recording to a file (optional)
def save_recording(recording, sample_rate, file_name='output.wav'):
    write(file_name, sample_rate, recording)

# Load model and processor
print("Loading model and processor...")
processor = WhisperProcessor.from_pretrained("openai/whisper-small")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-small")
print("Model and processor loaded")

print("Recording audio...")
audio_array, sampling_rate = record_audio()
print("Audio recorded")

print("Processing audio...")
input_features = processor(
    audio_array, sampling_rate=sampling_rate, return_tensors="pt").input_features
print("Audio processed")

print("Generating token ids...")
predicted_ids = model.generate(input_features)
print("Token ids generated")

print("Decoding token ids...")
transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)
print("Transcription:", transcription)
