import os
import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils.utils import audio_extraction, INPUT_VIDEO
import whisper

# AUDIO_PATH = audio_extraction(path = INPUT_VIDEO)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

audio_path = os.path.join(BASE_DIR, "outputs", "audio_file.mp3")


model = whisper.load_model("tiny")

# load audio and pad/trim it to fit 30 seconds
audio = whisper.load_audio(audio_path)
audio = whisper.pad_or_trim(audio)

# make log-Mel spectrogram and move to the same device as the model
mel = whisper.log_mel_spectrogram(audio, n_mels=model.dims.n_mels).to(model.device)

_, probs = model.detect_language(mel)
print(f"Detected language: {max(probs, key=probs.get)}")


# decode the audio
options = whisper.DecodingOptions()
result = whisper.decode(model, mel, options)

# print the recognized text
print(result.text)