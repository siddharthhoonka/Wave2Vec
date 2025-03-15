import streamlit as st
import torch
import torchaudio
from torchaudio.pipelines import WAV2VEC2_ASR_BASE_960H
import matplotlib.pyplot as plt
import io
from pydub import AudioSegment

# 🔥 Fix: Set FFmpeg path explicitly for pydub
AudioSegment.converter = "/usr/bin/ffmpeg"  # Adjust path for Windows if needed
AudioSegment.ffprobe = "/usr/bin/ffprobe"

# Set device
torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the Wav2Vec2 model
@st.cache_resource
def load_model():
    bundle = WAV2VEC2_ASR_BASE_960H
    model = bundle.get_model().to(device)
    labels = bundle.get_labels()
    sample_rate = bundle.sample_rate
    return model, labels, sample_rate

model, labels, sample_rate = load_model()

# Define GreedyCTCDecoder
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        indices = torch.argmax(emission, dim=-1)  # Get predicted class indices
        indices = torch.unique_consecutive(indices, dim=-1)  # Remove duplicates
        indices = [i for i in indices if i != self.blank]  # Remove blank tokens
        return "".join([self.labels[i] for i in indices])

decoder = GreedyCTCDecoder(labels=labels)

# Streamlit UI
st.title("Wave2Vec2 Speech-to-Text with Streamlit")

uploaded_file = st.file_uploader("Upload an audio file (MP3, WAV)", type=["mp3", "wav"])

if uploaded_file is not None:
    try:
        # ✅ Convert to WAV using pydub
        file_bytes = io.BytesIO(uploaded_file.read())
        audio = AudioSegment.from_file(file_bytes)
        wav_bytes = io.BytesIO()
        audio.export(wav_bytes, format="wav")
        wav_bytes.seek(0)

        # ✅ Load waveform using torchaudio
        waveform, input_sample_rate = torchaudio.load(wav_bytes)
        waveform = waveform.to(device)

        # ✅ Resample if sample rate is different
        if input_sample_rate != sample_rate:
            waveform = torchaudio.functional.resample(waveform, input_sample_rate, sample_rate)

        # ✅ Display audio player
        st.audio(wav_bytes, format="audio/wav")

        # ✅ Extract features from the model
        with torch.inference_mode():
            features, _ = model.extract_features(waveform)

        # ✅ Plot extracted features
        fig, ax = plt.subplots(len(features), 1, figsize=(16, 4.3 * len(features)))
        for i, feats in enumerate(features):
            ax[i].imshow(feats[0].cpu(), interpolation="nearest")
            ax[i].set_title(f"Feature from transformer layer {i+1}")
            ax[i].set_xlabel("Feature dimension")
            ax[i].set_ylabel("Frame (time-axis)")
        st.pyplot(fig)

        # ✅ Get predictions from model
        with torch.inference_mode():
            emission, _ = model(waveform)

        # ✅ Plot classification result
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(emission[0].cpu().T, interpolation="nearest")
        ax.set_title("Classification result")
        ax.set_xlabel("Frame (time-axis)")
        ax.set_ylabel("Class")
        st.pyplot(fig)

        # ✅ Decode the output to text
        transcript = decoder(emission[0])
        st.subheader("Transcript:")
        st.write(transcript)

    except Exception as e:
        st.error(f"Error processing audio file: {e}")

