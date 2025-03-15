import streamlit as st
import torch
import torchaudio
import matplotlib.pyplot as plt
import io

# Set device (use GPU if available)
torch.random.manual_seed(0)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load Wav2Vec2 model and pipeline
bundle = torchaudio.pipelines.WAV2VEC2_ASR_BASE_960H
model = bundle.get_model().to(device)

# Define CTC Decoder
class GreedyCTCDecoder(torch.nn.Module):
    def __init__(self, labels, blank=0):
        super().__init__()
        self.labels = labels
        self.blank = blank

    def forward(self, emission: torch.Tensor) -> str:
        indices = torch.argmax(emission, dim=-1)
        indices = torch.unique_consecutive(indices, dim=-1)
        indices = [i for i in indices if i != self.blank]
        return "".join([self.labels[i] for i in indices])

# Instantiate decoder
decoder = GreedyCTCDecoder(labels=bundle.get_labels())

# Streamlit UI
st.title("🎙️ Speech-to-Text with Wav2Vec2")
st.write("Upload an audio file to transcribe using Wav2Vec2.")

# File uploader
uploaded_file = st.file_uploader("Upload an audio file", type=["wav", "mp3"])

if uploaded_file is not None:
    # Convert uploaded file to BytesIO for torchaudio
    file_bytes = io.BytesIO(uploaded_file.read())

    # Load waveform from BytesIO
    try:
        waveform, sample_rate = torchaudio.load(file_bytes)
        waveform = waveform.to(device)

        # Resample if sample rate does not match model's expected rate
        if sample_rate != bundle.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, bundle.sample_rate)

        # Transcribe the audio
        with torch.inference_mode():
            emission, _ = model(waveform)

        # Decode transcript
        transcript = decoder(emission[0])

        # Display transcript
        st.subheader("📝 Transcription:")
        st.write(transcript)

        # Display audio player
        st.audio(uploaded_file, format="audio/mp3")

        # Plot emission data
        st.subheader("📊 Model Output (Emission Data):")
        fig, ax = plt.subplots(figsize=(12, 6))
        ax.imshow(emission[0].cpu().T, interpolation="nearest", aspect="auto")
        ax.set_title("Classification result")
        ax.set_xlabel("Frame (time-axis)")
        ax.set_ylabel("Class")
        st.pyplot(fig)

        # Extract features from transformer layers
        with torch.inference_mode():
            features, _ = model.extract_features(waveform)

        st.subheader("📈 Feature Maps from Transformer Layers:")
        fig, axs = plt.subplots(len(features), 1, figsize=(12, 3 * len(features)))
        if len(features) == 1:
            axs = [axs]
        for i, feats in enumerate(features):
            axs[i].imshow(feats[0].cpu(), interpolation="nearest", aspect="auto")
            axs[i].set_title(f"Feature from transformer layer {i + 1}")
            axs[i].set_xlabel("Feature dimension")
            axs[i].set_ylabel("Frame (time-axis)")
        st.pyplot(fig)

    except Exception as e:
        st.error(f"Error loading audio file: {e}")

# Footer
st.write("💡 Powered by Torchaudio, PyTorch, and Streamlit")
