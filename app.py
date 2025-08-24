from fastapi import FastAPI, UploadFile, File
from transformers import pipeline
import torchaudio
import os

app = FastAPI(title="Arabic Digit Recognition API")

# Load Whisper-Small Arabic ASR
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="Salama1429/KalemaTech-Arabic-STT-ASR-based-on-Whisper-Small"
)

@app.post("/predict")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        tmp_input = f"/tmp/{file.filename}"

        # Save uploaded file
        with open(tmp_input, "wb") as f:
            f.write(await file.read())

        # Load audio with torchaudio
        waveform, sr = torchaudio.load(tmp_input)

        # Resample to 16 kHz if needed
        if sr != 16000:
            resampler = torchaudio.transforms.Resample(sr, 16000)
            waveform = resampler(waveform)

        # Save as wav
        tmp_wav = f"/tmp/{os.path.splitext(file.filename)[0]}.wav"
        torchaudio.save(tmp_wav, waveform, 16000)

        # Run ASR
        transcription = asr_pipeline(tmp_wav)["text"]

        # Clean up
        os.remove(tmp_input)
        os.remove(tmp_wav)

        return {"transcription": transcription}

    except Exception as e:
        return {"error": str(e)}
