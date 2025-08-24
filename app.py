from fastapi import FastAPI, UploadFile, File
from speechbrain.pretrained import EncoderASR
import torchaudio
import os

app = FastAPI(title="Arabic Digit Recognition API")

# Load model once at startup
asr_model = EncoderASR.from_hparams(
    source="speechbrain/asr-wav2vec2-commonvoice-14-ar",
    savedir="pretrained_models/asr-wav2vec2-commonvoice-14-ar"
)

@app.post("/predict")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded file temporarily
        tmp_input = "/tmp/input_audio.m4a"
        with open(tmp_input, "wb") as f:
            f.write(await file.read())

        # Load audio using torchaudio
        waveform, sr = torchaudio.load(tmp_input)

        # Resample to 16kHz if needed
        if sr != 16000:
            import torchaudio.transforms as T
            resampler = T.Resample(sr, 16000)
            waveform = resampler(waveform)

        # Save as temporary wav file for ASR
        tmp_wav = "/tmp/audio.wav"
        torchaudio.save(tmp_wav, waveform, 16000)

        # Run ASR
        transcription = asr_model.transcribe_file(tmp_wav)

        # Clean up temp files
        os.remove(tmp_input)
        os.remove(tmp_wav)

        return {"transcription": transcription}

    except Exception as e:
        return {"error": str(e)}
