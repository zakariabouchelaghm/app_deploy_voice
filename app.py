from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from transformers import pipeline
from pydub import AudioSegment
import os

app = FastAPI(title="Arabic Digit Recognition API")

# Load the Whisper-Small Arabic ASR pipeline once at startup
asr_pipeline = pipeline(
    "automatic-speech-recognition",
    model="Salama1429/KalemaTech-Arabic-STT-ASR-based-on-Whisper-Small"
)

@app.post("/predict")
async def transcribe_audio(file: UploadFile = File(...)):
    try:
        # Save uploaded .m4a file temporarily
        tmp_m4a_path = f"/tmp/{file.filename}"
        with open(tmp_m4a_path, "wb") as f:
            f.write(await file.read())

        # Convert m4a to wav
        tmp_wav_path = f"/tmp/{os.path.splitext(file.filename)[0]}.wav"
        audio = AudioSegment.from_file(tmp_m4a_path, format="m4a")
        audio.export(tmp_wav_path, format="wav")

        # Run ASR
        transcription = asr_pipeline(tmp_wav_path)["text"]

        # Clean up temp files
        os.remove(tmp_m4a_path)
        os.remove(tmp_wav_path)

        return JSONResponse(content={"transcription": transcription})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {"message": "Arabic ASR server running!"}
