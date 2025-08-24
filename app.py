from fastapi import FastAPI, UploadFile, File
from fastapi.responses import JSONResponse
from transformers import pipeline
import shutil
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
        # Save uploaded file temporarily
        tmp_file_path = f"/tmp/{file.filename}"
        with open(tmp_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Run ASR
        transcription = asr_pipeline(tmp_file_path)["text"]

        # Clean up temp file
        os.remove(tmp_file_path)

        return JSONResponse(content={"transcription": transcription})

    except Exception as e:
        return JSONResponse(content={"error": str(e)}, status_code=500)

@app.get("/")
async def root():
    return {"message": "Arabic ASR server running!"}
