from fastapi import FastAPI, UploadFile, File
from speechbrain.inference import EncoderASR
from pydub import AudioSegment
import numpy as np
import io


app = FastAPI(title="Arabic Digit Recognition API")

asr_model = EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-14-ar", savedir="pretrained_models/asr-wav2vec2-commonvoice-14-ar")
@app.post("/predict")
async def transcribe_audio(file: UploadFile = File(...)):
    # Read the uploaded file into memory
    file_bytes = await file.read()
    
    # Convert m4a (or other formats) to WAV using pydub
    audio = AudioSegment.from_file(io.BytesIO(file_bytes))
    audio = audio.set_frame_rate(16000).set_channels(1)  # 16kHz mono
    
    # Export to bytes buffer in WAV format
    wav_io = io.BytesIO()
    audio.export(wav_io, format="wav")
    wav_io.seek(0)
    
    # Run ASR on WAV bytes
    transcription = asr_model.transcribe_file(wav_io)
    
    return {"transcription": transcription}

