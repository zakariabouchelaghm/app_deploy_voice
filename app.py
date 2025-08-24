from fastapi import FastAPI, UploadFile, File
from speechbrain.inference import EncoderASR
import torchaudio




app = FastAPI(title="Arabic Digit Recognition API")

asr_model = EncoderASR.from_hparams(source="speechbrain/asr-wav2vec2-commonvoice-14-ar", savedir="pretrained_models/asr-wav2vec2-commonvoice-14-ar")
@app.post("/predict")
async def transcribe_audio(file: UploadFile = File(...)):
   tmp_input = "/tmp/input_audio.m4a"
    with open(tmp_input, "wb") as f:
        f.write(await file.read())

    # Load audio with torchaudio
    waveform, sr = torchaudio.load(tmp_input)
    if sr != 16000:
        import torchaudio.transforms as T
        resampler = T.Resample(sr, 16000)
        waveform = resampler(waveform)

    # Save as temporary wav file
    tmp_wav = "/tmp/audio.wav"
    torchaudio.save(tmp_wav, waveform, 16000)
    # Run ASR on WAV bytes
    transcription = asr_model.transcribe_file(tmp_wav)
    
    return {"transcription": transcription}

