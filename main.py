import io
import wave
from contextlib import asynccontextmanager
from pathlib import Path
from urllib.parse import quote

import librosa
import numpy as np
import ollama
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile

BASE_DIR = Path(__file__).parent
CORE_DIR = BASE_DIR / "voicevox_core"

synthesizer: Synthesizer | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global synthesizer

    ort = Onnxruntime.load_once(filename=Onnxruntime.LIB_VERSIONED_FILENAME)
    ojt = OpenJtalk(CORE_DIR / "dict" / "open_jtalk_dic_utf_8-1.11")
    synthesizer = Synthesizer(ort, ojt)

    for vvm_path in sorted(CORE_DIR.glob("models/vvms/*.vvm")):
        model = VoiceModelFile.open(vvm_path)
        synthesizer.load_voice_model(model)

    yield

    synthesizer = None


app = FastAPI(title="VOICEVOX TTS API", lifespan=lifespan)

PITCH_SHIFT_SEMITONES = 3


def pitch_shift_wav(wav_bytes: bytes, n_steps: float = PITCH_SHIFT_SEMITONES) -> bytes:
    with wave.open(io.BytesIO(wav_bytes), "rb") as wf:
        sr = wf.getframerate()
    audio, _ = sf.read(io.BytesIO(wav_bytes), dtype="float32")
    shifted = librosa.effects.pitch_shift(audio, sr=sr, n_steps=n_steps)
    buf = io.BytesIO()
    sf.write(buf, shifted, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


class TTSRequest(BaseModel):
    text: str
    speaker_id: int = 0


@app.post("/tts")
def tts(req: TTSRequest):
    if synthesizer is None:
        raise HTTPException(status_code=500, detail="Synthesizer is not initialized")

    try:
        wav = synthesizer.tts(req.text, style_id=req.speaker_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")

    return StreamingResponse(io.BytesIO(wav), media_type="audio/wav")


class ChatRequest(BaseModel):
    text: str
    speaker_id: int = 0


@app.post("/chat")
def chat(req: ChatRequest):
    if synthesizer is None:
        raise HTTPException(status_code=500, detail="Synthesizer is not initialized")

    try:
        response = ollama.chat(
            model="gemma3",
            messages=[
                {"role": "system", "content": "100文字以内で簡潔に要点のみを短く回答してください。URLやリンクは絶対に含めないでください。"},
                {"role": "user", "content": req.text},
            ],
        )
        reply_text = response.message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama chat failed: {e}")

    reading_text = reply_text.replace("京店", "きょうみせ")

    try:
        wav = synthesizer.tts(reading_text, style_id=req.speaker_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")

    wav = pitch_shift_wav(wav)

    return StreamingResponse(
        io.BytesIO(wav),
        media_type="audio/wav",
        headers={"X-Response-Text": quote(reply_text)},
    )


@app.get("/speakers")
def speakers():
    if synthesizer is None:
        raise HTTPException(status_code=500, detail="Synthesizer is not initialized")

    return synthesizer.metas


@app.get("/usage")
def usage():
    return {
        "endpoints": [
            {
                "path": "/tts",
                "method": "POST",
                "description": "テキストからWAV音声を合成する",
                "request_body": {
                    "text": {"type": "string", "required": True, "description": "合成するテキスト"},
                    "speaker_id": {"type": "integer", "default": 0, "description": "話者ID"},
                },
                "response": "audio/wav",
                "example": 'curl -X POST http://localhost:8000/tts -H "Content-Type: application/json" -d \'{"text": "こんにちは", "speaker_id": 0}\' -o output.wav',
            },
            {
                "path": "/chat",
                "method": "POST",
                "description": "Ollama (gemma3) でチャット応答を生成し、ピッチシフト済みWAV音声を返す。応答テキストはX-Response-Textヘッダーに含まれる",
                "request_body": {
                    "text": {"type": "string", "required": True, "description": "チャットメッセージ"},
                    "speaker_id": {"type": "integer", "default": 0, "description": "話者ID"},
                },
                "response": "audio/wav",
                "response_headers": {
                    "X-Response-Text": "URLエンコードされた応答テキスト",
                },
                "example": 'curl -X POST http://localhost:8000/chat -H "Content-Type: application/json" -d \'{"text": "東京の天気は？", "speaker_id": 0}\' -o reply.wav -D -',
            },
            {
                "path": "/speakers",
                "method": "GET",
                "description": "利用可能な話者の一覧を返す",
                "response": "application/json",
                "example": "curl http://localhost:8000/speakers",
            },
            {
                "path": "/usage",
                "method": "GET",
                "description": "このAPI使用方法を返す",
                "response": "application/json",
                "example": "curl http://localhost:8000/usage",
            },
        ],
    }
