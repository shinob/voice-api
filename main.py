import io
import re
import wave
from contextlib import asynccontextmanager
from pathlib import Path
import time
from datetime import datetime, timezone, timedelta
from urllib.parse import quote

import librosa
import numpy as np
import ollama
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile
from weather import get_today_forecast

BASE_DIR = Path(__file__).parent
CORE_DIR = BASE_DIR / "voicevox_core"
KNOWLEDGE_FILE = BASE_DIR / "knowledge.txt"
READING_DICT_FILE = BASE_DIR / "reading_dict.txt"
WEATHER_AREA_CODE = "3220100"  # 松江市

synthesizer: Synthesizer | None = None
last_chat: dict | None = None  # {"user": str, "reply": str, "time": float}


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
REPLY_MAX_CHARS = 200


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


def _truncate_reply(text: str, max_chars: int = REPLY_MAX_CHARS) -> str:
    if len(text) <= max_chars:
        return text
    truncated = text[:max_chars]
    for sep in ["。", "、", ".", " "]:
        pos = truncated.rfind(sep)
        if pos > 0:
            return truncated[: pos + 1]
    return truncated


def _split_sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[。！？\n])", text)
    return [p for p in parts if p.strip()]


def _synthesize_long(text: str, style_id: int) -> bytes:
    sentences = _split_sentences(text)
    if not sentences:
        sentences = [text]

    segments: list[np.ndarray] = []
    sr = None
    for sentence in sentences:
        wav_bytes = synthesizer.tts(sentence.strip(), style_id=style_id)
        audio, cur_sr = sf.read(io.BytesIO(wav_bytes), dtype="float32")
        if sr is None:
            sr = cur_sr
        segments.append(audio)

    combined = np.concatenate(segments)
    buf = io.BytesIO()
    sf.write(buf, combined, sr, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def _apply_reading_dict(text: str) -> str:
    if not READING_DICT_FILE.exists():
        return text
    for line in READING_DICT_FILE.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split("\t", 1)
        if len(parts) == 2:
            text = text.replace(parts[0], parts[1])
    return text


def _get_weather_text() -> str:
    try:
        forecast = get_today_forecast(WEATHER_AREA_CODE)
        parts = [f"今日の{forecast['area']}の天気: {forecast['weather']}"]
        if forecast.get("temps"):
            temps = forecast["temps"]
            if temps.get("min") and temps.get("max"):
                parts.append(f"気温: {temps['min']}〜{temps['max']}℃")
            elif temps.get("max"):
                parts.append(f"最高気温: {temps['max']}℃")
        if forecast.get("pops"):
            pops = [p for p in forecast["pops"] if p]
            if pops:
                parts.append(f"降水確率: {'/'.join(pops)}%")
        return "。".join(parts)
    except Exception:
        return ""


def _build_system_prompt() -> str:
    now = datetime.now(timezone(timedelta(hours=9)))
    date_str = now.strftime("%Y年%m月%d日 %H時%M分")
    base = f"現在の日時は{date_str}（日本時間）です。100文字以内で簡潔に要点のみを短く回答してください。URLやリンクは絶対に含めないでください。"

    weather = _get_weather_text()
    if weather:
        base += f"\n\n{weather}"

    if KNOWLEDGE_FILE.exists():
        knowledge = KNOWLEDGE_FILE.read_text(encoding="utf-8").strip()
        if knowledge:
            base += f"\n\n以下はあなたが持つ確実な追加知識です。関連する質問にはこの情報を優先して回答してください。この追加知識に含まれない情報を使って回答する場合は「正確ではないかもしれませんが」と前置きしてください:\n{knowledge}"
    return base


class ChatRequest(BaseModel):
    text: str
    speaker_id: int = 0


@app.post("/chat")
def chat(req: ChatRequest):
    if synthesizer is None:
        raise HTTPException(status_code=500, detail="Synthesizer is not initialized")

    global last_chat
    messages = [{"role": "system", "content": _build_system_prompt()}]
    if last_chat and (time.monotonic() - last_chat["time"]) < 10:
        messages.append({"role": "user", "content": last_chat["user"]})
        messages.append({"role": "assistant", "content": last_chat["reply"]})
    messages.append({"role": "user", "content": req.text})

    try:
        response = ollama.chat(
            model="gemma3",
            messages=messages,
            keep_alive="30s",
        )
        reply_text = response.message.content
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Ollama chat failed: {e}")

    reply_text = _truncate_reply(reply_text)
    last_chat = {"user": req.text, "reply": reply_text, "time": time.monotonic()}
    reading_text = _apply_reading_dict(reply_text)

    try:
        wav = _synthesize_long(reading_text, style_id=req.speaker_id)
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
