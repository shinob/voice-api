import io
import re
import wave
from contextlib import asynccontextmanager
from pathlib import Path
import time
from datetime import datetime, timezone, timedelta
from urllib.parse import quote
import json

import librosa
import numpy as np
import ollama
import soundfile as sf
from fastapi import FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from voicevox_core.blocking import Onnxruntime, OpenJtalk, Synthesizer, VoiceModelFile
from weather import get_today_forecast

# RAG 設定
EMBEDDING_MODEL = "bge-m3"
RAG_TOP_K = 3
RAG_THRESHOLD = 0.3

BASE_DIR = Path(__file__).parent
CORE_DIR = BASE_DIR / "voicevox_core"
KNOWLEDGE_FILE = BASE_DIR / "knowledge.txt"
READING_DICT_FILE = BASE_DIR / "reading_dict.txt"
CHAT_LOG_FILE = BASE_DIR / "chat_log.jsonl"
WEATHER_AREA_CODE = "3220100"  # 松江市

synthesizer: Synthesizer | None = None
last_chat: dict | None = None  # {"user": str, "reply": str, "time": float}
knowledge_entries: list[dict] | None = None  # {"name": str, "text": str, "embedding": np.ndarray}


@asynccontextmanager
async def lifespan(app: FastAPI):
    global synthesizer, knowledge_entries

    ort = Onnxruntime.load_once(filename=Onnxruntime.LIB_VERSIONED_FILENAME)
    ojt = OpenJtalk(CORE_DIR / "dict" / "open_jtalk_dic_utf_8-1.11")
    synthesizer = Synthesizer(ort, ojt)

    for vvm_path in sorted(CORE_DIR.glob("models/vvms/*.vvm")):
        model = VoiceModelFile.open(vvm_path)
        synthesizer.load_voice_model(model)

    # knowledge.txt をパースしてベクトル化
    if KNOWLEDGE_FILE.exists():
        content = KNOWLEDGE_FILE.read_text(encoding="utf-8").strip()
        if content:
            entries = _parse_knowledge_entries(content)
            if entries:
                texts = [e["text"] for e in entries]
                embeddings = _embed_texts(texts)
                for entry, emb in zip(entries, embeddings):
                    entry["embedding"] = emb
                knowledge_entries = entries
                print(f"Loaded {len(knowledge_entries)} knowledge entries")

    yield

    synthesizer = None
    knowledge_entries = None


app = FastAPI(title="VOICEVOX TTS API", lifespan=lifespan)

PITCH_SHIFT_SEMITONES = 2
REPLY_MAX_CHARS = 200
VOICEVOX_CREDIT = "VOICEVOX:四国めたん"


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
    speaker_id: int = 2


@app.post("/tts")
def tts(req: TTSRequest):
    if synthesizer is None:
        raise HTTPException(status_code=500, detail="Synthesizer is not initialized")

    try:
        wav = synthesizer.tts(req.text, style_id=req.speaker_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")

    return StreamingResponse(
        io.BytesIO(wav),
        media_type="audio/wav",
        headers={"X-Voice-Credit": quote(VOICEVOX_CREDIT)},
    )


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


def _parse_knowledge_entries(content: str) -> list[dict]:
    """knowledge.txt をエントリごとに分割"""
    entries = []
    current_name = None
    current_lines = []

    for line in content.splitlines():
        # 行頭が非空白で始まり:で終わる → 新エントリの開始
        if line and not line[0].isspace() and line.rstrip().endswith(":"):
            if current_name and current_lines:
                entries.append({
                    "name": current_name,
                    "text": f"{current_name}:\n" + "\n".join(current_lines),
                })
            current_name = line.rstrip().rstrip(":")
            current_lines = []
        elif current_name is not None:
            current_lines.append(line)

    if current_name and current_lines:
        entries.append({
            "name": current_name,
            "text": f"{current_name}:\n" + "\n".join(current_lines),
        })

    return entries


def _embed_text(text: str) -> np.ndarray:
    """単一テキストの埋め込みを生成"""
    response = ollama.embed(model=EMBEDDING_MODEL, input=text)
    return np.array(response.embeddings[0], dtype=np.float32)


def _embed_texts(texts: list[str]) -> list[np.ndarray]:
    """複数テキストの一括埋め込み生成"""
    response = ollama.embed(model=EMBEDDING_MODEL, input=texts)
    return [np.array(emb, dtype=np.float32) for emb in response.embeddings]


def _cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    """コサイン類似度を計算"""
    return float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))


def _search_relevant_knowledge(query: str) -> tuple[list[dict], list[dict]]:
    """コサイン類似度で関連エントリを検索。(entries, scores) を返す"""
    if not knowledge_entries:
        return [], []

    query_embedding = _embed_text(query)
    scored = []
    for entry in knowledge_entries:
        score = _cosine_similarity(query_embedding, entry["embedding"])
        if score >= RAG_THRESHOLD:
            scored.append({"entry": entry, "score": score})

    scored.sort(key=lambda x: x["score"], reverse=True)
    top_items = scored[:RAG_TOP_K]
    entries = [item["entry"] for item in top_items]
    scores = [{"name": item["entry"]["name"], "score": round(item["score"], 3)} for item in top_items]
    return entries, scores


def _log_chat(user: str, reply: str, knowledge_used: list[dict] | None = None) -> None:
    """会話をJSONL形式でログに記録"""
    now = datetime.now(timezone(timedelta(hours=9)))
    record = {
        "timestamp": now.isoformat(),
        "user": user,
        "reply": reply,
    }
    if knowledge_used:
        record["knowledge_used"] = knowledge_used
    with CHAT_LOG_FILE.open("a", encoding="utf-8") as f:
        f.write(json.dumps(record, ensure_ascii=False) + "\n")


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


def _build_system_prompt(user_query: str | None = None) -> tuple[str, list[dict]]:
    """システムプロンプトを構築。(prompt, knowledge_used) を返す"""
    now = datetime.now(timezone(timedelta(hours=9)))
    date_str = now.strftime("%Y年%m月%d日 %H時%M分")
    base = f"現在の日時は{date_str}（日本時間）です。100文字以内で簡潔に要点のみを短く回答してください。URLやリンクは絶対に含めないでください。"

    weather = _get_weather_text()
    if weather:
        base += f"\n\n{weather}"

    knowledge_used = []
    # RAG: ユーザークエリがあれば関連エントリのみを取得
    if user_query and knowledge_entries:
        relevant, scores = _search_relevant_knowledge(user_query)
        if relevant:
            knowledge_text = "\n\n".join(e["text"] for e in relevant)
            base += f"\n\n以下はあなたが持つ確実な追加知識です。関連する質問にはこの情報を優先して回答してください。この追加知識に含まれない情報を使って回答する場合は「正確ではないかもしれませんが」と前置きしてください:\n{knowledge_text}"
            knowledge_used = scores
    elif KNOWLEDGE_FILE.exists() and not knowledge_entries:
        # RAG が無効の場合は従来通り全文を使用
        knowledge = KNOWLEDGE_FILE.read_text(encoding="utf-8").strip()
        if knowledge:
            base += f"\n\n以下はあなたが持つ確実な追加知識です。関連する質問にはこの情報を優先して回答してください。この追加知識に含まれない情報を使って回答する場合は「正確ではないかもしれませんが」と前置きしてください:\n{knowledge}"
    return base, knowledge_used


class ChatRequest(BaseModel):
    text: str
    speaker_id: int = 2


@app.post("/chat")
def chat(req: ChatRequest):
    if synthesizer is None:
        raise HTTPException(status_code=500, detail="Synthesizer is not initialized")

    global last_chat
    system_prompt, knowledge_used = _build_system_prompt(user_query=req.text)
    messages = [{"role": "system", "content": system_prompt}]
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
    _log_chat(req.text, reply_text, knowledge_used if knowledge_used else None)
    reading_text = re.sub(r"[（(][^）)]*[）)]", "", reply_text)
    reading_text = _apply_reading_dict(reading_text)

    try:
        wav = _synthesize_long(reading_text, style_id=req.speaker_id)
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Synthesis failed: {e}")

    wav = pitch_shift_wav(wav)

    return StreamingResponse(
        io.BytesIO(wav),
        media_type="audio/wav",
        headers={
            "X-Response-Text": quote(reply_text),
            "X-Voice-Credit": quote(VOICEVOX_CREDIT),
        },
    )


@app.get("/speakers")
def speakers():
    if synthesizer is None:
        raise HTTPException(status_code=500, detail="Synthesizer is not initialized")

    return synthesizer.metas


@app.get("/usage")
def usage():
    return {
        "credit": VOICEVOX_CREDIT,
        "endpoints": [
            {
                "path": "/tts",
                "method": "POST",
                "description": "テキストからWAV音声を合成する",
                "request_body": {
                    "text": {"type": "string", "required": True, "description": "合成するテキスト"},
                    "speaker_id": {"type": "integer", "default": 2, "description": "話者ID（デフォルト: 四国めたん ノーマル）"},
                },
                "response": "audio/wav",
                "response_headers": {
                    "X-Voice-Credit": "音声合成エンジンのクレジット表記",
                },
                "example": 'curl -X POST http://localhost:8080/tts -H "Content-Type: application/json" -d \'{"text": "こんにちは", "speaker_id": 2}\' -o output.wav',
            },
            {
                "path": "/chat",
                "method": "POST",
                "description": "Ollama (gemma3) でチャット応答を生成し、ピッチシフト済みWAV音声を返す。応答テキストはX-Response-Textヘッダーに含まれる",
                "request_body": {
                    "text": {"type": "string", "required": True, "description": "チャットメッセージ"},
                    "speaker_id": {"type": "integer", "default": 2, "description": "話者ID（デフォルト: 四国めたん ノーマル）"},
                },
                "response": "audio/wav",
                "response_headers": {
                    "X-Response-Text": "URLエンコードされた応答テキスト",
                    "X-Voice-Credit": "音声合成エンジンのクレジット表記",
                },
                "example": 'curl -X POST http://localhost:8080/chat -H "Content-Type: application/json" -d \'{"text": "東京の天気は？", "speaker_id": 2}\' -o reply.wav -D -',
            },
            {
                "path": "/speakers",
                "method": "GET",
                "description": "利用可能な話者の一覧を返す",
                "response": "application/json",
                "example": "curl http://localhost:8080/speakers",
            },
            {
                "path": "/usage",
                "method": "GET",
                "description": "このAPI使用方法を返す",
                "response": "application/json",
                "example": "curl http://localhost:8080/usage",
            },
        ],
    }
