# VOICEVOX TTS API

FastAPI + voicevox_core を使ったテキスト音声合成 API サーバー。

## 必要環境

- Python 3.10+
- Linux x86_64
- NVIDIA GPU + CUDA ドライバ（GPU版）

## セットアップ

```bash
./setup.sh
```

以下が自動で行われます:

- Python 仮想環境 (`.venv`) の作成
- 依存パッケージのインストール（FastAPI, uvicorn, voicevox_core）
- ONNX Runtime (CUDA)・Open JTalk 辞書・音声モデル (VVM) のダウンロード

## サーバー起動

```bash
source .venv/bin/activate
uvicorn main:app --host 0.0.0.0 --port 8000
```

## API

### `POST /tts`

テキストから WAV 音声を生成する。

**リクエスト:**

```json
{
  "text": "こんにちは",
  "speaker_id": 0
}
```

| フィールド | 型 | 必須 | 説明 |
|---|---|---|---|
| `text` | string | はい | 読み上げるテキスト |
| `speaker_id` | int | いいえ (デフォルト: 0) | 話者スタイル ID |

**レスポンス:** `audio/wav`

**例:**

```bash
curl -X POST http://localhost:8000/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "こんにちは"}' \
  --output test.wav
```

### `POST /chat`

ユーザーのテキストを Ollama (gemma3) に送り、返答を VOICEVOX で音声合成して WAV を返す。

**リクエスト:**

```json
{
  "text": "今日の天気はどうですか？",
  "speaker_id": 0
}
```

| フィールド | 型 | 必須 | 説明 |
|---|---|---|---|
| `text` | string | はい | Ollama に送るテキスト |
| `speaker_id` | int | いいえ (デフォルト: 0) | 話者スタイル ID |

**レスポンス:** `audio/wav`

レスポンスヘッダー `X-Response-Text` に Ollama の返答テキスト（URL エンコード済み）が含まれる。

**例:**

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "今日の天気はどうですか？"}' \
  --output reply.wav
```

### ファイル保存せずに直接再生する

`/tts` と `/chat` はどちらも `audio/wav` を返すため、同じ方法で直接再生できる。

#### curl + aplay (Linux)

```bash
# /tts
curl -s -X POST http://localhost:8080/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"こんにちは","speaker_id":0}' | aplay

# /chat（応答テキストもヘッダーから取得）
curl -s -D - -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"text":"今日の天気はどうですか？","speaker_id":0}' \
  | sed '1,/^\r$/d' | aplay
```

#### JavaScript (ブラウザ)

```javascript
// /tts
const res = await fetch("http://localhost:8080/tts", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: "こんにちは", speaker_id: 0 }),
});
const blob = await res.blob();
new Audio(URL.createObjectURL(blob)).play();

// /chat（応答テキストも取得）
const chatRes = await fetch("http://localhost:8080/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: "今日の天気はどうですか？", speaker_id: 0 }),
});
const replyText = decodeURIComponent(chatRes.headers.get("X-Response-Text"));
console.log("応答:", replyText);
const chatBlob = await chatRes.blob();
new Audio(URL.createObjectURL(chatBlob)).play();
```

#### Python

```python
import requests
import simpleaudio as sa
import io, wave
from urllib.parse import unquote

def play_wav(content):
    with wave.open(io.BytesIO(content), "rb") as wf:
        audio = sa.WaveObject(wf.readframes(wf.getnframes()),
                              wf.getnchannels(), wf.getsampwidth(), wf.getframerate())
        audio.play().wait_done()

# /tts
res = requests.post("http://localhost:8080/tts",
                     json={"text": "こんにちは", "speaker_id": 0})
play_wav(res.content)

# /chat（応答テキストも取得）
chat_res = requests.post("http://localhost:8080/chat",
                          json={"text": "今日の天気はどうですか？", "speaker_id": 0})
reply_text = unquote(chat_res.headers["X-Response-Text"])
print("応答:", reply_text)
play_wav(chat_res.content)
```

### `GET /speakers`

利用可能な話者一覧を返す。

```bash
curl http://localhost:8000/speakers
```
