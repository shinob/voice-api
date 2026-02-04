# VOICEVOX TTS API

FastAPI + voicevox_core を使ったテキスト音声合成 API サーバー。Ollama (gemma3) を使ったチャット機能も備える。

## クレジット

このプロジェクトは以下の音声合成エンジンを使用しています：

**VOICEVOX:四国めたん**

## 必要環境

- Python 3.10+
- Linux x86_64
- NVIDIA GPU + CUDA ドライバ（GPU版）
- Ollama + gemma3 モデル（`/chat` エンドポイント使用時）
- Ollama + bge-m3 モデル（`knowledge.txt` による RAG 使用時）

## セットアップ

```bash
./setup.sh
```

以下が自動で行われます:

- Python 仮想環境 (`.venv`) の作成
- 依存パッケージのインストール（FastAPI, uvicorn, voicevox_core, librosa 等）
- ONNX Runtime (CUDA)・Open JTalk 辞書・音声モデル (VVM) のダウンロード

## サーバー起動

```bash
./run.sh
```

または手動で:

```bash
source .venv/bin/activate
CORE_DIR="$(pwd)/voicevox_core"
export LD_LIBRARY_PATH="$CORE_DIR/onnxruntime/lib:$CORE_DIR/additional_libraries${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
uvicorn main:app --host 0.0.0.0 --port 8080
```

## API

### `POST /tts`

テキストから WAV 音声を生成する。

**リクエスト:**

```json
{
  "text": "こんにちは",
  "speaker_id": 2
}
```

| フィールド | 型 | 必須 | 説明 |
|---|---|---|---|
| `text` | string | はい | 読み上げるテキスト |
| `speaker_id` | int | いいえ (デフォルト: 2) | 話者スタイル ID（デフォルト: 四国めたん ノーマル） |

**レスポンス:** `audio/wav`

**レスポンスヘッダー:**

| ヘッダー | 説明 |
|---|---|
| `X-Voice-Credit` | 音声合成エンジンのクレジット表記 |

**例:**

```bash
curl -X POST http://localhost:8080/tts \
  -H "Content-Type: application/json" \
  -d '{"text": "こんにちは"}' \
  --output test.wav
```

### `POST /chat`

ユーザーのテキストを Ollama (gemma3) に送り、返答を VOICEVOX で音声合成して WAV を返す。
音声は +2 半音ピッチシフトされる。10秒以内の連続した会話はコンテキストが維持される。

**リクエスト:**

```json
{
  "text": "今日の天気はどうですか？",
  "speaker_id": 2
}
```

| フィールド | 型 | 必須 | 説明 |
|---|---|---|---|
| `text` | string | はい | Ollama に送るテキスト |
| `speaker_id` | int | いいえ (デフォルト: 2) | 話者スタイル ID（デフォルト: 四国めたん ノーマル） |

**レスポンス:** `audio/wav`

**レスポンスヘッダー:**

| ヘッダー | 説明 |
|---|---|
| `X-Response-Text` | Ollama の返答テキスト（URL エンコード済み） |
| `X-Voice-Credit` | 音声合成エンジンのクレジット表記 |

**例:**

```bash
curl -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"text": "今日の天気はどうですか？"}' \
  -D - \
  --output reply.wav
```

### `GET /speakers`

利用可能な話者一覧を返す。

```bash
curl http://localhost:8080/speakers
```

### `GET /usage`

API 使用方法を JSON で返す。

```bash
curl http://localhost:8080/usage
```

## 主な話者 ID

| ID | キャラクター | スタイル |
|---|---|---|
| 0 | 四国めたん | あまあま |
| 2 | 四国めたん | ノーマル（デフォルト） |
| 1 | ずんだもん | あまあま |
| 3 | ずんだもん | ノーマル |

全話者一覧は `/speakers` エンドポイントで確認できます。

## オプション機能

### 追加知識 (`knowledge.txt`)

プロジェクトルートに `knowledge.txt` を配置すると、`/chat` エンドポイントで RAG により関連エントリが検索され、システムプロンプトに注入されます。

- 起動時に `bge-m3` で全エントリをベクトル化
- 質問時にコサイン類似度で上位 3 件（閾値 0.3 以上）を取得
- エントリ形式: 行頭が非空白で `:` 終わりの行が見出し、以降がその内容

`bge-m3` が利用できない場合は `knowledge.txt` 全文をそのままプロンプトに含めるフォールバック動作になります。

### 読み辞書 (`reading_dict.txt`)

プロジェクトルートに `reading_dict.txt` を配置すると、音声合成前にテキスト置換が行われます。
形式: `置換前<TAB>置換後`（1行1エントリ、`#` で始まる行はコメント）

### 会話ログ (`chat_log.jsonl`)

`/chat` エンドポイントへのリクエストごとに、ユーザーの質問・AI の回答・使用された knowledge エントリとスコアを JSONL 形式で自動記録します。

## 直接再生する方法

`/tts` と `/chat` はどちらも `audio/wav` を返すため、同じ方法で直接再生できる。

### curl + aplay (Linux)

```bash
# /tts
curl -s -X POST http://localhost:8080/tts \
  -H "Content-Type: application/json" \
  -d '{"text":"こんにちは","speaker_id":2}' | aplay

# /chat（応答テキストもヘッダーから取得）
curl -s -D - -X POST http://localhost:8080/chat \
  -H "Content-Type: application/json" \
  -d '{"text":"今日の天気はどうですか？","speaker_id":2}' \
  | sed '1,/^\r$/d' | aplay
```

### JavaScript (ブラウザ)

```javascript
// /tts
const res = await fetch("http://localhost:8080/tts", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: "こんにちは", speaker_id: 2 }),
});
const blob = await res.blob();
new Audio(URL.createObjectURL(blob)).play();

// /chat（応答テキストも取得）
const chatRes = await fetch("http://localhost:8080/chat", {
  method: "POST",
  headers: { "Content-Type": "application/json" },
  body: JSON.stringify({ text: "今日の天気はどうですか？", speaker_id: 2 }),
});
const replyText = decodeURIComponent(chatRes.headers.get("X-Response-Text"));
console.log("応答:", replyText);
const chatBlob = await chatRes.blob();
new Audio(URL.createObjectURL(chatBlob)).play();
```

### Python

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
                     json={"text": "こんにちは", "speaker_id": 2})
play_wav(res.content)

# /chat（応答テキストも取得）
chat_res = requests.post("http://localhost:8080/chat",
                          json={"text": "今日の天気はどうですか？", "speaker_id": 2})
reply_text = unquote(chat_res.headers["X-Response-Text"])
print("応答:", reply_text)
play_wav(chat_res.content)
```

## ライセンス

使用しているライブラリ・リソースのライセンスについては `ライセンス.md` を参照してください。
