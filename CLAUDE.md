# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

FastAPI + voicevox_core による日本語テキスト音声合成 API サーバー。Ollama (gemma3) を使ったチャット→音声合成機能も備える。

## Commands

```bash
# セットアップ（venv作成、依存パッケージ・VOICEVOX関連リソースのダウンロード）
./setup.sh

# サーバー起動（LD_LIBRARY_PATH等を自動設定、ポート8080）
./run.sh

# systemd サービスとして起動
sudo systemctl start voice-api

# 手動起動
source .venv/bin/activate
CORE_DIR="$(pwd)/voicevox_core"
export LD_LIBRARY_PATH="$CORE_DIR/onnxruntime/lib:$CORE_DIR/additional_libraries${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
uvicorn main:app --host 0.0.0.0 --port 8080
```

## Architecture

2ファイル構成: `main.py`（FastAPI アプリ）と `weather.py`（気象庁API連携）。

### main.py

FastAPI の lifespan で VOICEVOX Synthesizer を初期化し、グローバル変数 `synthesizer` で保持。起動時に `voicevox_core/models/vvms/` 内の全 `.vvm` ファイルを動的にロードする。

**エンドポイント:**
- `POST /tts` — テキスト→WAV音声合成
- `POST /chat` — Ollama (gemma3) でチャット応答を生成し、ピッチシフト（+2半音）してWAV返却。`X-Response-Text` ヘッダーに応答テキスト（URLエンコード済み）を含む。10秒以内の連続会話はコンテキスト維持
- `GET /speakers` — 利用可能な話者一覧
- `GET /usage` — API ドキュメント（JSON）

**キーロジック:**
- `_synthesize_long()`: 長文を句点等で分割して合成・結合
- `_apply_reading_dict()`: `reading_dict.txt` による読み替え処理
- `_build_system_prompt()`: 現在日時・天気・RAG結果を含むシステムプロンプト生成。`(prompt, knowledge_used)` を返す
- `_search_relevant_knowledge()`: `bge-m3` の埋め込みとコサイン類似度で knowledge エントリを検索（閾値 0.3、上位 3 件）
- `_log_chat()`: 会話を `chat_log.jsonl` に JSONL 形式で記録
- 括弧除去: 音声合成前に `（）` `()` 内のテキストを除去し、読み上げをスキップ

### weather.py

気象庁の天気予報 API から今日の天気を取得。`WEATHER_AREA_CODE`（デフォルト: 松江市）の天気・気温・降水確率を返す。

### オプションファイル

- `knowledge.txt` — `/chat` で RAG により関連エントリを検索しプロンプトに注入。エントリは `bge-m3` で起動時にベクトル化
- `reading_dict.txt` — 音声合成前のテキスト置換辞書（TSV形式、`#` コメント対応）
- `chat_log.jsonl` — `/chat` の会話ログ（自動生成、JSONL形式）

### リソース配置

VOICEVOX 関連リソース（ONNX Runtime, Open JTalk辞書, VVMモデル）は `voicevox_core/` ディレクトリに配置。`setup.sh` で自動ダウンロードされる。手動起動時は `LD_LIBRARY_PATH` の設定が必須。

## Requirements

- Python 3.10+, Linux x86_64, NVIDIA GPU + CUDA ドライバ
- Ollama が起動済みで gemma3 モデルがプルされていること（`/chat` 利用時）
- Ollama に bge-m3 モデルがプルされていること（`knowledge.txt` による RAG 利用時）
