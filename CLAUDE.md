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

単一ファイル構成（`main.py`）。FastAPI の lifespan で VOICEVOX Synthesizer を初期化し、グローバル変数 `synthesizer` で保持する。起動時に `voicevox_core/models/vvms/` 内の全 `.vvm` ファイルを動的にロードする。

### エンドポイント

- `POST /tts` — テキスト→WAV音声合成
- `POST /chat` — Ollama (gemma3) でチャット応答を生成し、librosa でピッチシフト（+3半音）してWAV返却。`X-Response-Text` ヘッダーに応答テキスト（URLエンコード済み）を含む
- `GET /speakers` — 利用可能な話者一覧
- `GET /usage` — API ドキュメント（JSON）

### リソース配置

VOICEVOX 関連リソース（ONNX Runtime, Open JTalk辞書, VVMモデル）は `voicevox_core/` ディレクトリに配置。`setup.sh` で自動ダウンロードされる。手動起動時は `LD_LIBRARY_PATH` に `voicevox_core/onnxruntime/lib` と `voicevox_core/additional_libraries` を含める必要がある。

## Requirements

- Python 3.10+, Linux x86_64, NVIDIA GPU + CUDA ドライバ
- Ollama が起動済みで gemma3 モデルがプルされていること（`/chat` 利用時）
