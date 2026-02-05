# 動作環境要件

## ハードウェア要件

| 項目 | 要件 |
|------|------|
| **CPU アーキテクチャ** | x86_64 (AMD64) のみ。VOICEVOX Core の whl が `manylinux_2_34_x86_64` 用のため |
| **GPU** | NVIDIA GPU + CUDA ドライバ必須。`setup.sh` で `--devices cuda` を指定して ONNX Runtime をダウンロードしている |
| **VRAM** | VOICEVOX の推論 + Ollama で gemma3 と bge-m3 を動かすため、少なくとも 8GB 以上を推奨（gemma3 のモデルサイズに依存） |
| **メモリ (RAM)** | librosa による音声処理や Ollama の実行があるため十分なメモリが必要 |

## OS 要件

- **Linux** のみ対応（`download-linux-x64`, `.so` ライブラリ前提、`LD_LIBRARY_PATH` 使用）
- glibc **2.34 以上**（whl のファイル名 `manylinux_2_34` より。Ubuntu 22.04+ / Fedora 35+ 相当）

## ソフトウェア要件

### 1. Python

- **Python 3.10 以上**（voicevox_core の whl が `cp310-abi3` = CPython 3.10+ の Stable ABI）
- `python3 -m venv` が使えること

### 2. Python パッケージ (`requirements.txt` + voicevox_core)

| パッケージ | 用途 |
|-----------|------|
| `fastapi` | Web API フレームワーク |
| `uvicorn[standard]` | ASGI サーバー |
| `ollama` | Ollama Python クライアント |
| `librosa` | ピッチシフト処理 |
| `soundfile` | WAV 読み書き |
| `voicevox_core` | 音声合成エンジン（setup.sh が whl からインストール） |
| `requests` | weather.py で気象庁 API 呼び出し（requirements.txt に未記載だが他パッケージの依存で入る可能性が高い） |
| `numpy` | main.py で直接 import（librosa の依存で入る） |

### 3. Ollama

- **Ollama** がインストール済みかつ起動していること（デフォルトで `http://localhost:11434`）
- 以下のモデルを事前にプル済み:
  - **`gemma3`** — `/chat` エンドポイントのチャット応答生成に使用
  - **`bge-m3`** — `knowledge.txt` の RAG（ベクトル検索）用埋め込みモデル

### 4. VOICEVOX 関連リソース（`setup.sh` が自動ダウンロード）

- ONNX Runtime（CUDA 版）
- Open JTalk 辞書 (`open_jtalk_dic_utf_8-1.11`)
- VVM 音声モデルファイル群

すべて `voicevox_core/` ディレクトリに配置され、実行時に `LD_LIBRARY_PATH` で ONNX Runtime のライブラリパスを通す必要がある。

### 5. NVIDIA CUDA

- NVIDIA GPU ドライバがインストール済みであること
- CUDA ランタイムが利用可能であること（ONNX Runtime の CUDA 版が動作するために必要）

### 6. ネットワーク

- セットアップ時: GitHub から VOICEVOX 関連ファイルをダウンロード
- 実行時: 気象庁 API (`www.jma.go.jp`) へのアクセス、Ollama (`localhost:11434`) への接続

## 注意事項

- `weather.py` が `requests` を使っているが `requirements.txt` に含まれていない。明示的に追加するのが安全。
- `/tts` エンドポイントのみ使う場合は Ollama は不要。`/chat` を使う場合のみ gemma3 と bge-m3 が必要。
- systemd サービスとして動かす場合は `voice-api.service` を使用するが、`WorkingDirectory` が `/opt/voice-api` 固定になっている。
