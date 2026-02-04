#!/usr/bin/env bash

if [ -z "$1" ]; then
    echo "Usage: $0 <text>" >&2
    exit 1
fi

TEXT="$*"

curl -s -X POST http://localhost:8080/chat \
    -H "Content-Type: application/json" \
    -d "{\"text\": \"$TEXT\"}" \
    -D - --output /dev/null 2>&1 | \
    grep -i x-response-text | cut -d' ' -f2 | python3 -c "import sys,urllib.parse; print(urllib.parse.unquote(sys.stdin.read()))"

