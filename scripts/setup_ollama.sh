#!/bin/bash
# 📁 scripts/setup_ollama.sh
# =============================
# Ollama 한국어 모델 설치 스크립트.
#
# 실행: bash scripts/setup_ollama.sh
#
# 32GB Mac 추천 구성:
#   gemma2:9b (5.4GB) — 한국어 좋음, 범용
#
# 선택 설치:
#   EEVE-Korean-10.8B (6.5GB) — 한국어 최고 (Hugging Face에서 GGUF 변환 필요)

set -e

echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "🚀 Ollama 한국어 모델 설치"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 1. Ollama 설치 확인
if ! command -v ollama &> /dev/null; then
echo "❌ Ollama가 설치되어 있지 않습니다."
echo "   설치: brew install ollama"
echo "   또는: https://ollama.com/download"
exit 1
fi

echo "✅ Ollama 설치 확인"

# 2. Ollama 서버 시작 (백그라운드)
if ! curl -s http://localhost:11434/api/tags > /dev/null 2>&1; then
echo "📡 Ollama 서버 시작 중..."
ollama serve &
sleep 3
fi

echo "✅ Ollama 서버 실행 중"

# 3. 모델 설치
echo ""
echo "📥 gemma2:9b 설치 중 (5.4GB)..."
echo "   (한국어 성능 좋음, 범용 모델)"
ollama pull gemma2:9b

echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo "✅ 설치 완료!"
echo ""
echo "설치된 모델:"
ollama list
echo ""
echo "테스트:"
echo "  ollama run gemma2:9b '서울 강남구에서 카페 창업하려고 합니다. 조언해주세요.'"
echo ""
echo "프로젝트에서 사용:"
echo "  make serve  →  POST /api/v1/consult"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"