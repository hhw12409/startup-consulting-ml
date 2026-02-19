"""
📁 src/llm/router.py
======================
LLM 라우터 — Ollama 단독 사용.

[패턴] Facade — LLM 호출을 단순한 인터페이스로 제공

동작:
  1. Ollama 실행 중이면 → Ollama 사용 (한국어 모델 자동 탐지)
  2. Ollama 미실행 → 규칙 기반 템플릿 폴백 (LLM 없이 동작)
"""

from typing import Optional

from src.llm.base import BaseLLM
from src.llm.ollama_client import OllamaClient
from src.utils.logger import get_logger

logger = get_logger(__name__)


class LLMRouter:
    """
    LLM 라우터 (Ollama 단독).

    사용법:
        router = LLMRouter()
        response = router.generate("분석해주세요")
        print(router.active_llm)  # "Ollama (gemma2:9b)"
    """

    def __init__(self):
        self._client = OllamaClient()
        self._available = self._client.is_available()

        if self._available:
            logger.info("✅ LLM 활성: %s", self._client.name)
        else:
            logger.warning("⚠️ Ollama 미실행 → 규칙 기반 템플릿 모드")
            logger.warning("   시작: ollama serve && ollama pull gemma2:9b")

    @property
    def active_llm(self) -> str:
        """현재 활성 LLM 이름"""
        return self._client.name if self._available else "규칙 기반 템플릿"

    @property
    def is_llm_available(self) -> bool:
        return self._available

    def generate(
            self,
            prompt: str,
            system: Optional[str] = None,
            max_tokens: int = 2000,
            temperature: float = 0.7,
    ) -> str:
        """텍스트 생성. Ollama 불가 시 폴백 메시지 반환."""
        if self._available:
            try:
                return self._client.generate(prompt, system, max_tokens, temperature)
            except Exception as e:
                logger.warning("Ollama 호출 실패: %s → 폴백", e)

        return "[LLM 미연결] Ollama를 실행해주세요: ollama serve && ollama pull gemma2:9b"