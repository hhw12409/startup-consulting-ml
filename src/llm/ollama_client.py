"""
📁 src/llm/ollama_client.py
==============================
Ollama 로컬 LLM 클라이언트.

[패턴] Strategy — BaseLLM 구현체
[역할] 외부 API 없이 로컬에서 LLM을 실행합니다.

설치:
  1. brew install ollama (macOS) 또는 https://ollama.com
  2. ollama pull gemma2:9b       (추천: 한국어 좋음, 32GB Mac)
  3. ollama serve                (서버 시작)

한국어 모델 우선순위 (32GB Mac 기준):
  1순위: EEVE-Korean-10.8B    — 한국어 최고, 6.5GB
  2순위: gemma2:9b            — 한국어 좋음, 5.4GB
  3순위: llama3.1:8b          — 한국어 보통, 4.7GB
  4순위: gemma2:2b            — 가벼움, 1.6GB (메모리 부족 시)

사용법:
    client = OllamaClient()
    if client.is_available():
        response = client.generate("창업 분석해주세요")
        print(client.name)  # "Ollama (EEVE-Korean-10.8B)"
"""

import requests
from typing import Optional

from src.llm.base import BaseLLM
from src.utils.logger import get_logger

logger = get_logger(__name__)


# 한국어 성능 순으로 정렬 (32GB Mac 기준)
KOREAN_MODEL_PRIORITY = [
    "EEVE-Korean-10.8B",           # 한국어 특화, 최고 품질
    "eeve-korean-10.8b:latest",    # 태그 형식
    "gemma2:9b",                   # 구글, 한국어 좋음
    "gemma2:latest",               # gemma2 기본
    "llama3.1:8b",                 # Meta, 한국어 보통
    "llama3.1:latest",
    "gemma2:2b",                   # 경량 폴백
    "llama3.2:3b",                 # 경량 폴백
]


class OllamaClient(BaseLLM):
    """
    Ollama 로컬 LLM 클라이언트.

    한국어 모델을 자동 탐지하여 최적의 모델을 선택합니다.
    """

    DEFAULT_URL = "http://localhost:11434"

    def __init__(self, model: str = None, base_url: str = None):
        self._url = base_url or self.DEFAULT_URL
        self._model = model  # None이면 자동 탐지
        self._available = None

    @property
    def name(self) -> str:
        model = self._model or "미탐지"
        return f"Ollama ({model})"

    def is_available(self) -> bool:
        """Ollama 서버 실행 여부 확인 + 최적 모델 자동 탐지"""
        try:
            resp = requests.get(f"{self._url}/api/tags", timeout=3)
            if resp.status_code != 200:
                return False

            # 모델이 지정되지 않았으면 자동 탐지
            if not self._model:
                self._model = self._find_best_model(resp.json())

            self._available = self._model is not None
            return self._available

        except requests.ConnectionError:
            self._available = False
            return False

    def _find_best_model(self, tags_response: dict) -> Optional[str]:
        """
        설치된 모델 중 한국어 최적 모델을 찾습니다.

        우선순위: EEVE-Korean > gemma2:9b > llama3.1:8b > 기타
        """
        installed = []
        for m in tags_response.get("models", []):
            name = m.get("name", "")
            installed.append(name)

        if not installed:
            logger.warning("Ollama에 설치된 모델 없음. 'ollama pull gemma2:9b' 실행 필요")
            return None

        logger.info("Ollama 설치된 모델: %s", installed)

        # 우선순위 기반 매칭
        for priority_model in KOREAN_MODEL_PRIORITY:
            for inst in installed:
                if priority_model.lower() in inst.lower():
                    logger.info("✅ 최적 한국어 모델 선택: %s", inst)
                    return inst

        # 우선순위에 없으면 첫 번째 모델 사용
        fallback = installed[0]
        logger.info("한국어 특화 모델 없음 → 폴백: %s", fallback)
        return fallback

    def list_models(self) -> list[str]:
        """설치된 모델 목록 조회"""
        try:
            resp = requests.get(f"{self._url}/api/tags", timeout=3)
            if resp.status_code == 200:
                return [m["name"] for m in resp.json().get("models", [])]
        except requests.ConnectionError:
            pass
        return []

    def generate(
            self,
            prompt: str,
            system: Optional[str] = None,
            max_tokens: int = 2000,
            temperature: float = 0.7,
    ) -> str:
        if not self._model:
            if not self.is_available():
                raise RuntimeError("Ollama 서버 미실행. 'ollama serve'를 먼저 실행하세요.")

        payload = {
            "model": self._model,
            "prompt": prompt,
            "stream": False,
            "options": {
                "temperature": temperature,
                "num_predict": max_tokens,
            },
        }
        if system:
            payload["system"] = system

        try:
            resp = requests.post(
                f"{self._url}/api/generate",
                json=payload,
                timeout=180,  # 큰 모델은 시간이 걸릴 수 있음
            )
            resp.raise_for_status()
            text = resp.json().get("response", "")
            logger.debug("Ollama 응답 (%s): %d자", self._model, len(text))
            return text

        except Exception as e:
            logger.error("Ollama 호출 실패 (%s): %s", self._model, e)
            raise