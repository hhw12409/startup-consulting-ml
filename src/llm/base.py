"""
📁 src/llm/base.py
====================
LLM 추상 인터페이스.

[패턴] Strategy — Claude API / Ollama / 기타 LLM을 동일한 인터페이스로 교체 가능
[역할] 모든 LLM 클라이언트가 이 인터페이스를 구현합니다.

아키텍처:
  XGBoost (숫자 예측) → LLM (자연어 해석)
  - ML 모델: "생존확률 0.72, 리스크 0.35"
  - LLM:     "35세 카페 창업의 1년 생존확률은 72%로 양호합니다..."
"""

from abc import ABC, abstractmethod
from typing import Optional


class BaseLLM(ABC):
    """
    LLM 공통 인터페이스.

    사용법:
        llm: BaseLLM = ClaudeClient()         # Claude API
        llm: BaseLLM = OllamaClient()         # 로컬 Ollama
        response = llm.generate("분석해주세요", system="당신은 창업 컨설턴트입니다")
    """

    @abstractmethod
    def generate(
            self,
            prompt: str,
            system: Optional[str] = None,
            max_tokens: int = 2000,
            temperature: float = 0.7,
    ) -> str:
        """
        텍스트 생성.

        Args:
            prompt: 사용자 프롬프트
            system: 시스템 프롬프트 (역할 지정)
            max_tokens: 최대 생성 토큰 수
            temperature: 창의성 (0=정확, 1=창의적)

        Returns:
            생성된 텍스트
        """
        ...

    @abstractmethod
    def is_available(self) -> bool:
        """이 LLM이 현재 사용 가능한지 확인"""
        ...

    @property
    @abstractmethod
    def name(self) -> str:
        """LLM 이름 (로깅용)"""
        ...