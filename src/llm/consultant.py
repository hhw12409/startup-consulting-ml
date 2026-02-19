"""
📁 src/llm/consultant.py
===========================
LLM 기반 창업 컨설턴트 서비스.

[패턴] Facade — ML 예측 + 데이터 통계 + LLM 해석을 하나의 인터페이스로 통합
[역할] 4가지 기능을 제공합니다:
  1. 종합 컨설팅 리포트
  2. 맞춤형 전략 제안
  3. Q&A 대화
  4. 경쟁업체 분석

아키텍처:
  사용자 입력 → ML 예측 (XGBoost)
             → 데이터 통계 (stores_raw.csv) ← NEW!
             → LLM 해석 (Ollama)
             → 자연어 응답
"""

from typing import Optional

from src.llm.router import LLMRouter
from src.llm.data_context import DataContext
from src.llm.rag_store import RAGStore
from src.llm.prompts import (
    SYSTEM_CONSULTANT,
    build_report_prompt,
    build_strategy_prompt,
    build_qa_prompt,
    build_competitor_prompt,
)
from src.utils.logger import get_logger

logger = get_logger(__name__)


class StartupConsultant:
    """
    LLM 기반 창업 컨설턴트.

    사용법:
        consultant = StartupConsultant()
        print(consultant.active_llm)  # "Ollama (gemma2:9b)" 또는 "규칙 기반"

        report = consultant.generate_report(input_data, prediction)
        strategy = consultant.suggest_strategy(input_data, prediction)
        answer = consultant.ask(question, input_data, prediction)
        analysis = consultant.analyze_competitors(input_data, prediction)
    """

    def __init__(self, router: LLMRouter = None, data_context: DataContext = None, rag_store: RAGStore = None):
        self._router = router or LLMRouter()
        self._context = data_context or DataContext()
        self._rag = rag_store or RAGStore()

        if self._context.is_available:
            logger.info("📊 데이터 컨텍스트 활성 (프롬프트에 실제 통계 주입)")
        else:
            logger.warning("📊 데이터 컨텍스트 비활성 (stores_raw.csv 없음)")

        if self._rag.doc_count > 0:
            logger.info("🔍 RAG 활성: %d건 벡터DB", self._rag.doc_count)
        else:
            logger.info("🔍 RAG 비활성 (벡터DB 없음, make build-rag 실행 필요)")

    @property
    def active_llm(self) -> str:
        return self._router.active_llm

    def _get_data_stats(self, input_data: dict, query: str = None) -> str:
        """입력 데이터에서 업종/지역을 추출하여 통계 + RAG 컨텍스트 생성"""
        category = input_data.get("business_category", "")
        district = input_data.get("district", "")

        parts = []

        # A) 통계 컨텍스트
        stats = self._context.get_context(category=category, district=district)
        if stats:
            parts.append(stats)

        # B) RAG 검색 컨텍스트
        if self._rag.doc_count > 0:
            search_query = query or f"{district} {category}"
            rag_context = self._rag.get_rag_context(search_query, top_k=5)
            if rag_context:
                parts.append(rag_context)

        return "\n\n".join(parts)

    # ================================================================
    # 1. 종합 컨설팅 리포트
    # ================================================================
    def generate_report(self, input_data: dict, prediction: dict) -> str:
        logger.info("컨설팅 리포트 생성 시작 (LLM: %s)", self.active_llm)

        if not self._router.is_llm_available:
            return self._fallback_report(input_data, prediction)

        data_stats = self._get_data_stats(input_data)
        prompt = build_report_prompt(input_data, prediction, data_context=data_stats)
        report = self._router.generate(prompt, system=SYSTEM_CONSULTANT)

        logger.info("컨설팅 리포트 생성 완료: %d자", len(report))
        return report

    # ================================================================
    # 2. 맞춤형 전략 제안
    # ================================================================
    def suggest_strategy(self, input_data: dict, prediction: dict) -> str:
        logger.info("전략 제안 생성 시작")

        if not self._router.is_llm_available:
            return self._fallback_strategy(input_data, prediction)

        data_stats = self._get_data_stats(input_data)
        prompt = build_strategy_prompt(input_data, prediction, data_context=data_stats)
        return self._router.generate(prompt, system=SYSTEM_CONSULTANT)

    # ================================================================
    # 3. Q&A 대화
    # ================================================================
    def ask(
            self,
            question: str,
            input_data: dict,
            prediction: dict,
            chat_history: list[dict] = None,
    ) -> str:
        logger.info("Q&A 질문: %s", question[:50])

        if not self._router.is_llm_available:
            return "LLM이 연결되지 않아 Q&A 기능을 사용할 수 없습니다. ollama serve를 실행해주세요."

        data_stats = self._get_data_stats(input_data, query=question)
        prompt = build_qa_prompt(question, input_data, prediction, chat_history, data_context=data_stats)
        return self._router.generate(prompt, system=SYSTEM_CONSULTANT, temperature=0.5)

    # ================================================================
    # 4. 경쟁업체 분석
    # ================================================================
    def analyze_competitors(self, input_data: dict, prediction: dict) -> str:
        logger.info("경쟁 분석 시작")

        if not self._router.is_llm_available:
            return self._fallback_competitor(input_data, prediction)

        data_stats = self._get_data_stats(input_data)
        prompt = build_competitor_prompt(input_data, prediction, data_context=data_stats)
        return self._router.generate(prompt, system=SYSTEM_CONSULTANT)

    # ================================================================
    # 규칙 기반 폴백 (LLM 없이 동작)
    # ================================================================
    def _fallback_report(self, input_data: dict, prediction: dict) -> str:
        s = prediction.get("survival", {})
        f = prediction.get("financials", {})
        r = prediction.get("risk", {})

        survival_1yr = s.get("one_year", 0)
        risk_level = r.get("level", "MEDIUM")

        if survival_1yr >= 0.7:
            survival_text = "양호합니다. 평균 이상의 생존 가능성을 보입니다."
        elif survival_1yr >= 0.5:
            survival_text = "보통 수준입니다. 철저한 준비가 필요합니다."
        else:
            survival_text = "주의가 필요합니다. 리스크 요인을 반드시 점검하세요."

        factors = r.get("factors", [])
        factors_text = "\n".join(f"  - {fac}" for fac in factors) if factors else "  - 특이사항 없음"

        # 데이터 통계 추가
        data_stats = self._get_data_stats(input_data)
        data_section = f"\n{data_stats}\n" if data_stats else ""

        return f"""📊 창업 컨설팅 리포트 (규칙 기반)
{'=' * 50}

■ 종합 평가
  1년 생존확률 {survival_1yr:.1%} — {survival_text}

■ 재무 전망
  예상 월매출: {f.get('monthly_revenue', 0):,}원
  예상 월순이익: {f.get('monthly_profit', 0):,}원
  손익분기 도달: {f.get('break_even_months', 0)}개월

■ 리스크 등급: {risk_level} (점수: {r.get('score', 0):.2f})
  주요 위험 요인:
{factors_text}
{data_section}
■ 권장 사항
  {chr(10).join(f'  - {rec}' for rec in prediction.get('recommendations', ['추가 분석 필요']))}

※ 더 상세한 분석을 원하시면 ollama serve를 실행하여
  LLM 기반 분석을 이용해주세요.
"""

    def _fallback_strategy(self, input_data: dict, prediction: dict) -> str:
        category = input_data.get("business_category", "일반")
        return f"[{category}] 업종 전략: LLM 연결 후 상세 전략을 확인할 수 있습니다."

    def _fallback_competitor(self, input_data: dict, prediction: dict) -> str:
        count = input_data.get("nearby_competitor_count", 0)
        return f"주변 경쟁업체 {count}개: LLM 연결 후 상세 분석을 확인할 수 있습니다."