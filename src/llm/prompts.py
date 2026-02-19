"""
📁 src/llm/prompts.py
=======================
프롬프트 템플릿.

[패턴] Template Method — 프롬프트 골격을 정의하고, 데이터만 교체
[역할] ML 예측 결과 + 실제 데이터 통계 + 입력 정보를 LLM 프롬프트로 변환

[개선] 수집된 상가 데이터 통계가 프롬프트에 자동 주입됩니다.
  Before: LLM의 일반 지식만으로 답변
  After:  실제 데이터 통계 + ML 예측 + 일반 지식
"""

# ================================================================
# 시스템 프롬프트 (LLM의 역할 정의)
# ================================================================
SYSTEM_CONSULTANT = """당신은 대한민국 소상공인 창업 전문 컨설턴트 AI '황피티'입니다.

인사:
- 모든 응답의 첫 줄에 반드시 "안녕 난 황피티야 너의 질문에 대답을 해줄게" 라고 인사한 뒤 답변을 시작합니다.

역할:
- ML 모델의 예측 결과와 실제 상가 데이터 통계를 바탕으로 정확하고 실용적인 창업 컨설팅을 제공합니다.
- "실제 데이터 기반 분석" 섹션이 제공되면, 반드시 그 통계를 근거로 답변하세요.
- 숫자를 나열하지 말고, 의미를 해석하여 자연스러운 한국어로 설명합니다.
- 긍정적인 면과 주의할 점을 균형있게 전달합니다.
- 구체적이고 실행 가능한 조언을 제공합니다.

톤:
- 전문적이지만 친근하게
- 불필요한 전문용어 대신 쉬운 말로
- 단정적 표현 대신 확률과 가능성으로
- 데이터에 근거한 구체적 수치 언급
"""


# ================================================================
# 1. 컨설팅 리포트 프롬프트
# ================================================================
def build_report_prompt(input_data: dict, prediction: dict, data_context: str = "") -> str:
    """
    ML 예측 결과 → 자연어 컨설팅 리포트 프롬프트.

    Args:
        input_data: 사용자 입력 (나이, 업종, 투자금 등)
        prediction: ML 예측 결과 (survival, financials, risk 등)
        data_context: 실제 데이터 통계 (DataContext.get_context 결과)
    """
    survival = prediction.get("survival", {})
    financials = prediction.get("financials", {})
    risk = prediction.get("risk", {})

    context_section = f"\n{data_context}\n" if data_context else ""

    return f"""다음은 창업 예측 AI 모델의 분석 결과입니다. 이를 바탕으로 종합 컨설팅 리포트를 작성해주세요.

## 창업자 정보
- 나이: {input_data.get('founder_age', '미상')}세
- 성별: {input_data.get('founder_gender', '미상')}
- 학력: {input_data.get('founder_education', '미상')}
- 경력: {input_data.get('experience_years', 0)}년
- 업종 관련 경험: {'있음' if input_data.get('has_related_experience') else '없음'}

## 사업 정보
- 업종: {input_data.get('business_category', '미상')} / {input_data.get('business_sub_category', '')}
- 지역: {input_data.get('district', '미상')}
- 초기 투자금: {input_data.get('initial_investment', 0):,}원
- 월 임대료: {input_data.get('monthly_rent', 0):,}원
- 매장 크기: {input_data.get('store_size_sqm', 0)}㎡
- 직원 수: {input_data.get('employee_count', 0)}명
- 프랜차이즈: {'예' if input_data.get('is_franchise') else '아니오'}

## ML 모델 예측 결과
- 1년 생존확률: {survival.get('one_year', 0):.1%}
- 3년 생존확률: {survival.get('three_year', 0):.1%}
- 예상 월매출: {financials.get('monthly_revenue', 0):,}원
- 예상 월순이익: {financials.get('monthly_profit', 0):,}원
- 손익분기 도달: {financials.get('break_even_months', 0)}개월
- 리스크 점수: {risk.get('score', 0):.2f} (등급: {risk.get('level', '미상')})
- 리스크 요인: {', '.join(risk.get('factors', []))}
{context_section}
## 요청
위 정보를 바탕으로 다음 구조의 컨설팅 리포트를 작성해주세요:

1. **종합 평가** (2~3문장으로 핵심 요약)
2. **생존율 분석** (실제 데이터 통계와 비교하여 의미 해석)
3. **재무 분석** (투자 대비 수익성, 손익분기점 의미)
4. **경쟁 환경** (해당 지역+업종의 실제 경쟁 현황 데이터 기반)
5. **리스크 분석** (주요 위험 요인별 상세 설명)
6. **실행 전략** (3~5가지 구체적이고 실행 가능한 조언)
"""


# ================================================================
# 2. 맞춤형 전략 제안 프롬프트
# ================================================================
def build_strategy_prompt(input_data: dict, prediction: dict, data_context: str = "") -> str:
    """업종/지역 기반 맞춤형 전략 제안 프롬프트."""
    category = input_data.get("business_category", "")
    district = input_data.get("district", "")
    risk_level = prediction.get("risk", {}).get("level", "MEDIUM")

    context_section = f"\n{data_context}\n" if data_context else ""

    return f"""당신은 '{category}' 업종, '{district}' 지역 전문 창업 컨설턴트입니다.

리스크 등급: {risk_level}
투자금: {input_data.get('initial_investment', 0):,}원
경험: {input_data.get('experience_years', 0)}년
{context_section}
이 창업자를 위한 맞춤형 전략을 다음 항목별로 제안해주세요.
실제 데이터 통계가 제공된 경우, 반드시 그 수치를 근거로 전략을 수립하세요.

1. **입지 전략** — 이 지역에서 이 업종이 성공하려면? (경쟁업체 수 참고)
2. **차별화 전략** — 주변 경쟁업체와 어떻게 차별화할 것인가?
3. **마케팅 전략** — 초기 고객 확보 방법 (예산 고려)
4. **비용 절감 전략** — 초기 투자금을 효율적으로 사용하는 방법
5. **성장 로드맵** — 1개월/3개월/6개월/1년 단위 목표

각 전략은 구체적이고 실행 가능해야 합니다. 추상적인 조언은 피하세요.
"""


# ================================================================
# 3. Q&A 대화 프롬프트
# ================================================================
def build_qa_prompt(
        question: str,
        input_data: dict,
        prediction: dict,
        chat_history: list[dict] = None,
        data_context: str = "",
) -> str:
    """사용자 질문에 대한 Q&A 프롬프트."""
    history_text = ""
    if chat_history:
        for msg in chat_history[-5:]:
            role = "사용자" if msg["role"] == "user" else "컨설턴트"
            history_text += f"{role}: {msg['content']}\n"

    survival = prediction.get("survival", {})
    risk = prediction.get("risk", {})

    context_section = f"\n{data_context}\n" if data_context else ""

    return f"""다음은 창업자의 정보와 AI 예측 결과입니다.

업종: {input_data.get('business_category', '미상')} ({input_data.get('business_sub_category', '')})
지역: {input_data.get('district', '미상')}
투자금: {input_data.get('initial_investment', 0):,}원
1년 생존확률: {survival.get('one_year', 0):.1%}
리스크: {risk.get('level', '미상')} ({risk.get('score', 0):.2f})
{context_section}
{f"이전 대화:{chr(10)}{history_text}" if history_text else ""}

사용자 질문: {question}

실제 데이터 통계가 제공된 경우, 반드시 그 수치를 근거로 답변하세요.
이 창업자의 상황에 맞게 정확하고 실용적으로 답변해주세요.
모르는 것은 솔직히 모른다고 하되, 관련 정보를 찾아볼 것을 제안해주세요.
"""


# ================================================================
# 4. 경쟁업체 분석 프롬프트
# ================================================================
def build_competitor_prompt(input_data: dict, prediction: dict, data_context: str = "") -> str:
    """경쟁업체 분석 리포트 프롬프트."""
    context_section = f"\n{data_context}\n" if data_context else ""

    return f"""다음 창업 정보를 바탕으로 경쟁 환경 분석 리포트를 작성해주세요.

업종: {input_data.get('business_category', '미상')} / {input_data.get('business_sub_category', '')}
지역: {input_data.get('district', '미상')}
주변 경쟁업체 수: {input_data.get('nearby_competitor_count', 0)}개
리스크 등급: {prediction.get('risk', {}).get('level', '미상')}
{context_section}
실제 데이터 통계가 제공된 경우, 반드시 그 수치를 근거로 분석하세요.

다음 구조로 분석해주세요:

1. **경쟁 강도 평가** — 이 지역 이 업종의 실제 경쟁 현황 (데이터 기반)
2. **경쟁자 유형 분석** — 어떤 유형의 경쟁자가 있을지 (프랜차이즈, 개인, 온라인)
3. **시장 포화도** — 이 업종이 이 지역에서 포화 상태인지 (상가 수 기반)
4. **기회 요인** — 경쟁에도 불구하고 진입 가능한 틈새
5. **경쟁 우위 확보 방안** — 구체적인 차별화 포인트 3가지
"""