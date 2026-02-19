"""
📁 src/serving/app.py
======================
FastAPI 애플리케이션.

엔드포인트:
  GET  /health              → 서버 상태
  POST /api/v1/predict      → ML 예측 (숫자)
  POST /api/v1/consult      → ML 예측 + LLM 컨설팅 리포트
  POST /api/v1/strategy     → 맞춤형 전략 제안
  POST /api/v1/ask          → Q&A 대화
  POST /api/v1/competitors  → 경쟁업체 분석

실행: uvicorn src.serving.app:app --reload
문서: http://localhost:8000/docs
"""

from contextlib import asynccontextmanager
from fastapi import FastAPI, Depends, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Optional

from config.settings import get_settings
from src.utils.logger import setup_logging, get_logger
from src.serving.schemas import PredictionRequest, PredictionResponse
from src.serving.predictor import Predictor
from src.serving.dependencies import get_predictor, get_consultant

logger = get_logger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    setup_logging()
    logger.info("🚀 서버 시작: %s", get_settings().APP_VERSION)
    yield
    logger.info("👋 서버 종료")


app = FastAPI(
    title="창업 컨설턴트 AI API",
    description="ML 예측 + LLM 기반 창업 컨설팅 서비스",
    version=get_settings().APP_VERSION,
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── 추가 스키마 ──

class QARequest(BaseModel):
    """Q&A 요청"""
    question: str = Field(..., description="질문")
    prediction_input: PredictionRequest
    chat_history: Optional[list[dict]] = Field(default=None, description="이전 대화 기록")


class LLMResponse(BaseModel):
    """LLM 응답"""
    success: bool = True
    llm_provider: str = ""
    prediction: Optional[dict] = None
    analysis: str = ""
    error: Optional[str] = None


# ── 엔드포인트 ──

@app.get("/health")
async def health():
    from src.serving.dependencies import get_consultant
    consultant = get_consultant()
    return {
        "status": "ok",
        "version": get_settings().APP_VERSION,
        "llm": consultant.active_llm,
    }


@app.post("/api/v1/predict", response_model=PredictionResponse)
async def predict(
        req: PredictionRequest,
        predictor: Predictor = Depends(get_predictor),
):
    """ML 예측만 (숫자 결과)"""
    try:
        result = predictor.predict(req.to_dict())
        return PredictionResponse(success=True, data=result)
    except Exception as e:
        logger.error("예측 실패: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))


@app.post("/api/v1/consult", response_model=LLMResponse)
async def consult(req: PredictionRequest):
    """ML 예측 + LLM 종합 컨설팅 리포트"""
    try:
        predictor = get_predictor()
        consultant = get_consultant()

        input_data = req.to_dict()
        prediction = predictor.predict(input_data)

        # 원본 입력에 API 스키마 필드도 포함
        full_input = {**input_data, **req.model_dump()}

        report = consultant.generate_report(full_input, prediction)

        return LLMResponse(
            success=True,
            llm_provider=consultant.active_llm,
            prediction=prediction,
            analysis=report,
        )
    except Exception as e:
        logger.error("컨설팅 실패: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))


@app.post("/api/v1/strategy", response_model=LLMResponse)
async def strategy(req: PredictionRequest):
    """맞춤형 전략 제안"""
    try:
        predictor = get_predictor()
        consultant = get_consultant()

        input_data = req.to_dict()
        prediction = predictor.predict(input_data)
        full_input = {**input_data, **req.model_dump()}

        result = consultant.suggest_strategy(full_input, prediction)

        return LLMResponse(
            success=True,
            llm_provider=consultant.active_llm,
            prediction=prediction,
            analysis=result,
        )
    except Exception as e:
        logger.error("전략 제안 실패: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))


@app.post("/api/v1/ask", response_model=LLMResponse)
async def ask(req: QARequest):
    """Q&A 대화"""
    try:
        predictor = get_predictor()
        consultant = get_consultant()

        input_data = req.prediction_input.to_dict()
        prediction = predictor.predict(input_data)
        full_input = {**input_data, **req.prediction_input.model_dump()}

        answer = consultant.ask(
            question=req.question,
            input_data=full_input,
            prediction=prediction,
            chat_history=req.chat_history,
        )

        return LLMResponse(
            success=True,
            llm_provider=consultant.active_llm,
            prediction=prediction,
            analysis=answer,
        )
    except Exception as e:
        logger.error("Q&A 실패: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))


@app.post("/api/v1/competitors", response_model=LLMResponse)
async def competitors(req: PredictionRequest):
    """경쟁업체 분석"""
    try:
        predictor = get_predictor()
        consultant = get_consultant()

        input_data = req.to_dict()
        prediction = predictor.predict(input_data)
        full_input = {**input_data, **req.model_dump()}

        result = consultant.analyze_competitors(full_input, prediction)

        return LLMResponse(
            success=True,
            llm_provider=consultant.active_llm,
            prediction=prediction,
            analysis=result,
        )
    except Exception as e:
        logger.error("경쟁 분석 실패: %s", e, exc_info=True)
        raise HTTPException(500, detail=str(e))