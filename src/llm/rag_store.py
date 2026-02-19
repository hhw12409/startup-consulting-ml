"""
📁 src/llm/rag_store.py
=========================
RAG 벡터스토어 — ChromaDB + Ollama 임베딩.

[역할] 수집된 상가 데이터를 벡터DB에 저장하고,
       사용자 질문에 관련된 상가 데이터를 검색하여 프롬프트에 주입합니다.

[흐름]
  구축: stores_raw.csv → 텍스트 변환 → Ollama 임베딩 → ChromaDB 저장
  검색: 질문 → Ollama 임베딩 → ChromaDB 유사도 검색 → TOP-K 결과

[설치]
  pip install chromadb
  ollama pull nomic-embed-text   # 임베딩 전용 모델 (274MB)

사용법:
    store = RAGStore()
    store.build()                             # 1회: 벡터DB 구축
    results = store.search("강남구 카페 현황")  # 관련 상가 검색
"""

import os
import json
import requests
import pandas as pd
from pathlib import Path
from typing import Optional

from config.settings import get_settings
from src.utils.logger import get_logger

logger = get_logger(__name__)

# Ollama 임베딩 모델 (경량, 빠름)
EMBED_MODEL = "nomic-embed-text"
OLLAMA_URL = "http://localhost:11434"

# ChromaDB 저장 경로
CHROMA_DIR = "data/06_vector_db"
COLLECTION_NAME = "stores"


class RAGStore:
    """
    RAG 벡터스토어.

    사용법:
        store = RAGStore()

        # 1회: 벡터DB 구축 (stores_raw.csv → ChromaDB)
        store.build()

        # 검색: 질문에 관련된 상가 데이터 반환
        results = store.search("강남구에서 카페 창업하려면?", top_k=5)

        # 프롬프트용 텍스트 반환
        context = store.get_rag_context("강남구 카페 경쟁 현황")
    """

    def __init__(self, chroma_dir: str = None):
        self._chroma_dir = chroma_dir or CHROMA_DIR
        self._collection = None

    # ================================================================
    # 벡터DB 구축
    # ================================================================

    def build(self, batch_size: int = 100, max_docs: int = None) -> int:
        """
        stores_raw.csv → ChromaDB 벡터DB 구축.

        Args:
            batch_size: 임베딩 배치 크기 (Ollama 호출 단위)
            max_docs: 최대 문서 수 (테스트용, None=전체)

        Returns:
            저장된 문서 수
        """
        # 1. 데이터 로드
        df = self._load_data()
        if df.empty:
            return 0

        if max_docs:
            df = df.head(max_docs)
            logger.info("max_docs=%d 적용", max_docs)

        # 2. 텍스트 변환
        docs = self._to_documents(df)
        logger.info("문서 변환 완료: %d건", len(docs))

        # 3. ChromaDB 컬렉션 생성
        collection = self._get_or_create_collection(reset=True)

        # 4. 배치 임베딩 & 저장
        total_saved = 0
        total_batches = (len(docs) + batch_size - 1) // batch_size

        for i in range(0, len(docs), batch_size):
            batch_num = i // batch_size + 1
            batch = docs[i:i + batch_size]

            texts = [d["text"] for d in batch]
            ids = [d["id"] for d in batch]
            metadatas = [d["metadata"] for d in batch]

            try:
                # Ollama 임베딩
                embeddings = self._embed_batch(texts)

                if embeddings:
                    collection.add(
                        documents=texts,
                        embeddings=embeddings,
                        ids=ids,
                        metadatas=metadatas,
                    )
                    total_saved += len(batch)

                if batch_num % 50 == 0 or batch_num == total_batches:
                    logger.info("  [%d/%d] 배치 저장 완료 (%d건 누적)", batch_num, total_batches, total_saved)

            except Exception as e:
                logger.error("배치 %d 실패: %s", batch_num, e)
                # 연속 실패 시 중단
                if total_saved == 0 and batch_num >= 3:
                    logger.error("⚠️ 연속 실패 → 구축 중단. Ollama 임베딩 모델 확인 필요")
                    logger.error("  설치: ollama pull nomic-embed-text")
                    break

        logger.info("━━━ RAG 벡터DB 구축 완료: %d건 저장 ━━━", total_saved)
        return total_saved

    # ================================================================
    # 검색
    # ================================================================

    def search(self, query: str, top_k: int = 5) -> list[dict]:
        """
        질문에 관련된 상가 데이터를 검색합니다.

        Args:
            query: 검색 질문
            top_k: 반환할 최대 결과 수

        Returns:
            [{"text": "...", "metadata": {...}, "distance": 0.23}, ...]
        """
        collection = self._get_or_create_collection()

        if collection.count() == 0:
            logger.warning("벡터DB가 비어있습니다. 먼저 build()를 실행하세요.")
            return []

        try:
            query_embedding = self._embed_text(query)
            if not query_embedding:
                return []

            results = collection.query(
                query_embeddings=[query_embedding],
                n_results=top_k,
            )

            items = []
            for i in range(len(results["ids"][0])):
                items.append({
                    "text": results["documents"][0][i],
                    "metadata": results["metadatas"][0][i] if results["metadatas"] else {},
                    "distance": results["distances"][0][i] if results["distances"] else 0,
                })

            logger.debug("RAG 검색 '%s' → %d건", query[:30], len(items))
            return items

        except Exception as e:
            logger.error("RAG 검색 실패: %s", e)
            return []

    def get_rag_context(self, query: str, top_k: int = 5) -> str:
        """
        검색 결과를 프롬프트용 텍스트로 반환합니다.

        Args:
            query: 검색 질문
            top_k: 반환할 최대 결과 수

        Returns:
            프롬프트에 삽입할 텍스트
        """
        results = self.search(query, top_k)

        if not results:
            return ""

        lines = ["## 🔍 유사 사례 데이터 (RAG 검색 결과)\n"]
        for i, r in enumerate(results, 1):
            lines.append(f"**사례 {i}** (유사도: {1 - r['distance']:.2f})")
            lines.append(r["text"])
            lines.append("")

        lines.append("위 유사 사례를 참고하여 분석해주세요.\n")

        return "\n".join(lines)

    @property
    def doc_count(self) -> int:
        """저장된 문서 수"""
        try:
            collection = self._get_or_create_collection()
            return collection.count()
        except Exception:
            return 0

    # ================================================================
    # 내부 메서드
    # ================================================================

    def _load_data(self) -> pd.DataFrame:
        """DB(stores 테이블)에서 상가 데이터 로드"""
        try:
            from src.database.repository import StoreRepository
            repo = StoreRepository()
            df = repo.to_dataframe()
            if not df.empty:
                # DB 컬럼은 이미 문자열이 아닐 수 있으므로 str 변환
                df = df.astype(str)
                logger.info("DB에서 데이터 로드: %d건", len(df))
                return df
        except Exception as e:
            logger.warning("DB 로드 실패: %s", e)

        logger.error("데이터를 로드할 수 없습니다. 먼저 'make collect'로 수집하세요.")
        return pd.DataFrame()

    def _to_documents(self, df: pd.DataFrame) -> list[dict]:
        """DataFrame → 벡터DB 문서 리스트로 변환"""
        docs = []

        # 컬럼명 탐색 (DB 컬럼명 우선)
        name_col = self._find_col(df, ["store_name", "bizesNm"])
        cat_col = self._find_col(df, ["category_large", "business_category", "indsLclsCdNm"])
        sub_col = self._find_col(df, ["category_mid", "business_sub_category", "indsMclsCdNm"])
        detail_col = self._find_col(df, ["category_small", "indsSclsCdNm", "business_detail"])
        dist_col = self._find_col(df, ["adong_name", "district", "adongNm"])
        addr_col = self._find_col(df, ["road_address", "rdnmAdr", "lnoAdr"])
        sgg_col = self._find_col(df, ["sgg_name", "sggNm", "sgg_nm"])
        status_col = self._find_col(df, ["biz_status_cd", "b_stt_cd", "b_stt"])

        for idx, row in df.iterrows():
            name = row.get(name_col, "상호 미상") if name_col else "상호 미상"
            cat = row.get(cat_col, "") if cat_col else ""
            sub = row.get(sub_col, "") if sub_col else ""
            detail = row.get(detail_col, "") if detail_col else ""
            dist = row.get(dist_col, "") if dist_col else ""
            addr = row.get(addr_col, "") if addr_col else ""
            sgg = row.get(sgg_col, "") if sgg_col else ""

            # 사업자 상태
            status = ""
            if status_col:
                code = row.get(status_col, "")
                status_map = {"01": "영업중", "02": "휴업", "03": "폐업"}
                status = status_map.get(str(code), "")

            # 텍스트로 변환
            parts = [f"{name}"]
            if cat:
                parts.append(f"업종: {cat}")
            if sub:
                parts.append(f"세부: {sub}")
            if detail:
                parts.append(f"상세: {detail}")
            if sgg and dist:
                parts.append(f"위치: {sgg} {dist}")
            elif dist:
                parts.append(f"위치: {dist}")
            if addr and str(addr) != "nan":
                parts.append(f"주소: {addr}")
            if status:
                parts.append(f"상태: {status}")

            text = " | ".join(parts)

            # 메타데이터
            metadata = {}
            if cat:
                metadata["category"] = str(cat)
            if sub:
                metadata["sub_category"] = str(sub)
            if dist:
                metadata["district"] = str(dist)
            if sgg:
                metadata["sgg"] = str(sgg)
            if status:
                metadata["status"] = status

            docs.append({
                "id": f"store_{idx}",
                "text": text,
                "metadata": metadata,
            })

        return docs

    def _get_or_create_collection(self, reset: bool = False):
        """ChromaDB 컬렉션 가져오기/생성"""
        import chromadb

        client = chromadb.PersistentClient(path=self._chroma_dir)

        if reset:
            # 기존 컬렉션 삭제 후 재생성
            try:
                client.delete_collection(COLLECTION_NAME)
            except Exception:
                pass

        collection = client.get_or_create_collection(
            name=COLLECTION_NAME,
            metadata={"hnsw:space": "cosine"},  # 코사인 유사도
        )

        return collection

    def _embed_text(self, text: str) -> list[float]:
        """단일 텍스트 임베딩 (Ollama)"""
        try:
            resp = requests.post(
                f"{OLLAMA_URL}/api/embeddings",
                json={"model": EMBED_MODEL, "prompt": text},
                timeout=30,
            )
            resp.raise_for_status()
            return resp.json().get("embedding", [])
        except Exception as e:
            logger.error("임베딩 실패: %s", e)
            return []

    def _embed_batch(self, texts: list[str]) -> list[list[float]]:
        """배치 임베딩 (Ollama는 개별 호출)"""
        embeddings = []
        for text in texts:
            emb = self._embed_text(text)
            if emb:
                embeddings.append(emb)
            else:
                # 실패 시 빈 벡터 대신 에러
                return []
        return embeddings

    def _find_col(self, df: pd.DataFrame, candidates: list[str]) -> str:
        """DataFrame에서 존재하는 첫 번째 컬럼명 반환"""
        for col in candidates:
            if col in df.columns:
                return col
        return ""