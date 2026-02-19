"""
📁 scripts/run_build_rag.py
==============================
RAG 벡터DB 구축 스크립트.

stores_raw.csv → Ollama 임베딩 → ChromaDB 저장

실행:
  python scripts/run_build_rag.py                 # 전체 구축
  python scripts/run_build_rag.py --max 1000      # 테스트 (1000건만)
  python scripts/run_build_rag.py --query "강남구 카페"  # 검색 테스트

필요:
  pip install chromadb
  ollama pull nomic-embed-text
"""

import sys
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from dotenv import load_dotenv
load_dotenv()

from src.utils.logger import setup_logging, get_logger
from src.utils.timer import timer
from src.llm.rag_store import RAGStore

logger = get_logger(__name__)


def main():
    parser = argparse.ArgumentParser(description="RAG 벡터DB 구축")
    parser.add_argument("--max", type=int, default=None, help="최대 문서 수 (테스트용)")
    parser.add_argument("--batch", type=int, default=100, help="배치 크기")
    parser.add_argument("--query", type=str, default=None, help="검색 테스트 질문")
    args = parser.parse_args()

    setup_logging()
    store = RAGStore()  # DB에서 자동으로 데이터 로드

    if args.query:
        # 검색 테스트
        logger.info("🔍 검색 테스트: '%s'", args.query)
        logger.info("저장된 문서: %d건", store.doc_count)

        results = store.search(args.query, top_k=5)
        if results:
            for i, r in enumerate(results, 1):
                print(f"\n[{i}] (유사도: {1 - r['distance']:.3f})")
                print(f"  {r['text']}")
        else:
            print("검색 결과 없음. 먼저 build를 실행하세요.")

        # 프롬프트용 컨텍스트
        print("\n" + "=" * 50)
        print("📋 프롬프트용 컨텍스트:")
        print(store.get_rag_context(args.query))

    else:
        # 벡터DB 구축
        logger.info("━━━ RAG 벡터DB 구축 시작 ━━━")
        count = store.build(batch_size=args.batch, max_docs=args.max)
        logger.info("━━━ 완료: %d건 저장 ━━━", count)
        logger.info("")
        logger.info("검색 테스트:")
        logger.info("  python scripts/run_build_rag.py --query '강남구 카페'")


if __name__ == "__main__":
    main()