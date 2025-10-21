# 💬 YoungSeek — 해외 영수증 기반 RAG 챗봇 (스마트 여행 가계부)

해외에서 받은 영수증을 업로드하면  
→ **OCR(문자 인식)** 으로 항목·금액을 추출하고  
→ **의미 기반 검색(RAG)** 으로 질문에 답하며  
→ **지출 요약**과 **대화형 분석**을 제공하는 웹 서비스입니다.

<p align="center">
  <img src="./assets/youngseek_architecture.png" width="720" alt="YoungSeek Architecture">
</p>

---

## ✨ 주요 기능

- **영수증 업로드 & OCR**: 이미지/PDF에서 항목·금액·통화 등 추출 (Azure Document Intelligence)
- **의미 기반 검색(RAG)**: Embedding + Azure Cognitive Search로 문장형 질의도 정확히 탐색
- **GPT 질의응답**: “이번 달 교통비 얼마나 썼어?” 같은 질문에 **한국어**로 응답
- **지출 요약 & 시각화**: 통화·카테고리별 요약, 파이/바 차트 대시보드
- **백업 & 인덱싱 자동화**: Blob Storage 저장 + Search 인덱싱 자동 수행
- **웹 UI**: Flask 기반 화면 (Gradio 프로토타입 포함)

---

## 🏗️ 아키텍처 개요

1. **사용자 업로드** (브라우저/Flask)
2. **OCR 분석** (Azure Document Intelligence)
3. **Embedding 생성 & 인덱싱** (Azure Cognitive Search)
4. **RAG 질의응답** (Azure OpenAI)
5. **결과 시각화** (Flask + Chart.js)
6. **영수증/CSV 백업** (Azure Blob Storage)

[Upload] → [OCR] → [Embedding/Indexing] → [RAG Q&A] → [UI/Charts] ↘
↑ ↙
[Blob Storage Backup / CSV]

## 프로젝트 구조 
youngseek/
├─ app.py                   # Flask UI (업로드/요약/질의응답)
├─ templates/
│   └─ index.html           # 메인 페이지
├─ static/
│   └─ yongseek.png         # 로고/이미지
├─ rag/
│   ├─ indexer.py           # Embedding & 인덱싱
│   ├─ retriever.py         # 검색/리트리벌
│   └─ prompt.py            # 프롬프트 템플릿
├─ api/                     # (선택) Django REST 백엔드
│   ├─ manage.py
│   └─ ...                  # DRF 엔드포인트
├─ assets/
│   ├─ youngseek_architecture.png
│   └─ youngseek_rag_flow.png
├─ requirements.txt
└─ .env (local)

