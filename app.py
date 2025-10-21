import os
import json
import uuid
import time
from datetime import datetime
from flask import Flask, request, jsonify, render_template, session
import requests
from azure.storage.blob import BlobServiceClient
from openai import AzureOpenAI
import pandas as pd
from dotenv import load_dotenv
import io
import csv
import traceback 


# --- 1. Flask 앱 및 기본 설정 ---
app = Flask(__name__)
app.secret_key = 'my-secret-key-team04'
# 업로드된 이미지를 임시 저장할 폴더
if not os.path.exists("uploads"):
    os.makedirs("uploads")

# --- 2. Azure 서비스 설정 ---
load_dotenv()
GPT_DEPLOYMENT = "gpt-4.1"
AZURE_ENDPOINT = os.getenv("AZURE_ENDPOINT")
AZURE_KEY  = os.getenv("AZURE_KEY")
MODEL_ID  = os.getenv("MODEL_ID")
API_VERSION  = os.getenv("API_VERSION")
OPENAI_URL  = os.getenv("OPENAI_URL")
OPENAI_KEY  = os.getenv("OPENAI_KEY")
OPENAI_ENDPOINT = os.getenv("OPENAI_ENDPOINT")
BLOB_CONNECTION_STRING  = os.getenv("BLOB_CONNECTION_STRING")
BLOB_CONTAINER_NAME  = os.getenv("BLOB_CONTAINER_NAME")
AISEARCH_KEY  = os.getenv("AISEARCH_KEY")
SEARCH_VERSION  = os.getenv("SEARCH_VERSION")
INDEX_NAME = os.getenv("INDEX_NAME")
SEARCH_SERVICE  = os.getenv("SEARCH_SERVICE")
CATEGORIES = ['교통비', '숙박비', '식비', '입장료및 체험 활동비', '쇼핑 및 기념품비', '기타']
EXCHANGE_RATES = { "JPN": 9, "EUR": 1500, "USD": 1300, "KRW": 1 }

# --- 3. Azure 클라이언트 초기화 ---
client = AzureOpenAI(api_key=OPENAI_KEY, azure_endpoint=OPENAI_ENDPOINT, 
                     api_version="2023-12-01-preview")

# --- 4. 백엔드 헬퍼 함수 ---
def analyze_receipt_rest(image_path):
    # (코드는 이전과 동일)
    try:
        post_url = f"{AZURE_ENDPOINT}/formrecognizer/documentModels/{MODEL_ID}:analyze?api-version={API_VERSION}"
        headers = {"Ocp-Apim-Subscription-Key": AZURE_KEY, "Content-Type": "image/jpeg"}
        with open(image_path, "rb") as f: response = requests.post(post_url, headers=headers, data=f)
        if response.status_code != 202: raise Exception(f"요청 실패: {response.status_code} {response.text}")
        operation_url = response.headers["Operation-Location"]
        for _ in range(10):
            result = requests.get(operation_url, headers={"Ocp-Apim-Subscription-Key": AZURE_KEY})
            if result.status_code == 200:
                result_json = result.json()
                if result_json.get("status") == "succeeded": return result_json.get("analyzeResult", {}).get("content", "")
                elif result_json.get("status") == "failed": raise Exception("Azure DI 분석 실패")
            time.sleep(1)
        raise Exception("Azure DI 분석 시간 초과")
    except Exception as e: raise Exception(f"Azure 분석 중 오류: {str(e)}")

def call_gpt_for_csv(prompt):
    # (코드는 이전과 동일)
    try:
        headers = {"Content-Type": "application/json", "api-key": OPENAI_KEY}
        system_prompt = '''
          You are a multilingual receipt parser and CSV formatter for a financial RAG system.

                        Your task is to extract structured information from receipts and return a clean CSV table with the following:

                        🔹 Columns (in order):
                        1. 아이디 (id): 32-character UUID per row (must be unique)
                        2. 가게명 (store): do not translate store name  
                        3. 날짜 (date): format as YYYY-MM-DD if possible
                        4. 품목명 (items): translate to Korean unless it's a brand name
                        5. 가격 (price): numeric only (no currency symbol)
                        6. 통화 (currency): e.g., KRW, EUR, JPY, USD
                        7. 수량 (quantity): default to 1 if missing
                        8. 카테고리 (category): one of [식비, 교통비, 숙박비, 쇼핑&기념품비, 입장료및 체험 활동비, 기타]
                        9. 국가 (country): infer from currency
                        10. 내용 (content): write a full Korean sentence like:  
                                **"[날짜] [국가] [가게명]에서 [품목명]을(를) [가격][통화 단위]에 구매함"**  
                                → e.g. `"2024년 7월 1일 독일 ZARA에서 셔츠를 49,000원에 구매함"`
                        

                        🔹 Formatting Rules:
                        1. Output must be CSV only, with no explanation or markdown.
                        2. Start with header:id, store, date, items, price, currency, quantity, category, country, content
                        3. If any field contains commas, wrap in double quotes

        '''
        payload = {"messages": [{"role": "system", "content": system_prompt}, {"role": "user", "content": prompt}], "temperature": 0.1, "max_tokens": 1500}
        response = requests.post(f"{OPENAI_ENDPOINT}openai/deployments/{GPT_DEPLOYMENT}/chat/completions?api-version=2025-01-01-preview", headers=headers, json=payload)
        response.raise_for_status()
        return response.json()["choices"][0]["message"]["content"]
    except Exception as e: raise Exception(f"GPT CSV 변환 오류: {str(e)}")

def generate_embedding(text):
    # (코드는 이전과 동일)
    response = client.embeddings.create(model="text-embedding-ada-002", input=text)
    return response.data[0].embedding

def upload_csv_to_blob(filename, csv_string):
    # (코드는 이전과 동일)
    blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
    blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob=filename)
    blob_client.upload_blob(csv_string.encode('utf-8-sig'), overwrite=True)

def process_image_and_get_data(image_path):
    """
    [최종 통합 파이프라인]
    이미지 한 장을 받아 '두 갈래 길'로 데이터를 처리하고,
    UI용 JSON과 RAG용 파일명을 반환합니다.
    """
    # 1. 공통 처리: OCR -> CSV 생성 -> Blob 백업
    raw_text = analyze_receipt_rest(image_path)
    if not raw_text.strip(): raise Exception("이미지에서 텍스트를 추출하지 못했습니다.")
    
    csv_text = call_gpt_for_csv(raw_text)
    if not csv_text or not "id,store" in csv_text: raise Exception("GPT가 유효한 CSV를 생성하지 못했습니다.")
    
    now_str = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"receipt_{now_str}.csv"
    upload_csv_to_blob(csv_filename, csv_text)

    # 이제 '두 갈래 길'로 나뉩니다.
    csv_file = io.StringIO(csv_text)
    reader = csv.DictReader(csv_file)
    rows = list(reader) # reader를 리스트로 변환하여 여러 번 사용

    # 2. 첫 번째 길: RAG를 위한 인덱싱
    documents_to_upload = []
    for row in rows:
        content = row.get("content", "").strip()
        if not content: continue
        vector = generate_embedding(content)
        documents_to_upload.append({
            "@search.action": "upload", "id": row.get("id", str(uuid.uuid4().hex)), "store": row.get("store"),
            "date": row.get("date"), "items": row.get("items"),
            "price": float(row.get("price", "0").replace(",", ".")), "currency": row.get("currency"),
            "quantity": int(row.get("quantity", 1)), "category": row.get("category"),
            "country": row.get("country"), "content": content, "content_vector": vector
        })

    if documents_to_upload:
        url = f"https://{SEARCH_SERVICE}.search.windows.net/indexes/{INDEX_NAME}/docs/index?api-version={SEARCH_VERSION}"
        headers = {"Content-Type": "application/json", "api-key": AISEARCH_KEY}
        response = requests.post(url, headers=headers, json={"value": documents_to_upload})
        if response.status_code not in [200, 201]: raise Exception(f"Search 업로드 실패: {response.status_code} {response.text}")

    # 3. 두 번째 길: UI 시각화를 위한 JSON 생성
    total_price = sum(float(row.get("price", "0").replace(",", ".")) for row in rows)
    ui_json_output = {
        "merchantName": rows[0].get('store') if rows else "N/A",
        "transactionDate": rows[0].get('date') if rows else "N/A",
        "total": f"{total_price:.2f}",
        "currency": rows[0].get('currency') if rows else "N/A",
        "items": [{"name": row.get("items"), "price": row.get("price")} for row in rows]
    }
    
    return ui_json_output, csv_filename

# (카테고리 분류 등 나머지 헬퍼 함수는 이전과 동일)
def get_category_from_gpt(item_name):
    """
    GPT의 답변이 약간 다르더라도, 우리가 정의한 카테고리 중 하나를 찾아 반환합니다.
    훨씬 더 안정적으로 작동합니다.
    """
    if not item_name or item_name == 'N/A':
        return '기타'
    
    system_prompt = f"다음 품목을 다음 카테고리 중 하나로 분류해줘: {', '.join(CATEGORIES)}. 답변은 오직 카테고리 이름 하나여야 합니다. 환율은 {EXCHANGE_RATES} 사용해서 나타내줘."
    
    try:
        response = client.chat.completions.create(
            model=GPT_DEPLOYMENT,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": item_name}
            ],
            temperature=0,
            max_tokens=20
        )
        gpt_answer = response.choices[0].message.content.strip()

        # GPT 답변에 우리의 카테고리 중 하나라도 포함되어 있는지 확인
        for cat in CATEGORIES:
            if cat in gpt_answer:
                print(f"    - GPT 분류: '{item_name}' -> '{cat}' (원본: '{gpt_answer}')")
                return cat
        
        # 만약 포함된 것이 없다면 '기타'로 처리
        print(f"    - GPT 분류 실패: '{item_name}' -> '기타' (원본: '{gpt_answer}')")
        return '기타'
        
    except Exception:
        return '기타'


# --- 5. Flask API 라우트 (최종 정리 버전) ---

@app.route('/')
def index():
    """메인 HTML 페이지만 렌더링. 모든 로직은 JS와 API로 처리."""
    session.clear()
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_receipt_api():
    if 'receiptImage' not in request.files:
        return jsonify({"error": "이미지 파일이 없습니다."}), 400
    file = request.files['receiptImage']
    
    filepath = os.path.join("uploads", str(uuid.uuid4()) + "_" + file.filename)
    file.save(filepath)

    try:
        # 통합된 파이프라인 함수 호출
        structured_data, filename = process_image_and_get_data(filepath)
        print("✅ 파이프라인 처리 성공!") # 성공 로그 추가
        return jsonify({
            "receiptData": structured_data,
            "filename": filename
        })
        
    # [❗❗❗ 여기가 가장 중요합니다 ❗❗❗]
    except Exception as e:
        # 서버 터미널에 아주 크고 명확하게 오류를 출력합니다.
        print("\n🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥")
        print("🔥🔥🔥 /upload 경로에서 심각한 오류 발생! 🔥🔥🔥")
        print(f"🔥🔥🔥 오류 메시지: {e}")
        traceback.print_exc()  # 오류의 상세 내용을 터미널에 모두 출력
        print("🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥🔥\n")
        
        # 클라이언트에게는 여전히 에러 JSON을 보냅니다.
        return jsonify({"error": f"서버 내부 오류: {str(e)}"}), 500
    finally:
        # 처리 후 임시 파일 삭제
        if os.path.exists(filepath):
            os.remove(filepath)

@app.route('/chat', methods=['POST'])
def chat_api():
    """/chat API: 채팅 메시지를 받아 RAG 응답(JSON)을 반환합니다."""
    data = request.get_json()
    user_prompt = data.get("prompt")
    if not user_prompt: return jsonify({"error": "메시지가 없습니다."}), 400

    chat_history = session.get('history', [])
    messages = [{"role": "system", "content": 
                 '''
                 You are a friendly financial assistant named '영식이'. 
                 - Include specific details from receipts such as store name, date, items, price, currency, and category when relevant.
                - If the user's question is vague, politely ask for clarification.
                - Maintain a conversational and helpful tone at all times.
                - When the user asks for summaries or comparisons, provide calculated insights clearly.
                - If a Korean store name is given by the user, try to match it to its English equivalent as stored in the receipt data.
                - Always respond in Korean unless the user specifically asks for another language.
                - Use emojis to make the conversation more friendly and engaging
                - Except the [doc](citations) after the chatbot answer.
                '''
                 }]
    messages.extend(chat_history)
    messages.append({"role": "user", "content": user_prompt})

    try:
        response = client.chat.completions.create(
            model=GPT_DEPLOYMENT, messages=messages,
            extra_body={
                "data_sources": [{
                    "type": "azure_search", "parameters": {
                        "endpoint": f"https://{SEARCH_SERVICE}.search.windows.net", "index_name": INDEX_NAME,
                        "semantic_configuration": "final-semantic",
                        "query_type": "vector_semantic_hybrid",
                        "embedding_dependency": {"type": "deployment_name", "deployment_name": "text-embedding-ada-002"},
                        "fields_mapping": {
                                "content_fields": ["content", "items", "store",'price', 'category'],
                                "vector_fields": ["content_vector"],
                                "title_field": "items",
                                "url_field": None,
                                "filepath_field": None
                            },
                            "strictness": 2,
                            "top_n_documents": 50,
                            "authentication": {
                                "type": "api_key",
                                "key": AISEARCH_KEY
                            }
                    }
                }]
            }
        )
        bot_response = response.choices[0].message.content
        chat_history.append({"role": "user", "content": user_prompt})
        chat_history.append({"role": "assistant", "content": bot_response})
        session['history'] = chat_history
        return jsonify({"response": bot_response})
    except Exception as e:
        return jsonify({"error": f"챗봇 오류: {str(e)}"}), 500

# (리포트, 삭제 등 나머지 API는 이전과 동일하게 유지)
@app.route('/delete-receipt', methods=['POST'])
def delete_receipt():
    filename = request.json.get('filename')
    if not filename: return jsonify({"error": "삭제할 파일명이 없습니다."}), 400
    try:
        blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
        blob_client = blob_service_client.get_blob_client(container=BLOB_CONTAINER_NAME, blob=filename)
        blob_client.delete_blob()
        # [수정] 성공 응답을 return 해야 합니다.
        return jsonify({"message": f"{filename} 삭제 완료"}), 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# ... (파일의 다른 부분은 그대로 유지) ...

@app.route('/generate-web-report', methods=['POST'])
def generate_web_report():
    session_data = request.json
    if not session_data: return jsonify({"error": "분석할 데이터가 없습니다."}), 400

    def safe_float(value):
        if value is None: return 0.0
        try: return float(str(value).replace(',', '.').strip())
        except (ValueError, TypeError): return 0.0

    try:
        all_processed_items = []
        for receipt_obj in session_data:
            receipt = receipt_obj.get('data', {})
            currency = receipt.get('currency', 'N/A')
            rate = EXCHANGE_RATES.get(currency, 1)
            items_list = receipt.get('items', [])
            
            items_processed_for_this_receipt = False
            if items_list and isinstance(items_list, list):
                for item in items_list:
                    item_price = safe_float(item.get('price'))
                    if item_price > 0:
                        all_processed_items.append({
                            "카테고리": get_category_from_gpt(item.get('name')),
                            "통화": currency, "가격": item_price, "원화가격": item_price * rate
                        })
                        items_processed_for_this_receipt = True

            if not items_processed_for_this_receipt:
                receipt_total = safe_float(receipt.get('total'))
                if receipt_total > 0:
                    all_processed_items.append({
                        "카테고리": "기타", "통화": currency,
                        "가격": receipt_total, "원화가격": receipt_total * rate
                    })

        if not all_processed_items: return jsonify({"error": "분석할 유효한 지출 항목이 없습니다."}), 400
        
        df = pd.DataFrame(all_processed_items)
        summary = df.groupby(['통화', '카테고리']).agg(가격=('가격', 'sum'), 원화가격=('원화가격', 'sum')).reset_index()
        results = {}
        for currency in summary['통화'].unique():
            currency_df = summary[summary['통화'] == currency].set_index('카테고리').reindex(CATEGORIES, fill_value=0)
            if currency_df['가격'].sum() > 0:
                chart_data_df = currency_df[currency_df['가격'] > 0]
                results[currency] = {
                    "total_orig": currency_df['가격'].sum(),
                    "total_krw": currency_df['원화가격'].sum(),
                    "table_data": currency_df.reset_index().to_dict('records'),
                    "chart_labels": chart_data_df.index.tolist(),
                    "chart_data": chart_data_df['가격'].tolist(),
                    # [수정] 날짜 정보 제거, 환율 정보만 남김
                    "exchange_rate": EXCHANGE_RATES.get(currency, 1)
                }
        return jsonify(results)
    
    except Exception as e:
        import traceback; traceback.print_exc()
        return jsonify({"error": f"리포트 생성 오류: {str(e)}"}), 500
    
# 가계부 불러오는 코드
@app.route('/get-ledger', methods=['GET'])
def get_ledger():
    """
    [새로운 기능] Blob Storage의 모든 CSV를 읽어와 나라별/카테고리별로
    집계된 전체 가계부 데이터를 반환합니다.
    """
    try:
        blob_service_client = BlobServiceClient.from_connection_string(BLOB_CONNECTION_STRING)
        container_client = blob_service_client.get_container_client(BLOB_CONTAINER_NAME)

        all_dfs = []
        blob_list = container_client.list_blobs()
        
        for blob in blob_list:
            blob_client = container_client.get_blob_client(blob.name)
            downloader = blob_client.download_blob(max_concurrency=1, encoding='utf-8-sig')
            blob_text = downloader.readall()
            
            # 비어있는 파일 건너뛰기
            if not blob_text.strip():
                continue

            string_io = io.StringIO(blob_text)
            df = pd.read_csv(string_io)
            all_dfs.append(df)

        if not all_dfs:
            return jsonify({}), 200 # 데이터가 없으면 빈 객체 반환

        # 모든 데이터프레임을 하나로 합침
        master_df = pd.concat(all_dfs, ignore_index=True)

        # 데이터 클리닝 및 계산
        master_df['price'] = pd.to_numeric(master_df['price'].astype(str).str.replace(',', '.'), errors='coerce').fillna(0)
        master_df['quantity'] = pd.to_numeric(master_df['quantity'], errors='coerce').fillna(1)
        master_df['total_price'] = master_df['price'] * master_df['quantity']
        
        # 환율 적용
        master_df['rate'] = master_df['currency'].map(EXCHANGE_RATES).fillna(1)
        master_df['krw_price'] = master_df['total_price'] * master_df['rate']
        
        # 나라별, 카테고리별로 원화(₩) 합계 계산
        summary = master_df.groupby(['country', 'category']).agg(
            total_orig = ('total_price', 'sum'),
            total_krw=('krw_price', 'sum')
        ).reset_index()

        # 프론트엔드에 보낼 데이터 구조로 재구성
        ledger_data = {}
        country_currency_map = master_df.groupby('country')['currency'].first().to_dict()


        for country in summary['country'].unique():
            country_df = summary[summary['country'] == country]

            data_list = country_df[['category', 'total_orig', 'total_krw']].to_dict('records')

            ledger_data[country] = {
            'total_orig': country_df['total_orig'].sum(),# 국가별 현지 통화 총합
            'total_krw': country_df['total_krw'].sum(),
            'currency': country_currency_map.get(country, ''),# 국가의 통화 코드 (예: 'EUR')
            'data': data_list # 카테고리별 데이터
        }
        
        return jsonify(ledger_data)

    except Exception as e:
        import traceback
        traceback.print_exc()
        return jsonify({"error": f"전체 가계부 데이터를 불러오는 중 오류 발생: {str(e)}"}), 500
    
# --- 6. 서버 실행 ---
if __name__ == '__main__':
    app.run(debug=True, port=5001)
