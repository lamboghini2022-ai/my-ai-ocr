import os
import json
import httpx
import re
import io
import base64
import asyncio
import textwrap
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# ---------------------------------------------------------
# IMPORT CÁC THƯ VIỆN XỬ LÝ ẢNH & FILE
# ---------------------------------------------------------
from PIL import Image

DOCX_AVAILABLE = False
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    print("[WARNING] Chưa cài python-docx.")

PYPDF_AVAILABLE = False
try:
    import PyPDF2
    PYPDF_AVAILABLE = True
except ImportError:
    print("[WARNING] Chưa cài PyPDF2. PDF sẽ không được chia trang.")

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="SaaS OCR Reader Backend")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root_endpoint():
    return JSONResponse(content={
        "status": "online",
        "message": "Backend OCR Reader (Async) đang chạy!",
    })

class ExtractRequest(BaseModel):
    fileBase64: Optional[str] = None
    mimeType: Optional[str] = None
    rawText: Optional[str] = None

# ==========================================
# THUẬT TOÁN HỖ TRỢ: CHIA NHỎ PDF THÀNH TỪNG TRANG
# ==========================================
def split_pdf_base64_to_pages(pdf_b64: str) -> list[tuple[str, str]]:
    if not PYPDF_AVAILABLE:
        print("[WARNING] Không có PyPDF2, sẽ gửi nguyên file PDF.")
        return [(pdf_b64, "application/pdf")]
        
    try:
        # Xóa header base64 nếu còn sót
        clean_b64 = re.sub(r'^data:[a-zA-Z0-9/+]+;base64,', '', pdf_b64)
        pdf_bytes = base64.b64decode(clean_b64)
        
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        total_pages = len(pdf_reader.pages)
        print(f"[INFO] Phát hiện tệp PDF. Tổng số trang: {total_pages}")
        
        pages_b64_list = []
        for i in range(total_pages):
            pdf_writer = PyPDF2.PdfWriter()
            pdf_writer.add_page(pdf_reader.pages[i])
            
            output_stream = io.BytesIO()
            pdf_writer.write(output_stream)
            page_bytes = output_stream.getvalue()
            
            page_b64 = base64.b64encode(page_bytes).decode('utf-8')
            pages_b64_list.append((page_b64, "application/pdf"))
            
            del pdf_writer
            output_stream.close()
            
        return pages_b64_list
    except Exception as e:
        print(f"[ERROR] Lỗi phân tách trang PDF: {str(e)}")
        return [(pdf_b64, "application/pdf")]

# ==========================================
# 1. API XỬ LÝ OCR (HỖ TRỢ ĐA LUỒNG - CONCURRENCY)
# ==========================================
@app.post("/api/extract") 
async def extract_text(req: ExtractRequest):
    print("\n========== BẮT ĐẦU XỬ LÝ YÊU CẦU OCR (ASYNC) ==========")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return JSONResponse(status_code=500, content={"error": "Chưa cấu hình GEMINI_API_KEY."})

    model_name = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    PROMPT_TEXT = r"""
Bạn là một Hệ thống Trích xuất Dữ liệu OCR chuyên nghiệp. Nhiệm vụ của bạn là số hóa nội dung và BẮT BUỘC trả về định dạng JSON là một MẢNG (ARRAY) CHỨA CÁC OBJECT. 

🚨 QUY TẮC SỐNG CÒN:
- TUYỆT ĐỐI BÁM SÁT nội dung gốc. KHÔNG TỰ BỊA CHỮ, KHÔNG tự giải bài tập.
- Chỉ làm nhiệm vụ của một máy đọc (OCR) thuần túy.

Cấu trúc JSON bắt buộc:
[
  {
    "visual": "Đề kiểm tra môn Vật Lý\n\nCâu 1: Tính vận tốc...",
    "spoken": "Đề kiểm tra môn Vật Lý. Câu một: Tính vận tốc..."
  }
]

📐 QUY TẮC CHO "visual":
- KHÔNG DÙNG THẺ HTML. Dùng ký tự ngắt dòng `\n\n` để chia đoạn.
- Mọi công thức Toán/Lý/Hóa BẮT BUỘC dùng mã LaTeX. (Nhân đôi dấu gạch chéo ngược: `\\frac{a}{b}`).
- Inline: Bọc bằng `$`. Block: Bọc bằng `$$`.

🚨 QUY TẮC CHO "spoken" (ĐỂ ĐỌC TTS):
- Chia câu ngắn (1-2 câu). KHÔNG chứa LaTeX, dịch ra tiếng Việt trơn.
- Chỉ chứa chữ cái, số và dấu câu cơ bản.
    """

    # DANH SÁCH CÁC TÁC VỤ CẦN QUÉT
    items_to_scan = [] 

    if req.fileBase64 and req.mimeType:
        clean_b64 = req.fileBase64.split(",", 1)[1] if "," in req.fileBase64 else req.fileBase64
        mime_type_lower = req.mimeType.lower()
        
        # 1. FILE WORD (Chỉ xử lý 1 lần, không cần chia nhỏ)
        if "wordprocessingml.document" in mime_type_lower or "msword" in mime_type_lower:
            if not DOCX_AVAILABLE:
                return JSONResponse(status_code=500, content={"error": "Thiếu thư viện python-docx."})
            try:
                doc = Document(io.BytesIO(base64.b64decode(clean_b64)))
                extracted_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                items_to_scan.append({"type": "text", "content": f"Nội dung file Word:\n{extracted_text}"})
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Lỗi đọc Word: {e}"})
                
        # 2. FILE ẢNH (Chia nhỏ nếu quá dài)
        elif mime_type_lower.startswith("image/"):
            try:
                img = Image.open(io.BytesIO(base64.b64decode(clean_b64)))
                width, height = img.size
                MAX_HEIGHT = 2000 
                
                if height > MAX_HEIGHT:
                    print(f"[INFO] Ảnh quá dài. Tiến hành cắt nhỏ...")
                    for i in range(0, height, MAX_HEIGHT):
                        box = (0, i, width, min(i + MAX_HEIGHT, height))
                        chunk_img = img.crop(box)
                        buffered = io.BytesIO()
                        if chunk_img.mode in ("RGBA", "P"):
                            chunk_img = chunk_img.convert("RGB")
                        chunk_img.save(buffered, format="JPEG")
                        chunk_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        items_to_scan.append({"type": "inline", "b64": chunk_b64, "mime": "image/jpeg"})
                else:
                    items_to_scan.append({"type": "inline", "b64": clean_b64, "mime": req.mimeType})
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Lỗi xử lý ảnh: {e}"})
                
        # 3. FILE PDF (Chia thành từng trang bằng PyPDF2)
        elif "application/pdf" in mime_type_lower:
            pdf_pages = split_pdf_base64_to_pages(clean_b64)
            for page_b64, p_mime in pdf_pages:
                items_to_scan.append({"type": "inline", "b64": page_b64, "mime": p_mime})
                
        # 4. ĐỊNH DẠNG KHÁC
        else:
            items_to_scan.append({"type": "inline", "b64": clean_b64, "mime": req.mimeType})

    if not items_to_scan and not req.rawText:
        return JSONResponse(status_code=400, content={"error": "Không có dữ liệu đầu vào."})

    # ==========================================
    # CƠ CHẾ XỬ LÝ ĐA LUỒNG (CONCURRENCY)
    # ==========================================
    max_concurrent_tasks = 5  
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    async def process_single_page(idx: int, item: dict, client: httpx.AsyncClient):
        async with semaphore:
            print(f"[INFO] Đang thực thi OCR tác vụ {idx + 1}/{len(items_to_scan)}...")
            
            parts = []
            if item["type"] == "text":
                parts.append({"text": item["content"]})
            else:
                parts.append({"inlineData": {"mimeType": item["mime"], "data": item["b64"]}})
                
            if req.rawText:
                parts.append({"text": req.rawText})
                
            parts.append({"text": PROMPT_TEXT})
            
            payload = {
                "contents": [{"parts": parts}],
                "safetySettings": [
                    {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
                    {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
                ],
                "generationConfig": {
                    "temperature": 0.0, 
                    "maxOutputTokens": 8192,
                    "responseMimeType": "application/json" 
                }
            }
            
            try:
                # Tăng timeout cho từng tác vụ riêng lẻ
                resp = await client.post(url, json=payload, timeout=120.0)
                if resp.status_code != 200:
                    print(f"[LỖI TÁC VỤ {idx + 1}] Gemini báo lỗi: {resp.text}")
                    return [] # Trả về mảng rỗng để không làm hỏng toàn bộ tiến trình
                    
                data = resp.json()
                candidate = data.get("candidates", [])[0]
                raw_result = candidate["content"]["parts"][0]["text"].strip()
                
                # Cứu vãn JSON
                raw_result = re.sub(r',\s*([\]}])', r'\1', raw_result)
                if not raw_result.endswith("]"):
                    last_brace = raw_result.rfind("}")
                    if last_brace != -1:
                        raw_result = raw_result[:last_brace + 1] + "]"
                        
                parsed_json = json.loads(raw_result, strict=False)
                print(f"[THÀNH CÔNG TÁC VỤ {idx + 1}] Trích xuất {len(parsed_json)} đoạn.")
                return parsed_json
            except Exception as e:
                print(f"[LỖI TÁC VỤ {idx + 1}] Lỗi mạng hoặc parse JSON: {e}")
                return []

    # Chạy toàn bộ các tác vụ song song
    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
    async with httpx.AsyncClient(trust_env=False, limits=limits) as client:
        tasks = [
            process_single_page(idx, item, client) 
            for idx, item in enumerate(items_to_scan)
        ]
        
        # Hàm gather sẽ đợi tất cả các trang quét xong
        results = await asyncio.gather(*tasks)

    # ==========================================
    # GỘP KẾT QUẢ VÀ TRẢ VỀ FRONTEND
    # ==========================================
    final_merged_json = []
    for res_array in results:
        if isinstance(res_array, list):
            final_merged_json.extend(res_array)
            
    if not final_merged_json:
        return JSONResponse(status_code=500, content={"error": "OCR thất bại trên tất cả các trang. Vui lòng thử lại."})

    print(f"\n[HOÀN TẤT] Tổng cộng đã quét thành công {len(final_merged_json)} đoạn OCR.")
    return {"result": final_merged_json}


# ==========================================
# 2. API PROXY GOOGLE TTS (ĐỌC TỪNG CÂU)
# ==========================================
@app.get("/api/tts")
async def get_tts(text: str = Query(...), lang: str = "vi"):
    if not text or not text.strip():
        return JSONResponse(status_code=400, content={"error": "Văn bản rỗng."})
        
    target_url = "https://translate.googleapis.com/translate_tts"
    headers = {"User-Agent": "Mozilla/5.0"}
    text_chunks = textwrap.wrap(text, width=200, break_long_words=False)
    
    async def stream_audio():
        async with httpx.AsyncClient(trust_env=False) as client:
            for chunk in text_chunks:
                params = {"client": "gtx", "ie": "UTF-8", "tl": lang, "q": chunk}
                try:
                    async with client.stream("GET", target_url, params=params, headers=headers) as r:
                        if r.status_code == 200:
                            async for data in r.aiter_bytes():
                                yield data
                except Exception as e:
                    print(f"[TTS STREAM ERROR]: {e}")

    return StreamingResponse(stream_audio(), media_type="audio/mpeg")


# ==========================================
# 3. API GHÉP NỐI MP3 HÀNG LOẠT (AUDIOBOOK)
# ==========================================
class BulkTTSRequest(BaseModel):
    texts: list[str]
    lang: str = "vi"

@app.post("/api/tts/bulk")
async def bulk_tts(req: BulkTTSRequest):
    print(f"\n========== TỔNG HỢP AUDIO TỔNG ({len(req.texts)} phần tử) ==========")
    headers = {"User-Agent": "Mozilla/5.0"}
    combined_audio = bytearray()
    target_url = "https://translate.googleapis.com/translate_tts"
    
    async with httpx.AsyncClient(trust_env=False) as client:
        for text in req.texts:
            if not text or not text.strip(): continue
            text_chunks = textwrap.wrap(text, width=200, break_long_words=False)
            
            for chunk in text_chunks:
                params = {"client": "gtx", "ie": "UTF-8", "tl": req.lang, "q": chunk}
                try:
                    resp = await client.get(target_url, params=params, headers=headers, timeout=15.0)
                    if resp.status_code == 200:
                        combined_audio.extend(resp.content)
                except Exception as e:
                    print(f"[WARNING] Bỏ qua đoạn lỗi mạng: {e}")
                
    if not combined_audio:
        return JSONResponse(status_code=500, content={"error": "Không thể tải audio."})
        
    return StreamingResponse(
        io.BytesIO(combined_audio), 
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment; filename=Merged_OCR_AudioBook.mp3"}
    )


if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
