import os
import json
import httpx
import re
import io
import base64
import asyncio
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

from PIL import Image

DOCX_AVAILABLE = False
try:
    from docx import Document
    DOCX_AVAILABLE = True
except ImportError:
    pass

PYPDF_AVAILABLE = False
try:
    import PyPDF2
    PYPDF_AVAILABLE = True
except ImportError:
    pass

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
        "message": "Backend OCR Reader (Hybrid) đang chạy mượt mà!",
    })

class ExtractRequest(BaseModel):
    fileBase64: Optional[str] = None
    mimeType: Optional[str] = None
    rawText: Optional[str] = None

def split_pdf_base64_to_pages(pdf_b64: str) -> list[tuple[str, str]]:
    if not PYPDF_AVAILABLE:
        return [(pdf_b64, "application/pdf")]
        
    try:
        clean_b64 = re.sub(r'^data:[a-zA-Z0-9/+]+;base64,', '', pdf_b64)
        pdf_bytes = base64.b64decode(clean_b64)
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(pdf_bytes))
        
        pages_b64_list = []
        for i in range(len(pdf_reader.pages)):
            pdf_writer = PyPDF2.PdfWriter()
            pdf_writer.add_page(pdf_reader.pages[i])
            output_stream = io.BytesIO()
            pdf_writer.write(output_stream)
            page_b64 = base64.b64encode(output_stream.getvalue()).decode('utf-8')
            pages_b64_list.append((page_b64, "application/pdf"))
            output_stream.close()
            
        return pages_b64_list
    except Exception:
        return [(pdf_b64, "application/pdf")]

# ==========================================
# 1. API TRÍCH XUẤT OCR (Bản xịn không lỗi)
# ==========================================
@app.post("/api/extract") 
async def extract_text(req: ExtractRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return JSONResponse(status_code=500, content={"error": "Chưa cấu hình GEMINI_API_KEY."})

    model_name = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    PROMPT_TEXT = r"""
Bạn là Hệ thống Trích xuất Dữ liệu OCR chuyên nghiệp. Nhiệm vụ của bạn là số hóa nội dung một cách chính xác tuyệt đối.

🚨 QUY TẮC SỐNG CÒN:
- BÁM SÁT nội dung gốc. KHÔNG TỰ BỊA CHỮ, KHÔNG giải bài tập.
- Cứ xong 2-3 câu văn, hoặc 1 câu hỏi trắc nghiệm thì ngắt ra một object mới.

📐 QUY TẮC "visual" (Nội dung gốc - HIỂN THỊ CHUẨN):
- KHÔNG DÙNG THẺ HTML. 
- BẮT BUỘC dùng `\n\n` để ngắt đoạn, đảm bảo khoảng cách rộng rãi, KHÔNG dính liền nhau.
- Gặp dòng kẻ đứt hoặc chỗ trống cần điền, hãy in ra dấu chấm. (Hệ thống sẽ tự thay thế sau).
- Công thức Toán/Lý/Hóa BẮT BUỘC dùng mã LaTeX. Inline: bọc bằng `$`. Block: bọc bằng `$$`.

🚨 QUY TẮC "spoken" (Đọc TTS):
- Dịch hoàn toàn ra tiếng Việt trơn (vd: $v$ -> "vận tốc", $\frac{1}{2}$ -> "một phần hai").
- Không chứa ký hiệu Toán học/LaTeX, chia thành câu ngắn.
    """

    items_to_scan = [] 

    if req.fileBase64 and req.mimeType:
        clean_b64 = req.fileBase64.split(",", 1)[1] if "," in req.fileBase64 else req.fileBase64
        mime_type_lower = req.mimeType.lower()
        
        if "wordprocessingml.document" in mime_type_lower or "msword" in mime_type_lower:
            if not DOCX_AVAILABLE:
                return JSONResponse(status_code=500, content={"error": "Thiếu thư viện python-docx."})
            try:
                doc = Document(io.BytesIO(base64.b64decode(clean_b64)))
                extracted_text = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                CHUNK_SIZE = 1500 
                for i in range(0, len(extracted_text), CHUNK_SIZE):
                    items_to_scan.append({"type": "text", "content": f"Nội dung file Word:\n{extracted_text[i:i+CHUNK_SIZE]}"})
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Lỗi đọc Word: {e}"})
                
        elif mime_type_lower.startswith("image/"):
            try:
                img = Image.open(io.BytesIO(base64.b64decode(clean_b64)))
                width, height = img.size
                if width > 1500:
                    ratio = 1500 / width
                    img = img.resize((1500, int(height * ratio)), Image.Resampling.LANCZOS)
                    width, height = img.size

                MAX_HEIGHT = 600
                if height > MAX_HEIGHT:
                    for i in range(0, height, MAX_HEIGHT):
                        box = (0, i, width, min(i + MAX_HEIGHT, height))
                        chunk_img = img.crop(box)
                        buffered = io.BytesIO()
                        if chunk_img.mode in ("RGBA", "P"):
                            chunk_img = chunk_img.convert("RGB")
                        chunk_img.save(buffered, format="JPEG", quality=85)
                        chunk_b64 = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        items_to_scan.append({"type": "inline", "b64": chunk_b64, "mime": "image/jpeg"})
                else:
                    items_to_scan.append({"type": "inline", "b64": clean_b64, "mime": req.mimeType})
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Lỗi xử lý ảnh: {e}"})
                
        elif "application/pdf" in mime_type_lower:
            pdf_pages = split_pdf_base64_to_pages(clean_b64)
            for page_b64, p_mime in pdf_pages:
                items_to_scan.append({"type": "inline", "b64": page_b64, "mime": p_mime})
        else:
            items_to_scan.append({"type": "inline", "b64": clean_b64, "mime": req.mimeType})

    if req.rawText:
        CHUNK_SIZE = 1500
        for i in range(0, len(req.rawText), CHUNK_SIZE):
            items_to_scan.append({"type": "text", "content": req.rawText[i:i+CHUNK_SIZE]})

    if not items_to_scan:
        return JSONResponse(status_code=400, content={"error": "Không có dữ liệu đầu vào."})

    max_concurrent_tasks = 5  
    semaphore = asyncio.Semaphore(max_concurrent_tasks)
    
    json_schema = {
        "type": "ARRAY",
        "description": "Danh sách các đoạn văn bản trích xuất được.",
        "items": {
            "type": "OBJECT",
            "properties": {
                "visual": {"type": "STRING"},
                "spoken": {"type": "STRING"}
            },
            "required": ["visual", "spoken"]
        }
    }
    
    async def process_single_page(item: dict, client: httpx.AsyncClient):
        async with semaphore:
            parts = [{"text": item["content"]}] if item["type"] == "text" else [{"inlineData": {"mimeType": item["mime"], "data": item["b64"]}}]
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
                    "responseMimeType": "application/json",
                    "responseSchema": json_schema
                }
            }
            
            for _ in range(3):
                try:
                    resp = await client.post(url, json=payload, timeout=60.0)
                    if resp.status_code == 200:
                        raw_result = resp.json().get("candidates", [])[0]["content"]["parts"][0]["text"].strip()
                        return json.loads(raw_result)
                except Exception:
                    await asyncio.sleep(1)
            return []

    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
    async with httpx.AsyncClient(trust_env=False, limits=limits) as client:
        tasks = [process_single_page(item, client) for item in items_to_scan]
        results = await asyncio.gather(*tasks)

    final_merged_json = []
    for res_array in results:
        if isinstance(res_array, list):
            for item in res_array:
                # Sửa Regex: visual -> gạch dưới dài, spoken -> xóa đi để khỏi đọc "chấm chấm"
                if "visual" in item and isinstance(item["visual"], str):
                    item["visual"] = re.sub(r'\.{2,}', '_______________', item["visual"])
                if "spoken" in item and isinstance(item["spoken"], str):
                    item["spoken"] = re.sub(r'[_\.]{2,}', '', item["spoken"])
            final_merged_json.extend(res_array)
            
    if not final_merged_json:
        return JSONResponse(status_code=500, content={"error": "Trích xuất thất bại."})

    return {"result": final_merged_json}

# ==========================================
# HÀM BĂM NHỎ TEXT ĐỂ VƯỢT GIỚI HẠN GOOGLE TTS
# ==========================================
def split_text_for_google(text: str, max_chars: int = 200) -> list[str]:
    """Cắt nhỏ câu quá dài (dựa theo dấu cách) để API Google không bị lỗi."""
    words = text.split()
    chunks = []
    current_chunk = []
    current_length = 0
    
    for word in words:
        if current_length + len(word) + 1 > max_chars:
            chunks.append(" ".join(current_chunk))
            current_chunk = [word]
            current_length = len(word)
        else:
            current_chunk.append(word)
            current_length += len(word) + 1
            
    if current_chunk:
        chunks.append(" ".join(current_chunk))
    return chunks

# ==========================================
# 2. API GOOGLE TTS STREAM (ĐỌC LẺ)
# ==========================================
@app.get("/api/tts")
async def get_tts(text: str = Query(...), lang: str = "vi"):
    target_url = "https://translate.googleapis.com/translate_tts"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    # Cắt nhỏ văn bản nếu nó dài hơn 200 ký tự
    text_chunks = split_text_for_google(text, max_chars=200)
    
    async def stream_audio():
        async with httpx.AsyncClient(trust_env=False) as client:
            for chunk in text_chunks:
                params = {"client": "gtx", "ie": "UTF-8", "tl": lang, "q": chunk}
                async with client.stream("GET", target_url, params=params, headers=headers) as r:
                    if r.status_code == 200:
                        async for byte_chunk in r.aiter_bytes():
                            yield byte_chunk

    return StreamingResponse(stream_audio(), media_type="audio/mpeg")

# ==========================================
# 3. API GHÉP NỐI MP3 HÀNG LOẠT BẰNG GOOGLE TTS 
# ==========================================
class BulkTTSRequest(BaseModel):
    texts: list[str]
    lang: str = "vi"

@app.post("/api/tts/bulk")
async def bulk_tts(req: BulkTTSRequest):
    print(f"\n========== TỔNG HỢP AUDIO TỔNG ({len(req.texts)} phần tử) ==========")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    combined_audio = bytearray()
    target_url = "https://translate.googleapis.com/translate_tts"
    
    async with httpx.AsyncClient(trust_env=False) as client:
        for block_text in req.texts:
            if not block_text or not block_text.strip():
                continue
            
            # Khắc phục lỗi "chỉ xuất được 1 câu" -> cắt nhỏ văn bản trước khi gửi Google
            chunks = split_text_for_google(block_text, max_chars=190)
            
            for chunk in chunks:
                params = {
                    "client": "gtx",
                    "ie": "UTF-8",
                    "tl": req.lang,
                    "q": chunk
                }
                try:
                    resp = await client.get(target_url, params=params, headers=headers, timeout=15.0)
                    if resp.status_code == 200:
                        combined_audio.extend(resp.content)
                except Exception as e:
                    print(f"[WARNING] Bỏ qua đoạn âm thanh lỗi: {e}")
                
    if not combined_audio:
        return JSONResponse(status_code=500, content={"error": "Không thể tải audio từ server TTS."})
        
    return StreamingResponse(
        io.BytesIO(combined_audio), 
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment; filename=Merged_OCR_AudioBook.mp3"}
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
