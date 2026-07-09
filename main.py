import os
import json
import httpx
import re
import io
import base64
import asyncio
import textwrap
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import edge_tts  

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
        "message": "Backend OCR Reader (Async) đang chạy cực mượt!",
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
        total_pages = len(pdf_reader.pages)
        
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
    except Exception:
        return [(pdf_b64, "application/pdf")]

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
- Chia nhỏ nội dung ra làm nhiều phần tử. Cứ xong 2-3 câu văn, hoặc 1 câu hỏi trắc nghiệm thì tạo một object mới.

📐 QUY TẮC "visual" (Nội dung gốc):
- KHÔNG DÙNG THẺ HTML. Dùng `\n\n` để ngắt đoạn.
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
                extracted_text = "\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                for i in range(0, len(extracted_text), 3000):
                    items_to_scan.append({"type": "text", "content": f"Nội dung file Word:\n{extracted_text[i:i+3000]}"})
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Lỗi đọc Word: {e}"})
                
        elif mime_type_lower.startswith("image/"):
            try:
                img = Image.open(io.BytesIO(base64.b64decode(clean_b64)))
                width, height = img.size
                MAX_HEIGHT = 1000  
                
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
        for i in range(0, len(req.rawText), 3000):
            items_to_scan.append({"type": "text", "content": req.rawText[i:i+3000]})

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
                "visual": {
                    "type": "STRING",
                    "description": "Văn bản gốc giữ nguyên bố cục. Công thức dùng mã LaTeX."
                },
                "spoken": {
                    "type": "STRING",
                    "description": "Văn bản dịch thuần tiếng Việt để đọc TTS."
                }
            },
            "required": ["visual", "spoken"]
        }
    }
    
    async def process_single_page(idx: int, item: dict, client: httpx.AsyncClient):
        async with semaphore:
            parts = []
            if item["type"] == "text":
                parts.append({"text": item["content"]})
            else:
                parts.append({"inlineData": {"mimeType": item["mime"], "data": item["b64"]}})
                
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
            
            max_retries = 3
            for attempt in range(max_retries):
                try:
                    resp = await client.post(url, json=payload, timeout=120.0)
                    if resp.status_code == 200:
                        data = resp.json()
                        candidate = data.get("candidates", [])[0]
                        raw_result = candidate["content"]["parts"][0]["text"].strip()
                        
                        try:
                            parsed_json = json.loads(raw_result)
                            return parsed_json
                        except json.JSONDecodeError:
                            try:
                                last_brace = raw_result.rfind('}')
                                if last_brace != -1:
                                    fixed_raw = raw_result[:last_brace+1] + ']'
                                    parsed_json = json.loads(fixed_raw)
                                    return parsed_json
                            except Exception:
                                pass
                            return []
                            
                    elif resp.status_code in [429, 503]:
                        if attempt < max_retries - 1:
                            await asyncio.sleep(2 ** (attempt + 1))
                            continue
                        return []
                    else:
                        print(f"Lỗi API Gemini - Mã: {resp.status_code}")
                        return []
                        
                except Exception as e:
                    print(f"Lỗi kết nối Gemini: {e}")
                    if attempt < max_retries - 1:
                        await asyncio.sleep(2 ** (attempt + 1))
                        continue
                    return []
            return []

    limits = httpx.Limits(max_keepalive_connections=10, max_connections=20)
    async with httpx.AsyncClient(trust_env=False, limits=limits) as client:
        tasks = [
            process_single_page(idx, item, client) 
            for idx, item in enumerate(items_to_scan)
        ]
        results = await asyncio.gather(*tasks)

    final_merged_json = []
    for res_array in results:
        if isinstance(res_array, list):
            final_merged_json.extend(res_array)
            
    if not final_merged_json:
        return JSONResponse(status_code=500, content={"error": "Máy chủ AI đang bận. Vui lòng thử lại."})

    return {"result": final_merged_json}


# ==============================================================
# HỆ THỐNG TTS MỚI: DỌN SẠCH KÝ TỰ CẤM VÀ ĐÓNG GÓI 1 FILE CHUẨN
# ==============================================================

def clean_ssml_chars(text: str) -> str:
    """Loại bỏ ký tự gây sập nguồn hệ thống Edge-TTS"""
    if not text:
        return ""
    return text.replace("&", " và ").replace("<", " ").replace(">", " ").replace("#", " ")

@app.get("/api/tts")
async def get_tts(text: str = Query(...), lang: str = "vi"):
    if not text or not text.strip():
        return JSONResponse(status_code=400, content={"error": "Văn bản rỗng."})

    voice = "vi-VN-HoaiMyNeural" if "vi" in lang.lower() else "en-US-AriaNeural"
    clean_text = clean_ssml_chars(text.strip())

    try:
        communicate = edge_tts.Communicate(clean_text, voice)
        audio_data = bytearray()
        
        # Gom toàn bộ âm thanh thành 1 cục duy nhất
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
                
        # Trả về Response nguyên khối để trình duyệt biết độ dài, không tự tắt ngang
        return Response(content=bytes(audio_data), media_type="audio/mpeg")
        
    except Exception as e:
        print(f"Lỗi Edge TTS: {e}")
        return JSONResponse(status_code=500, content={"error": f"Lỗi âm thanh: {str(e)}"})


class BulkTTSRequest(BaseModel):
    texts: list[str]
    lang: str = "vi"

@app.post("/api/tts/bulk")
async def bulk_tts(req: BulkTTSRequest):
    if not req.texts:
        return JSONResponse(status_code=400, content={"error": "Danh sách rỗng."})

    voice = "vi-VN-HoaiMyNeural" if "vi" in req.lang.lower() else "en-US-AriaNeural"

    try:
        combined_audio = bytearray()
        for text in req.texts:
            if not text or not text.strip():
                continue
                
            clean_text = clean_ssml_chars(text.strip())
            communicate = edge_tts.Communicate(clean_text, voice)
            
            async for chunk in communicate.stream():
                if chunk["type"] == "audio":
                    combined_audio.extend(chunk["data"])
                    
        return Response(
            content=bytes(combined_audio), 
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=Merged_OCR_AudioBook.mp3"}
        )
        
    except Exception as e:
        print(f"Lỗi Bulk TTS: {e}")
        return JSONResponse(status_code=500, content={"error": f"Lỗi tạo audio hàng loạt: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
