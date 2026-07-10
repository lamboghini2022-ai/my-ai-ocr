import os
import json
import httpx
import re
import io
import base64
import asyncio
import textwrap
import hashlib # Thêm thư viện để tạo cache
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

# Tạo thư mục static và thư mục cache cho TTS
if not os.path.exists("static"):
    os.makedirs("static")
if not os.path.exists("tts_cache"):
    os.makedirs("tts_cache")

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

    # Đã sửa lại PROMPT để ép buộc cách dòng bằng \n\n, giúp hiển thị không bị dính chữ
    PROMPT_TEXT = r"""
Bạn là Hệ thống Trích xuất Dữ liệu OCR chuyên nghiệp. Nhiệm vụ của bạn là số hóa nội dung một cách chính xác tuyệt đối.

🚨 QUY TẮC SỐNG CÒN:
- BÁM SÁT nội dung gốc. KHÔNG TỰ BỊA CHỮ, KHÔNG giải bài tập.
- Chia nhỏ nội dung ra làm nhiều phần tử. Cứ xong 2-3 câu văn, hoặc 1 câu hỏi trắc nghiệm thì tạo một object mới.

📐 QUY TẮC "visual" (Nội dung gốc - SỬA HIỂN THỊ RÕ RÀNG):
- KHÔNG DÙNG THẺ HTML. 
- BẮT BUỘC dùng `\n\n` (hai dấu xuống dòng) để ngắt đoạn, ngắt câu hỏi. Phải đảm bảo khoảng cách rộng rãi, KHÔNG để các dòng dính liền nhau.
- Công thức Toán/Lý/Hóa BẮT BUỘC dùng mã LaTeX. Inline: bọc bằng `$`. Block: bọc bằng `$$`.

🚨 QUY TẮC "spoken" (Đọc TTS):
- Dịch hoàn toàn ra tiếng Việt trơn (vd: $v$ -> "vận tốc", $\frac{1}{2}$ -> "một phần hai").
- Không chứa ký hiệu Toán học/LaTeX, chia thành câu ngắn.
🚫 QUY TẮC XỬ LÝ KHOẢNG TRỐNG ĐIỀN TỪ (DẤU CHẤM/GẠCH DƯỚI):\n
- Đối với các dòng dấu chấm (.........) hoặc nét đứt trong ảnh gốc: BẮT BUỘC CHUYỂN ĐỔI TOÀN BỘ THÀNH DẤU GẠCH DƯỚI (_________).\n
- Hãy xuất ra một dải gạch dưới dài tương đối (khoảng từ 10 đến 30 dấu gạch). TUYỆT ĐỐI KHÔNG được lặp vô tận gây lỗi.\n\n
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
                # Đã sửa \n thành \n\n để hiển thị đẹp hơn
                extracted_text = "\n\n".join([p.text for p in doc.paragraphs if p.text.strip()])
                
                # Giảm chunk từ 3000 xuống 1500 để load nhanh hơn, tránh timeout
                CHUNK_SIZE = 1500 
                for i in range(0, len(extracted_text), CHUNK_SIZE):
                    items_to_scan.append({"type": "text", "content": f"Nội dung file Word:\n{extracted_text[i:i+CHUNK_SIZE]}"})
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Lỗi đọc Word: {e}"})
                
        elif mime_type_lower.startswith("image/"):
            try:
                img = Image.open(io.BytesIO(base64.b64decode(clean_b64)))
                width, height = img.size
                
                # Khắc phục lỗi số 2: Resize nếu ảnh quá rộng và chia nhỏ MAX_HEIGHT gắt hơn
                if width > 1500:
                    ratio = 1500 / width
                    img = img.resize((1500, int(height * ratio)), Image.Resampling.LANCZOS)
                    width, height = img.size

                MAX_HEIGHT = 600  # Giảm từ 1000 xuống 600 để chia nhỏ hơn nữa
                
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
        CHUNK_SIZE = 1500 # Giảm từ 3000 xuống 1500
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
                "visual": {
                    "type": "STRING",
                    "description": "Văn bản gốc giữ nguyên bố cục. Công thức dùng mã LaTeX. LUÔN cách đoạn bằng \n\n"
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
        return JSONResponse(status_code=500, content={"error": "Máy chủ AI đang bận hoặc file quá phức tạp. Vui lòng thử lại."})

    return {"result": final_merged_json}


# ==============================================================
# HỆ THỐNG TTS: ĐÃ TÍCH HỢP CACHING ĐỂ KHÔNG PHẢI GỌI LẠI SERVER
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

    # Khắc phục lỗi số 1: Caching Audio bằng Hash
    text_hash = hashlib.md5(f"{clean_text}_{voice}".encode('utf-8')).hexdigest()
    cache_path = os.path.join("tts_cache", f"{text_hash}.mp3")

    # Nếu đã từng sinh audio này, trả về ngay lập tức (Không tốn tgian load)
    if os.path.exists(cache_path):
        return FileResponse(cache_path, media_type="audio/mpeg")

    try:
        communicate = edge_tts.Communicate(clean_text, voice)
        audio_data = bytearray()
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
                
        # Lưu lại cache cho lần sau
        with open(cache_path, "wb") as f:
            f.write(audio_data)

        return Response(content=bytes(audio_data), media_type="audio/mpeg")
        
    except Exception as e:
        print(f"Lỗi Edge TTS: {e}")
        return JSONResponse(status_code=500, content={"error": f"Lỗi âm thanh: {str(e)}"})

# ==========================================
# 3. API GHÉP NỐI MP3 HÀNG LOẠT (ĐÃ SỬA LỖI VÀ BÁO CÁO LỖI CHI TIẾT)
# ==========================================
class BulkTTSRequest(BaseModel):
    texts: list[str]
    lang: str = "vi"

def split_text_for_google_tts(text: str, max_chars: int = 150) -> list[str]:
    """Cắt nhỏ văn bản đảm bảo Google TTS không bị quá tải (dưới 200 ký tự)"""
    if len(text) <= max_chars:
        return [text]
    sentences = re.split(r'(?<=[.,!?])\s+', text)
    chunks = []
    current_chunk = ""
    for sentence in sentences:
        if len(current_chunk) + len(sentence) + 1 <= max_chars:
            current_chunk = f"{current_chunk} {sentence}".strip()
        else:
            if current_chunk:
                chunks.append(current_chunk)
            if len(sentence) > max_chars:
                words = sentence.split(' ')
                sub_chunk = ""
                for word in words:
                    if len(sub_chunk) + len(word) + 1 <= max_chars:
                        sub_chunk = f"{sub_chunk} {word}".strip()
                    else:
                        chunks.append(sub_chunk)
                        sub_chunk = word
                current_chunk = sub_chunk
            else:
                current_chunk = sentence
    if current_chunk:
        chunks.append(current_chunk)
    return chunks

@app.post("/api/tts/bulk")
async def bulk_tts(req: BulkTTSRequest):
    print(f"\n========== TỔNG HỢP AUDIO TỔNG ({len(req.texts)} phần tử) ==========")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    combined_audio = bytearray()
    target_url = "https://translate.googleapis.com/translate_tts"
    
    # Chia nhỏ text trước khi xử lý để chống gãy URL và quá tải Google TTS
    final_texts = []
    for text in req.texts:
        if text and text.strip():
            final_texts.extend(split_text_for_google_tts(text.strip(), 150))
    
    async with httpx.AsyncClient(trust_env=False) as client:
        for idx, text in enumerate(final_texts):
            # Dùng tham số 'params' thay vì nối chuỗi trực tiếp để tránh lỗi ký tự đặc biệt
            params = {
                "client": "gtx",
                "ie": "UTF-8",
                "tl": req.lang,
                "q": text
            }
            try:
                resp = await client.get(target_url, params=params, headers=headers, timeout=15.0)
                if resp.status_code == 200:
                    combined_audio.extend(resp.content)
                else:
                    # In rõ lỗi khi không thể tải đoạn văn bản
                    print(f"[LỖI GOOGLE TTS] Không thể tải đoạn {idx}. Mã lỗi: {resp.status_code} - Text: '{text[:50]}...'")
            except Exception as e:
                print(f"[WARNING] Kết nối thất bại ở đoạn âm thanh thứ {idx}: {e}")
                
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
