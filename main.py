import os
import json
import httpx
import re
import io
import base64
import asyncio
import hashlib
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

    PROMPT_TEXT = r"""
Bạn là Hệ thống Trích xuất Dữ liệu OCR chuyên nghiệp. Nhiệm vụ của bạn là số hóa nội dung một cách chính xác tuyệt đối.

📐 QUY TẮC "visual" (Nội dung gốc - SỬA HIỂN THỊ RÕ RÀNG):
- KHÔNG DÙNG THẺ HTML. 
- BẮT BUỘC dùng `\n\n` (hai dấu xuống dòng) để ngắt đoạn, ngắt câu hỏi. Phải đảm bảo khoảng cách rộng rãi, KHÔNG để các dòng dính liền nhau.
- Công thức Toán/Lý/Hóa BẮT BUỘC dùng mã LaTeX. Inline: bọc bằng `$`. Block: bọc bằng `$$`.

🚨 QUY TẮC "spoken" (Đọc TTS):
- Dịch hoàn toàn ra tiếng Việt trơn (vd: $v$ -> "vận tốc", $\frac{1}{2}$ -> "một phần hai").
- Không chứa ký hiệu Toán học/LaTeX, chia thành câu ngắn.

🚨 QUY TẮC SỐNG CÒN:
- Cứ sau 2-3 câu văn hoặc mỗi câu trắc nghiệm, hãy ngắt cấu trúc ra thành một object riêng biệt trong mảng JSON.
- Đảm bảo trả về mảng JSON hợp lệ chứa các object có thuộc tính "visual" và "spoken".
    """

    parts = []
    if req.fileBase64 and req.mimeType:
        if "," in req.fileBase64:
            clean_b64 = req.fileBase64.split(",", 1)[1]
        else:
            clean_b64 = req.fileBase64
            
        mime_type_lower = req.mimeType.lower()
        
        if "wordprocessingml.document" in mime_type_lower or "msword" in mime_type_lower:
            if not DOCX_AVAILABLE:
                return JSONResponse(status_code=500, content={"error": "Hệ thống thiếu thư viện python-docx."})
            
            try:
                print("[INFO] Đang bóc tách text từ file Word (.docx)...")
                doc_bytes = base64.b64decode(clean_b64)
                doc = Document(io.BytesIO(doc_bytes))
                extracted_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                
                if not extracted_text.strip():
                    return JSONResponse(status_code=400, content={"error": "File Word này bị trống hoặc chỉ chứa hình ảnh."})
                    
                parts.append({"text": f"Nội dung file tài liệu Word được cung cấp:\n{extracted_text}"})
            except Exception as e:
                return JSONResponse(status_code=400, content={"error": f"Lỗi đọc file Word: {str(e)}"})
        else:
            parts.append({"inlineData": {"mimeType": req.mimeType, "data": clean_b64}})
        
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
            "temperature": 0.0,  # Ép hẳn về 0.0 chống AI tự bịa từ
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json" 
        }
    }

    async with httpx.AsyncClient(trust_env=False) as client:
        try:
            print("[INFO] Đang chuyển tiếp gói tin đến Google Gemini API...")
            response = await client.post(url, json=payload, timeout=60.0)
            
            if response.status_code != 200:
                return JSONResponse(status_code=response.status_code, content={"error": f"Lỗi Gemini: {response.text}"})
            
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return JSONResponse(status_code=400, content={"error": "Bị chặn bởi chính sách an toàn đầu vào."})
                
            candidate = candidates[0]
            raw_result = candidate["content"]["parts"][0]["text"].strip()
            raw_result = re.sub(r',\s*([\]}])', r'\1', raw_result)
            
            try:
                parsed_json = json.loads(raw_result, strict=False)
                
                # --- XỬ LÝ REGEX ĐỔI DẤU CHẤM THÀNH GẠCH NGANG ---
                if isinstance(parsed_json, list):
                    for item in parsed_json:
                        if "visual" in item and isinstance(item["visual"], str):
                            # Đếm bao nhiêu dấu chấm liên tiếp (từ 2 dấu trở lên) thì thay bằng bấy nhiêu dấu gạch ngang
                            item["visual"] = re.sub(r'\.{2,}', lambda m: '-' * len(m.group()), item["visual"])
                        
                        if "spoken" in item and isinstance(item["spoken"], str):
                            # Ở phần đọc thành tiếng, xóa bỏ chuỗi dấu chấm/gạch để TTS không đọc vấp
                            item["spoken"] = re.sub(r'\.{2,}', '', item["spoken"])
                            item["spoken"] = re.sub(r'_{2,}', '', item["spoken"])

                print(f"[SUCCESS] Trích xuất thành công {len(parsed_json)} đoạn văn bản.")
                return {"result": parsed_json} 
            except json.JSONDecodeError as e:
                return JSONResponse(status_code=500, content={"error": "AI trả về chuỗi JSON không hợp lệ.", "raw": raw_result})
            
        except httpx.ReadTimeout:
            return JSONResponse(status_code=504, content={"error": "Quá thời gian phản hồi (60 giây)."})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Lỗi nội bộ: {str(e)}"})


# ==========================================
# 2. API PROXY EDGE TTS (ĐỌC LẺ)
# ==========================================
def clean_text_for_tts(text: str) -> str:
    return text.replace("&", " và ").replace("<", " ").replace(">", " ").replace("#", " ")

@app.get("/api/tts")
async def get_tts(text: str = Query(...), lang: str = "vi"):
    if not text or not text.strip():
        return JSONResponse(status_code=400, content={"error": "Văn bản rỗng."})

    voice = "vi-VN-HoaiMyNeural" if "vi" in lang.lower() else "en-US-AriaNeural"
    clean_text = clean_text_for_tts(text.strip())

    text_hash = hashlib.md5(f"{clean_text}_{voice}".encode('utf-8')).hexdigest()
    cache_path = os.path.join("tts_cache", f"{text_hash}.mp3")

    if os.path.exists(cache_path):
        return FileResponse(cache_path, media_type="audio/mpeg")

    try:
        communicate = edge_tts.Communicate(clean_text, voice)
        audio_data = bytearray()
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
                
        with open(cache_path, "wb") as f:
            f.write(audio_data)

        return Response(content=bytes(audio_data), media_type="audio/mpeg")
    except Exception as e:
        return JSONResponse(status_code=500, content={"error": f"Lỗi âm thanh: {str(e)}"})


# ==============================================================
# 3. API ĐỌC HÀNG LOẠT (BULK TTS) - TẢI 1 FILE DUY NHẤT ĐỌC LIÊN TỤC
# ==============================================================
class BulkTTSRequest(BaseModel):
    texts: list[str]
    lang: str = "vi"

@app.post("/api/tts/bulk")
async def bulk_tts(req: BulkTTSRequest):
    print(f"\n========== TỔNG HỢP AUDIO TỔNG ({len(req.texts)} phần tử) ==========")
    
    # Ghép các câu lại với nhau, cách nhau bằng khoảng lặng hợp lý (dấu chấm lửng)
    full_text = " ... ".join([t.strip() for t in req.texts if t and t.strip()])
    if not full_text:
        return JSONResponse(status_code=400, content={"error": "Danh sách văn bản trống."})

    clean_text = clean_text_for_tts(full_text)
    voice = "vi-VN-HoaiMyNeural" if "vi" in req.lang.lower() else "en-US-AriaNeural"
    
    # Tạo mã hash cho file gộp để làm cache
    text_hash = hashlib.md5(f"bulk_{clean_text}_{voice}".encode('utf-8')).hexdigest()
    cache_path = os.path.join("tts_cache", f"{text_hash}_merged.mp3")
    
    # Nếu file đã tồn tại trong cache, trả về luôn để Frontend Play ngay lập tức
    if os.path.exists(cache_path):
        return FileResponse(
            cache_path, 
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=Full_AudioBook.mp3"}
        )
        
    try:
        # Chuyển đổi toàn bộ khối văn bản thành 1 file âm thanh duy nhất bằng edge_tts
        communicate = edge_tts.Communicate(clean_text, voice)
        audio_data = bytearray()
        
        async for chunk in communicate.stream():
            if chunk["type"] == "audio":
                audio_data.extend(chunk["data"])
                
        with open(cache_path, "wb") as f:
            f.write(audio_data)
            
        return FileResponse(
            cache_path, 
            media_type="audio/mpeg",
            headers={"Content-Disposition": "attachment; filename=Full_AudioBook.mp3"}
        )
    except Exception as e:
        print(f"[ERROR] Lỗi tạo bulk TTS: {e}")
        return JSONResponse(status_code=500, content={"error": f"Không thể gộp file âm thanh: {str(e)}"})

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
