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
        📐 QUY TẮC "visual" (Nội dung gốc - SỬA HIỂN THỊ RÕ RÀNG):
        - KHÔNG DÙNG THẺ HTML. 
        - BẮT BUỘC dùng `\n\n` (hai dấu xuống dòng) để ngắt đoạn, ngắt câu hỏi. Phải đảm bảo khoảng cách rộng rãi, KHÔNG để các dòng dính liền nhau.
        - Công thức Toán/Lý/Hóa BẮT BUỘC dùng mã LaTeX. Inline: bọc bằng `$`. Block: bọc bằng `$$`.
        
        🚨 QUY TẮC "spoken" (Đọc TTS):
        - Dịch hoàn toàn ra tiếng Việt trơn (vd: $v$ -> "vận tốc", $\frac{1}{2}$ -> "một phần hai").
        - Không chứa ký hiệu Toán học/LaTeX, chia thành câu ngắn.
        🚨 QUY TẮC SỐNG CÒN:
        🚫 QUY TẮC 3: XỬ LÝ KHOẢNG TRỐNG ĐIỀN TỪ (DẤU CHẤM/GẠCH DƯỚI):\n"
        - Đối với các dòng dấu chấm (.........) hoặc nét đứt trong ảnh gốc: BẮT BUỘC CHUYỂN ĐỔI TOÀN BỘ THÀNH DẤU GẠCH DƯỚI (_________).\n"
        - Hãy xuất ra một dải gạch dưới dài tương đối (khoảng từ 10 đến 30 dấu gạch). TUYỆT ĐỐI KHÔNG được lặp vô tận gây lỗi.\n\n"
            """

    parts = []
    if req.fileBase64 and req.mimeType:
        # VÁ LỖI TẠI ĐÂY: Dùng split(",") thay vì regex để không bao giờ bị lỗi do MIME type dài
        if "," in req.fileBase64:
            clean_b64 = req.fileBase64.split(",", 1)[1]
        else:
            clean_b64 = req.fileBase64
            
        mime_type_lower = req.mimeType.lower()
        
        # Nhận diện file Word (kiểm tra cả docx và ms-word)
        if "wordprocessingml.document" in mime_type_lower or "msword" in mime_type_lower:
            if not DOCX_AVAILABLE:
                return JSONResponse(status_code=500, content={"error": "Hệ thống thiếu thư viện python-docx. Hãy thêm 'python-docx' vào file requirements.txt trên server."})
            
            try:
                print("[INFO] Đang bóc tách text từ file Word (.docx)...")
                doc_bytes = base64.b64decode(clean_b64)
                doc = Document(io.BytesIO(doc_bytes))
                extracted_text = "\n".join([para.text for para in doc.paragraphs if para.text.strip()])
                
                if not extracted_text.strip():
                    return JSONResponse(status_code=400, content={"error": "File Word này bị trống hoặc chỉ chứa toàn hình ảnh, không có văn bản nào để trích xuất."})
                    
                parts.append({"text": f"Nội dung file tài liệu Word được cung cấp:\n{extracted_text}"})
            except Exception as e:
                print(f"[ERROR] Lỗi đọc file Word: {e}")
                return JSONResponse(status_code=400, content={"error": f"Lỗi đọc file Word. Hãy chắc chắn đây là file định dạng .docx (đời mới), không hỗ trợ file .doc cũ. Chi tiết: {str(e)}"})
        else:
            # Nếu là PDF/Ảnh thì gửi file vào inlineData cho Gemini tự đọc
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
            "temperature": 0.2,  # ÉP NHIỆT ĐỘ VỀ 0.0 ĐỂ KHÔNG ĐƯỢC TỰ BỊA CHỮ
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json" 
        }
    }

    async with httpx.AsyncClient(trust_env=False) as client:
        try:
            print("[INFO] Đang chuyển tiếp gói tin đến Google Gemini API...")
            response = await client.post(url, json=payload, timeout=60.0)
            
            if response.status_code != 200:
                print(f"[API ERROR] Google API trả về mã lỗi: {response.status_code}")
                return JSONResponse(status_code=response.status_code, content={"error": f"Lỗi Gemini: {response.text}"})
            
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return JSONResponse(status_code=400, content={"error": "Bị chặn bởi chính sách an toàn đầu vào."})
                
            candidate = candidates[0]
            if "content" not in candidate:
                return JSONResponse(status_code=400, content={"error": "AI vi phạm bộ lọc đầu ra."})

            raw_result = candidate["content"]["parts"][0]["text"].strip()
            
            # Vá luôn lỗi AI thỉnh thoảng dư dấu phẩy ở cuối mảng JSON
            raw_result = re.sub(r',\s*([\]}])', r'\1', raw_result)
            
            try:
                parsed_json = json.loads(raw_result, strict=False)
                print(f"[SUCCESS] Trích xuất thành công {len(parsed_json)} đoạn văn bản.")
                return {"result": parsed_json} 
            except json.JSONDecodeError as e:
                print(f"[CRITICAL ERROR] JSON lỗi định dạng: {e}")
                return JSONResponse(status_code=500, content={"error": "AI trả về chuỗi JSON không hợp lệ.", "raw": raw_result})
            
        except httpx.ReadTimeout:
            print("[TIMEOUT] Quá thời gian 60 giây.")
            return JSONResponse(status_code=504, content={"error": "Quá thời gian phản hồi (60 giây)."})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Lỗi nội bộ: {str(e)}"})

# ==========================================
# 2. API PROXY GOOGLE TTS (ĐỌC TỪNG CÂU)
# ==========================================
@app.get("/api/tts")
async def get_tts(text: str = Query(...), lang: str = "vi"):
    target_url = "https://translate.googleapis.com/translate_tts"
    # Dùng params để HTTPX tự động xử lý dấu cách, chống lỗi sập server
    params = {
        "client": "gtx",
        "ie": "UTF-8",
        "tl": lang,
        "q": text
    }
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    async def stream_audio():
        async with httpx.AsyncClient(trust_env=False) as client:
            async with client.stream("GET", target_url, params=params, headers=headers) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk

    return StreamingResponse(stream_audio(), media_type="audio/mpeg")

# ==========================================
# 3. API GHÉP NỐI MP3 HÀNG LOẠT 
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
        for text in req.texts:
            if not text or not text.strip():
                continue
            
            # Dùng params để chống lỗi
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
