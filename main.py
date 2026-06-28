import os
import json
import base64
from io import BytesIO
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import google.generativeai as genai
from gtts import gTTS

# =================================================================
# CẤU HÌNH API KEY TỪ BIẾN MÔI TRƯỜNG CỦA RENDER
# =================================================================
api_key = os.getenv("GEMINI_API_KEY")
if api_key:
    genai.configure(api_key=api_key)

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

os.makedirs("static", exist_ok=True)
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/", response_class=HTMLResponse)
async def read_index():
    with open("static/index.html", "r", encoding="utf-8") as f:
        return f.read()

# -------------------------------------------------------------------
# 1. API: XỬ LÝ OCR (ĐÃ SỬA FULL LỖI)
# -------------------------------------------------------------------
class OCRRequest(BaseModel):
    file_base64: Optional[str] = None
    mime_type: Optional[str] = None
    raw_text: Optional[str] = None

@app.post("/api/ocr")
async def process_ocr(req: OCRRequest):
    if not api_key:
        raise HTTPException(status_code=500, detail="Chưa cấu hình GEMINI_API_KEY trên server.")
        
    try:
        model = genai.GenerativeModel("gemini-1.5-flash")
        
        prompt = """Bạn là trợ lý AI xử lý tài liệu. Trích xuất toàn bộ văn bản và trả về DUY NHẤT một mảng JSON.
Mỗi phần tử trong mảng JSON phải là MỘT CÂU NGẮN (tối đa 150 ký tự). Nếu câu gốc quá dài, HÃY TỰ ĐỘNG CẮT NGẮT thành nhiều phần tử liên tiếp nhau.
QUY TẮC BẮT BUỘC:
- "visual": Dùng mã LaTeX bọc trong $$...$$ (đứng một mình) hoặc \\( ... \\) (trong dòng) cho TẤT CẢ công thức.
- "spoken": Dịch công thức sang CHỮ TIẾNG VIỆT thuần túy để phát âm (vd: "x bình phương").
- ĐẢM BẢO ĐỊNH DẠNG JSON HỢP LỆ (escape đúng các ký tự backslash của LaTeX)."""

        parts = []
        if req.file_base64 and req.mime_type:
            # SỬA LỖI ĐỌC ẢNH: Chuyển Base64 thành Bytes
            b64_str = req.file_base64
            if "," in b64_str:
                b64_str = b64_str.split(",")[1]
            file_bytes = base64.b64decode(b64_str)
            parts.append({"mime_type": req.mime_type, "data": file_bytes})
            
        if req.raw_text:
            parts.append(f"Dữ liệu gốc:\n{req.raw_text}")
        parts.append(prompt)

        # SỬA LỖI SẬP JSON DO LATEX: Ép AI chỉ trả về JSON thuần chuẩn xác
        response = model.generate_content(
            parts, 
            generation_config={
                "temperature": 0.1,
                "response_mime_type": "application/json"
            }
        )
        
        text = response.text.strip()
        return json.loads(text)
    
    except Exception as e:
        print(f"Lỗi hệ thống OCR: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------
# 2. API: TẠO AUDIO TỪNG CÂU
# -------------------------------------------------------------------
@app.get("/api/tts")
async def single_tts(text: str, lang: str = "vi"):
    try:
        tts = gTTS(text=text, lang=lang)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        return Response(content=fp.read(), media_type="audio/mpeg")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# -------------------------------------------------------------------
# 3. API: GHÉP AUDIO HÀNG LOẠT (ĐÃ SỬA LỖI HỎNG FILE)
# -------------------------------------------------------------------
class BulkTTSRequest(BaseModel):
    texts: List[str]
    lang: str = "vi"

@app.post("/api/tts/bulk")
async def bulk_tts(req: BulkTTSRequest):
    try:
        # SỬA LỖI: Gộp toàn bộ văn bản lại thành 1 cục trước khi đưa cho gTTS
        combined_text = " ".join([text.strip() for text in req.texts if text.strip()])
        if not combined_text:
            raise HTTPException(status_code=400, detail="Không có nội dung chữ để chuyển thành âm thanh.")

        tts = gTTS(text=combined_text, lang=req.lang)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        return Response(content=fp.read(), media_type="audio/mpeg", headers={
            "Content-Disposition": "attachment; filename=audiobook.mp3"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
