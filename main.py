import os
import json
from io import BytesIO
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, Response
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import os
import google.generativeai as genai

# Dòng này sẽ tự động lấy Key từ Render thay vì dán cứng vào code
api_key = os.getenv("GEMINI_API_KEY") 
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
# 1. API: XỬ LÝ OCR
# -------------------------------------------------------------------
class OCRRequest(BaseModel):
    file_base64: Optional[str] = None
    mime_type: Optional[str] = None
    raw_text: Optional[str] = None

@app.post("/api/ocr")
async def process_ocr(req: OCRRequest):
    try:
        model = genai.GenerativeModel("gemini-2.5-flash")
        
        prompt = """Bạn là trợ lý AI xử lý tài liệu. Trích xuất toàn bộ văn bản và trả về DUY NHẤT một mảng JSON.
Mỗi phần tử trong mảng JSON phải là MỘT CÂU NGẮN (tối đa 150 ký tự). Nếu câu gốc quá dài, HÃY TỰ ĐỘNG CẮT NGẮT thành nhiều phần tử liên tiếp nhau.
QUY TẮC BẮT BUỘC:
- "visual": Dùng mã LaTeX bọc trong $$...$$ (đứng một mình) hoặc \\( ... \\) (trong dòng) cho TẤT CẢ công thức. Giữ nguyên khoảng trắng và xuống dòng.
- "spoken": Dịch công thức sang CHỮ TIẾNG VIỆT thuần túy để phát âm (vd: "x bình phương").
- Tuyệt đối không thêm văn bản ngoài mảng JSON."""

        parts = []
        if req.file_base64 and req.mime_type:
            parts.append({"mime_type": req.mime_type, "data": req.file_base64})
        if req.raw_text:
            parts.append(f"Dữ liệu gốc:\n{req.raw_text}")
        parts.append(prompt)

        response = model.generate_content(parts, generation_config={"temperature": 0.1})
        text = response.text.replace("```json", "").replace("```", "").strip()
        
        return json.loads(text)
    
    except Exception as e:
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
# 3. API: GHÉP AUDIO HÀNG LOẠT
# -------------------------------------------------------------------
class BulkTTSRequest(BaseModel):
    texts: List[str]
    lang: str = "vi"

@app.post("/api/tts/bulk")
async def bulk_tts(req: BulkTTSRequest):
    try:
        combined_mp3 = BytesIO()
        for text in req.texts:
            if text.strip():
                tts = gTTS(text=text, lang=req.lang)
                tts.write_to_fp(combined_mp3)
                
        combined_mp3.seek(0)
        return Response(content=combined_mp3.read(), media_type="audio/mpeg", headers={
            "Content-Disposition": "attachment; filename=audiobook.mp3"
        })
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
