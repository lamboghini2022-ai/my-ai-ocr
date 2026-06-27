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
from gtts import gTTS  # <--- Đã thêm thư viện gTTS bị thiếu

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
        
        # Đã siết chặt Prompt để cấm AI tự bịa đặt (Ảo giác)
        prompt = """Bạn là hệ thống OCR trích xuất văn bản và công thức toán học. Trả về kết quả dưới dạng DUY NHẤT một mảng JSON.
Mỗi phần tử trong mảng JSON là MỘT CÂU NGẮN (tối đa 150 ký tự).
QUY TẮC TỐI THƯỢNG (BẮT BUỘC TUÂN THỦ):
1. CHỈ TRÍCH XUẤT CHÍNH XÁC NHỮNG GÌ NHÌN THẤY TRONG ẢNH. Tuyệt đối không tự bịa đặt, không thêm thắt văn bản, không tự tạo ví dụ mẫu. Nếu ảnh chỉ có 1 chữ/1 số, chỉ trả về đúng chữ/số đó.
2. Nếu ảnh trống hoặc không có chữ, trả về mảng rỗng [].
3. "visual": Dùng mã LaTeX bọc trong $$...$$ cho công thức.
4. "spoken": Dịch công thức sang chữ Tiếng Việt thuần túy để phát âm."""

        parts = []
        if req.file_base64 and req.mime_type:
            # Đã thêm lệnh giải mã Base64 sang Bytes để AI đọc được ảnh
            file_bytes = base64.b64decode(req.file_base64)
            parts.append({"mime_type": req.mime_type, "data": file_bytes})
            
        if req.raw_text:
            parts.append(f"Dữ liệu gốc:\n{req.raw_text}")
        parts.append(prompt)

        # Ép AI luôn trả về JSON thuần
        response = model.generate_content(
            parts, 
            generation_config={"temperature": 0.1, "response_mime_type": "application/json"}
        )
        
        text = response.text.replace("```json", "").replace("```", "").strip()
        return json.loads(text)
    
    except Exception as e:
        print(f"Lỗi Server OCR: {e}")
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
# 3. API: GHÉP AUDIO HÀNG LOẠT (TẢI MP3)
# -------------------------------------------------------------------
class BulkTTSRequest(BaseModel):
    texts: List[str]
    lang: str = "vi"

@app.post("/api/tts/bulk")
async def bulk_tts(req: BulkTTSRequest):
    try:
        # Đã sửa lỗi tạo file MP3: Gộp chữ lại trước rồi mới xuất ra âm thanh
        combined_text = " ".join([text.strip() for text in req.texts if text.strip()])
        
        if not combined_text:
            raise HTTPException(status_code=400, detail="Không có nội dung để đọc")

        tts = gTTS(text=combined_text, lang=req.lang)
        fp = BytesIO()
        tts.write_to_fp(fp)
        fp.seek(0)
        
        return Response(content=fp.read(), media_type="audio/mpeg", headers={
            "Content-Disposition": "attachment; filename=audiobook.mp3"
        })
    except Exception as e:
        print(f"Lỗi Server MP3: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 10000))
    uvicorn.run(app, host="0.0.0.0", port=port)
