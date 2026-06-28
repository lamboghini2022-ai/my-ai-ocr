import os
import json
import httpx
import asyncio
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# 1. Cấu hình phục vụ các file tĩnh
if not os.path.exists("static"):
    os.makedirs("static")
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

# 2. Khai báo cấu trúc dữ liệu
class ExtractRequest(BaseModel):
    fileBase64: Optional[str] = None
    mimeType: Optional[str] = None
    rawText: Optional[str] = None

class BulkTTSRequest(BaseModel):
    texts: list[str]
    lang: str = "vi"

# ==========================================
# API 1: XỬ LÝ OCR & TRÍCH XUẤT ĐỀ TOÁN QUA GEMINI
# ==========================================
@app.post("/api/ocr")
async def extract_text(req: ExtractRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return JSONResponse(
            status_code=500, 
            content={"error": "Chưa cấu hình biến môi trường GEMINI_API_KEY trên Render."}
        )

    model_name = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    prompt = (
        "Bạn là AI trích xuất tài liệu OCR. Hãy trích xuất toàn bộ văn bản và trả về DUY NHẤT một mảng JSON.\n"
        "Mỗi phần tử là một câu, có định dạng: {\"visual\": \"...\", \"spoken\": \"...\"}.\n"
        "LƯU Ý QUAN TRỌNG CHO ĐỀ TOÁN:\n"
        "- Dùng mã LaTeX cho công thức toán học.\n"
        "- BẮT BUỘC: Mọi dấu gạch chéo ngược (\\) trong mã LaTeX phải được escape bằng 2 dấu gạch chéo (\\\\) để JSON hợp lệ. "
        "Ví dụ: viết \\\\frac thay vì \\frac, viết \\\\lim thay vì \\lim."
    )

    parts = []
    if req.fileBase64 and req.mimeType:
        parts.append({"inlineData": {"mimeType": req.mimeType, "data": req.fileBase64}})
    if req.rawText:
        parts.append({"text": req.rawText})
    parts.append({"text": prompt})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json" 
        }
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=60.0)
            if response.status_code != 200:
                return JSONResponse(status_code=response.status_code, content={"error": f"Lỗi Gemini API: {response.text}"})
            
            data = response.json()
            raw_result = data["candidates"][0]["content"]["parts"][0]["text"].strip()
            clean_text = raw_result.replace("```json", "").replace("```", "").strip()
            
            # Xử lý an toàn các ký tự điều khiển Toán học
            try:
                parsed_json = json.loads(clean_text, strict=False)
            except json.JSONDecodeError:
                fixed_text = clean_text.replace('\\', '\\\\').replace('\\\\"', '\\"')
                parsed_json = json.loads(fixed_text, strict=False)
                
            return {"result": parsed_json}
            
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Lỗi hệ thống hoặc định dạng: {str(e)}"})

# ==========================================
# API 2: ĐỌC TỪNG CÂU (Audio Player)
# ==========================================
@app.get("/api/tts")
async def get_tts(text: str = Query(...), lang: str = "vi"):
    target_url = f"https://translate.googleapis.com/translate_tts?client=gtx&ie=UTF-8&tl={lang}&q={text}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    async def stream_audio():
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", target_url, headers=headers) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk

    return StreamingResponse(stream_audio(), media_type="audio/mpeg")

# ==========================================
# API 3: GHÉP FILE MP3 TỔNG ĐỂ TẢI VỀ
# ==========================================
@app.post("/api/tts/bulk")
async def get_bulk_tts(req: BulkTTSRequest):
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
    
    async def generate_bulk_audio():
        async with httpx.AsyncClient() as client:
            for text in req.texts:
                if not text.strip(): 
                    continue
                target_url = f"https://translate.googleapis.com/translate_tts?client=gtx&ie=UTF-8&tl={req.lang}&q={text}"
                try:
                    response = await client.get(target_url, headers=headers)
                    if response.status_code == 200:
                        yield response.content 
                    await asyncio.sleep(1) # Nghỉ 1s tránh bị Google chặn
                except Exception:
                    continue

    return StreamingResponse(generate_bulk_audio(), media_type="audio/mpeg")

# Chạy Server cục bộ
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
