import os
import json
import httpx
import re
from fastapi import FastAPI, Query
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# Cấu hình phục vụ file tĩnh
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

class ExtractRequest(BaseModel):
    fileBase64: Optional[str] = None
    mimeType: Optional[str] = None
    rawText: Optional[str] = None

@app.post("/api/extract")
async def extract_text(req: ExtractRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return JSONResponse(status_code=500, content={"error": "Chưa cấu hình GEMINI_API_KEY"})

    url = f"https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-flash:generateContent?key={api_key}"

    # Prompt bắt buộc trả về JSON
    prompt = "Trích xuất văn bản từ ảnh và trả về mảng JSON. Định dạng: [{\"visual\": \"...\", \"spoken\": \"...\"}]. Bắt buộc dùng LaTeX cho công thức toán."

    parts = []
    if req.fileBase64:
        # Xóa tiền tố 'data:image/...;base64,' nếu có
        clean_b64 = re.sub(r'^data:[a-zA-Z0-9/+]+;base64,', '', req.fileBase64)
        parts.append({"inlineData": {"mimeType": req.mimeType or "image/jpeg", "data": clean_b64}})
    if req.rawText:
        parts.append({"text": req.rawText})
    parts.append({"text": prompt})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"responseMimeType": "application/json"},
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=60.0)
            if response.status_code != 200:
                return JSONResponse(status_code=response.status_code, content={"error": response.text})
            
            data = response.json()
            
            # Kiểm tra an toàn trước khi lấy dữ liệu
            candidates = data.get("candidates", [])
            if not candidates or "content" not in candidates[0]:
                return JSONResponse(status_code=400, content={"error": "AI từ chối trả lời (Safety Filter)."})

            text = candidates[0]["content"]["parts"][0]["text"]
            clean_text = text.replace("```json", "").replace("```", "").strip()
            
            return {"result": json.loads(clean_text, strict=False)}
            
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Lỗi hệ thống: {str(e)}"})

@app.get("/api/tts")
async def get_tts(text: str = Query(...)):
    target_url = f"https://translate.googleapis.com/translate_tts?client=gtx&ie=UTF-8&tl=vi&q={text}"
    async with httpx.AsyncClient() as client:
        r = await client.get(target_url, headers={"User-Agent": "Mozilla/5.0"})
        return StreamingResponse(r.iter_bytes(), media_type="audio/mpeg")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=int(os.getenv("PORT", 8000)))
