import os
import json
import httpx
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

app = FastAPI()

# Cấu hình phục vụ các file tĩnh nằm trong thư mục 'static'
app.mount("/static", StaticFiles(directory="static"), name="static")

# Tuyến đường mặc định khi truy cập vào trang web
@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

# Khai báo cấu trúc dữ liệu gửi lên từ Frontend
class ExtractRequest(BaseModel):
    fileBase64: Optional[str] = None
    mimeType: Optional[str] = None
    rawText: Optional[str] = None

# ==========================================
# 1. API XỬ LÝ OCR & TRÍCH XUẤT QUA GEMINI
# ==========================================
@app.post("/api/extract") # Chú ý: Đổi thành /api/ocr nếu frontend của bạn đang gọi đường dẫn đó
async def extract_text(req: ExtractRequest):
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return JSONResponse(
            status_code=500, 
            content={"error": "Chưa cấu hình biến môi trường GEMINI_API_KEY trên Render."}
        )

    model_name = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    # Đã thêm lệnh bắt buộc AI phải "escape" dấu gạch chéo cho các đề toán
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

    # [BẢN VÁ LỖI]: Bắt buộc thêm block generationConfig để Gemini trả về JSON chuẩn 100%
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
            
            # Làm sạch nếu AI lỡ tay bọc Markdown
            clean_text = raw_result.replace("```json", "").replace("```", "").strip()
            
            # [BẢN VÁ LỖI]: Thêm strict=False để Python bỏ qua các lỗi ký tự điều khiển ẩn (như dấu \n, \t)
            try:
                parsed_json = json.loads(clean_text, strict=False)
            except json.JSONDecodeError:
                # Lưới bảo vệ cuối cùng: Tự động sửa lỗi backslash toán học bằng Python nếu AI vẫn làm sai
                fixed_text = clean_text.replace('\\', '\\\\').replace('\\\\"', '\\"')
                parsed_json = json.loads(fixed_text, strict=False)
                
            return {"result": parsed_json}
            
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Lỗi hệ thống hoặc định dạng: {str(e)}"})

# ==========================================
# 2. API PROXY GOOGLE TTS (Sửa triệt để lỗi CORS)
# ==========================================
@app.get("/api/tts")
async def get_tts(text: str = Query(...), lang: str = "vi"):
    target_url = f"https://translate.googleapis.com/translate_tts?client=gtx&ie=UTF-8&tl={lang}&q={text}"
    # Giả lập User-Agent trình duyệt để Google không chặn IP của Render
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    async def stream_audio():
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", target_url, headers=headers) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk

    return StreamingResponse(stream_audio(), media_type="audio/mpeg")

# Chạy Server cục bộ khi test máy cá nhân
if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
