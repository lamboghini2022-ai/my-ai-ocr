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
@app.post("/api/extract")
async def extract_text(req: ExtractRequest):
    # Lấy API Key từ biến môi trường (Environment Variable) trên Render
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return JSONResponse(
            status_code=500, 
            content={"error": "Server chưa cấu hình biến môi trường GEMINI_API_KEY."}
        )

    model_name = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    prompt = (
        "Bạn là trợ lý AI xử lý tài liệu. Trích xuất toàn bộ văn bản và trả về DUY NHẤT một mảng JSON.\n"
        "CHÚ Ý QUAN TRỌNG ĐỂ KHÔNG BỊ LỖI AUDIO: Mỗi phần tử trong mảng JSON phải là MỘT CÂU NGẮN (tối đa 150 ký tự). "
        "Nếu câu gốc quá dài, HÃY TỰ ĐỘNG CẮT NGẮT thành nhiều phần tử liên tiếp nhau.\n\n"
        "QUY TẮC BẮT BUỘC CHO MẢNG JSON:\n"
        "- \"visual\": Dùng mã LaTeX bọc trong $$...$$ (đứng một mình) hoặc \\( ... \\) (trong dòng) cho TẤT CẢ công thức Toán/Hóa học để MathJax có thể vẽ. Giữ lại nguyên vẹn khoảng trắng (space) ở đầu dòng và ký tự xuống dòng (\\n) ở cuối để dựng layout như bản gốc.\n"
        "- \"spoken\": Dịch công thức sang CHỮ TIẾNG VIỆT thuần túy để máy tính phát âm (vd: \"x bình phương\", \"H hai O\").\n"
        "- Tuyệt đối không thêm văn bản ngoài mảng JSON."
    )

    parts = []
    if req.fileBase64 and req.mimeType:
        parts.append({"inlineData": {"mimeType": req.mimeType, "data": req.fileBase64}})
    if req.rawText:
        parts.append({"text": f"Dữ liệu gốc:\n{req.rawText}"})
    parts.append({"text": prompt})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {"temperature": 0.1}
    }

    async with httpx.AsyncClient() as client:
        try:
            response = await client.post(url, json=payload, timeout=60.0)
            if response.status_code != 200:
                return JSONResponse(status_code=response.status_code, content={"error": response.text})
            
            data = response.json()
            if "candidates" not in data or not data["candidates"]:
                return JSONResponse(status_code=500, content={"error": "AI không phản hồi kết quả hợp lệ."})
            
            result_text = data["candidates"][0]["content"]["parts"][0]["text"]
            # Làm sạch chuỗi bao bọc markdown nếu có
            result_text = result_text.replace("```json", "").replace("```", "").strip()
            
            parsed_json = json.loads(result_text)
            return {"result": parsed_json}
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": str(e)})

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
