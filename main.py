import os
import json
import httpx
import re
import io
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional

# Thử tải biến môi trường từ file .env nếu chạy local
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="SaaS OCR Reader Backend")

# ==========================================
# CẤU HÌNH QUAN TRỌNG: CORS MIDDLEWARE
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], 
    allow_credentials=True,
    allow_methods=["*"], 
    allow_headers=["*"], 
)

# Tạo thư mục static dự phòng nếu chạy chế độ monolithic
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root_endpoint():
    return JSONResponse(content={
        "status": "online",
        "message": "Backend OCR Reader đang chạy thành công trên Render!",
        "endpoints": {
            "OCR & Extract": "/api/extract [POST]",
            "TTS Stream": "/api/tts [GET]",
            "TTS Bulk Download": "/api/tts/bulk [POST]"
        }
    })

class ExtractRequest(BaseModel):
    fileBase64: Optional[str] = None
    mimeType: Optional[str] = None
    rawText: Optional[str] = None

# ==========================================
# 1. API XỬ LÝ OCR & TRÍCH XUẤT QUA GEMINI
# ==========================================
@app.post("/api/extract") 
async def extract_text(req: ExtractRequest):
    print("\n========== BẮT ĐẦU XỬ LÝ YÊU CẦU OCR LÀM MỚI ==========")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[LỖI] Thiếu cấu hình GEMINI_API_KEY")
        return JSONResponse(
            status_code=500, 
            content={"error": "Chưa cấu hình biến môi trường GEMINI_API_KEY trên Render."}
        )

    model_name = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    # BẢN VÁ PROMPT TỐI ƯU: Xuất JSON chuẩn cho Frontend, tự động áp dụng CSS làm đẹp bố cục
    PROMPT_TEXT = r"""
Bạn là Hệ thống Universal OCR kiêm Chuyên gia Phục dựng Bố cục. Nhiệm vụ của bạn là số hóa nội dung, GIỮ NGUYÊN BỐ CỤC, xuất ra định dạng JSON MẢNG (Array).

🎨 QUY TẮC 1: PHỤC DỰNG BỐ CỤC ĐẸP MẮT (Cho mục `visual`):
- Hãy chia văn bản thành các đối tượng JSON theo từng đoạn văn/khối để giao diện hiển thị thoáng đãng.
- Tiêu đề/Quốc hiệu: Bọc trong `<div style="text-align: center; font-size: 1.15em; font-weight: bold; color: #FF7A00; margin-bottom: 12px;">...</div>`.
- Ngày tháng: Bọc trong `<div style="text-align: right; font-style: italic; color: #8B8D91; margin-bottom: 10px;">...</div>`.
- Đoạn văn bình thường: Bọc trong `<div style="margin-bottom: 10px; line-height: 1.7; text-align: justify;">...</div>`.
- Bảng biểu: Bọc trong `<div style="overflow-x: auto; margin: 15px 0;"><table style="width: 100%; border-collapse: collapse; border: 1px solid rgba(255,122,0,0.3);">...</table></div>`. Các thẻ `<th>`, `<td>` cần thêm thuộc tính CSS `border: 1px solid rgba(255,122,0,0.3); padding: 8px;`.

🚨 QUY TẮC 2: CHIẾN LƯỢC VƯỢT LỖI RECITATION:
- BẮT BUỘC chèn thẻ HTML rỗng `<span></span>` sau MỖI 3 ĐẾN 5 TỪ liên tiếp TRONG VĂN BẢN THƯỜNG. 
- KHÔNG chèn thẻ `<span></span>` vào mã Toán LaTeX, HTML cấu trúc, hay mục `spoken`.

📐 QUY TẮC 3: XỬ LÝ TOÁN HỌC:
- Mọi công thức/biểu thức toán học bắt buộc phải dùng mã LaTeX nguyên bản.
- Công thức cùng dòng: Bọc trong ký hiệu `$ ... $`.
- Công thức đứng thành dòng độc lập: Bọc trong `<div style="text-align: center; padding: 12px 0; overflow-x: auto; font-size: 1.1em; color: #FFF;">$$...$$</div>`.

❌ QUY TẮC 4: ĐỊNH DẠNG ĐẦU RA BẮT BUỘC (JSON ARRAY):
Chỉ trả về MỘT MẢNG JSON HỢP LỆ (KHÔNG bọc trong markdown ```json). Cấu trúc:
[
  {
    "visual": "Mã HTML/CSS phục dựng kèm <span></span> và $ / $$ LaTeX",
    "spoken": "Văn bản thuần túy để đọc thành tiếng (Không chứa HTML hay LaTeX)"
  }
]
"""

    parts = []
    if req.fileBase64 and req.mimeType:
        clean_b64 = re.sub(r'^data:[a-zA-Z0-9/+]+;base64,', '', req.fileBase64)
        parts.append({"inlineData": {"mimeType": req.mimeType, "data": clean_b64}})
        
    if req.rawText:
        parts.append({"text": req.rawText})
    
    # SỬA LỖI: Truyền đúng biến PROMPT_TEXT thay vì prompt
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
            "temperature": 0.4, 
            "maxOutputTokens": 8192
        }
    }

    async with httpx.AsyncClient() as client:
        try:
            print("[INFO] Đang chuyển tiếp gói tin đến Google Gemini API...")
            response = await client.post(url, json=payload, timeout=60.0)
            
            if response.status_code != 200:
                print(f"[API ERROR] Google API trả về mã lỗi: {response.status_code} - {response.text}")
                return JSONResponse(status_code=response.status_code, content={"error": f"Lỗi phản hồi từ Gemini: {response.text}"})
            
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                prompt_feedback = data.get("promptFeedback", {})
                print("[SAFETY BLOCKED] Yêu cầu bị từ chối từ bộ lọc an toàn đầu vào:", prompt_feedback)
                return JSONResponse(status_code=400, content={"error": f"Yêu cầu bị chặn bởi chính sách an toàn đầu vào của Google. {prompt_feedback}"})
                
            candidate = candidates[0]
            if "content" not in candidate:
                finish_reason = candidate.get("finishReason", "Không xác định")
                print("[SAFETY FILTER TRIGGERED] Bị chặn ở luồng đầu ra. Lý do:", finish_reason)
                return JSONResponse(status_code=400, content={"error": f"AI không thể xuất kết quả do vi phạm bộ lọc đầu ra ({finish_reason})."})

            raw_result = candidate["content"]["parts"][0]["text"].strip()
            clean_text = raw_result.replace("```json", "").replace("```", "").strip()
            
            try:
                parsed_json = json.loads(clean_text, strict=False)
                print("[SUCCESS] Trích xuất thành công và biên dịch định dạng JSON sạch.")
                return {"result": parsed_json}
            except json.JSONDecodeError:
                print("[REPAIRING] Lỗi cấu trúc JSON thô, đang áp dụng bộ vá chuỗi tự động...")
                try:
                    fixed_text = clean_text.replace('\\', '\\\\').replace('\\\\"', '\\"')
                    parsed_json = json.loads(fixed_text, strict=False)
                    print("[SUCCESS] Đã sửa lỗi escape ký tự và biên dịch JSON thành công.")
                    return {"result": parsed_json}
                except Exception as parse_err:
                    print(f"[CRITICAL ERROR] Không thể cứu vãn cấu trúc JSON: {parse_err}")
                    return JSONResponse(status_code=500, content={"error": "Cấu trúc phản hồi từ AI bị lỗi định dạng nghiêm trọng.", "raw": clean_text})
            
        except httpx.ReadTimeout:
            print("[TIMEOUT] Google API không phản hồi trong 60 giây.")
            return JSONResponse(status_code=504, content={"error": "Quá thời gian phản hồi (60 giây) khi kết nối với hệ thống Google API."})
        except Exception as e:
            import traceback
            traceback.print_exc()
            return JSONResponse(status_code=500, content={"error": f"Lỗi xử lý hệ thống nội bộ: {str(e)}"})

# ==========================================
# 2. API PROXY GOOGLE TTS
# ==========================================
@app.get("/api/tts")
async def get_tts(text: str = Query(...), lang: str = "vi"):
    target_url = f"[https://translate.googleapis.com/translate_tts?client=gtx&ie=UTF-8&tl=](https://translate.googleapis.com/translate_tts?client=gtx&ie=UTF-8&tl=){lang}&q={text}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    async def stream_audio():
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", target_url, headers=headers) as r:
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
    print(f"\n========== BẮT ĐẦU TỔNG HỢP VÀ GHÉP AUDIO TỔNG ({len(req.texts)} phần tử) ==========")
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    combined_audio = bytearray()
    
    async with httpx.AsyncClient() as client:
        for text in req.texts:
            if not text or not text.strip():
                continue
            target_url = f"[https://translate.googleapis.com/translate_tts?client=gtx&ie=UTF-8&tl=](https://translate.googleapis.com/translate_tts?client=gtx&ie=UTF-8&tl=){req.lang}&q={text}"
            try:
                resp = await client.get(target_url, headers=headers, timeout=15.0)
                if resp.status_code == 200:
                    combined_audio.extend(resp.content)
            except Exception as e:
                print(f"[WARNING] Bỏ qua lỗi tải đoạn âm thanh cho câu: '{text[:20]}...'. Chi tiết: {e}")
                
    if not combined_audio:
        return JSONResponse(status_code=500, content={"error": "Toàn bộ chuỗi audio tải về trống hoặc không thể kết nối tới máy chủ TTS."})
        
    return StreamingResponse(
        io.BytesIO(combined_audio), 
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment; filename=Merged_OCR_AudioBook.mp3"}
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
