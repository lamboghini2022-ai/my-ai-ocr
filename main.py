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
# Vì giao diện HTML chạy độc lập (Local hoặc Host khác như Vercel/Netlify),
# Backend trên Render bắt buộc phải cho phép Cross-Origin để không bị chặn kết nối.
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các Origin truy cập. Bạn có thể thay đổi thành domain cụ thể nếu muốn bảo mật hơn.
    allow_credentials=True,
    allow_methods=["*"],  # Cho phép mọi phương thức (GET, POST, OPTIONS,...)
    allow_headers=["*"],  # Cho phép mọi Headers gửi lên
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

    # BẢN VÁ PROMPT TỐI ƯU: Ép AI chia nhỏ câu để chống lỗi Recitation (Bản quyền văn bản dài của Google)
     prompt = (
        "Bạn là AI trích xuất tài liệu OCR. Hãy trích xuất toàn bộ văn bản và trả về DUY NHẤT một mảng JSON.\n"
        "Mỗi phần tử là một câu, có định dạng: {\"visual\": \"...\", \"spoken\": \"...\"}.\n"
        "LƯU Ý QUAN TRỌNG CHO ĐỀ TOÁN:\n"
        "- Dùng mã LaTeX cho công thức toán học.\n"
        "- BẮT BUỘC: Mọi dấu gạch chéo ngược (\\) trong mã LaTeX phải được escape bằng 2 dấu gạch chéo (\\\\) để JSON hợp lệ. "
        "Ví dụ: viết \\\\frac thay vì \\frac, viết \\\\lim thay vì \\lim.\n"
        "- ĐẶC BIỆT: NẾU GẶP CÁC DÒNG DẤU CHẤM HOẶC GẠCH NGANG DÀI (ví dụ: ........ hoặc _______) dùng để điền đáp án, hãy rút gọn chúng lại thành 3 dấu chấm '...'. Tuyệt đối không in ra hàng ngàn dấu chấm."
        "Mỗi phần tử có định dạng: {\"visual\": \"...\", \"spoken\": \"...\"}.\n\n"
        "🚨 LƯU Ý TỐI QUAN TRỌNG (CHỐNG LỖI RECITATION BẢN QUYỀN):\n"
        "- API sẽ tự động khóa luồng nếu bạn in ra một đoạn văn hoặc bài thơ dài liên tục.\n"
        "- ĐỂ LÁCH LỖI: BẮT BUỘC băm nhỏ văn bản đến mức tối đa! Tách TỪNG DÒNG chữ ngắn trên ảnh thành MỘT phần tử JSON riêng biệt.\n"
        "- Tuyệt đối KHÔNG gộp nhiều dòng thơ/văn vào chung một giá trị string (ví dụ: cấm dùng \\n để nối dòng). Phải dùng cấu trúc JSON để ngắt mạch văn bản liên tục.\n\n"
        "LƯU Ý VỀ ĐỊNH DẠNG KHÁC:\n"
        "- Dùng mã LaTeX cho toán học. BẮT BUỘC: Mọi dấu gạch chéo ngược (\\) phải được escape bằng 2 dấu (\\\\). Ví dụ: \\\\frac thay vì \\frac.\n"
        "- Rút gọn các dòng dấu chấm hoặc gạch ngang dài để điền đáp án (ví dụ: ........) thành 3 dấu chấm '...'."
    )

    parts = []
    if req.fileBase64 and req.mimeType:
        clean_b64 = re.sub(r'^data:[a-zA-Z0-9/+]+;base64,', '', req.fileBase64)
        parts.append({"inlineData": {"mimeType": req.mimeType, "data": clean_b64}})
        print("[INFO] Đã chuẩn hóa chuỗi Base64 tài liệu đính kèm.")
        
    if req.rawText:
        parts.append({"text": req.rawText})
    
    parts.append({"text": prompt})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json",
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
# 2. API PROXY GOOGLE TTS (ĐỌC TỪNG CÂU VỚI CORS)
# ==========================================
@app.get("/api/tts")
async def get_tts(text: str = Query(...), lang: str = "vi"):
    target_url = f"https://translate.googleapis.com/translate_tts?client=gtx&ie=UTF-8&tl={lang}&q={text}"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    async def stream_audio():
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", target_url, headers=headers) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk

    return StreamingResponse(stream_audio(), media_type="audio/mpeg")

# ==========================================
# 3. API GHÉP NỐI MP3 HÀNG LOẠT (TẢI TRỌN BỘ OFFLINE)
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
            target_url = f"https://translate.googleapis.com/translate_tts?client=gtx&ie=UTF-8&tl={req.lang}&q={text}"
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
