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
# CẤU HÌNH CORS MIDDLEWARE (Quan trọng cho Render)
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép HTML local truy cập vào Render
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root_endpoint():
    return JSONResponse(content={
        "status": "online",
        "message": "Backend OCR Reader (Sync với May_Doc_Sach.html) đang chạy!",
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
    print("\n========== BẮT ĐẦU XỬ LÝ YÊU CẦU OCR ==========")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[LỖI] Thiếu cấu hình GEMINI_API_KEY")
        return JSONResponse(
            status_code=500, 
            content={"error": "Chưa cấu hình biến môi trường GEMINI_API_KEY trên Render."}
        )

    model_name = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    PROMPT_TEXT = r"""
Bạn là một Hệ thống Trích xuất Dữ liệu OCR chuyên nghiệp. Nhiệm vụ của bạn là số hóa nội dung và BẮT BUỘC trả về định dạng JSON là một MẢNG (ARRAY) CHỨA CÁC OBJECT. 

Cấu trúc JSON bắt buộc:
[
  {
    "visual": "Đề kiểm tra môn Vật Lý\n\nCâu 1: Tính vận tốc...",
    "spoken": "Đề kiểm tra môn Vật Lý. Câu một: Tính vận tốc..."
  }
]

📐 QUY TẮC CHO "visual" (GIAO DIỆN HIỂN THỊ TRÊN MÀN HÌNH):
- KHÔNG DÙNG THẺ HTML (vì Frontend dùng textContent). Hãy dùng ký tự ngắt dòng `\n\n` để chia đoạn, tạo khoảng trắng giúp giao diện dễ nhìn.
- Mọi công thức Toán/Lý/Hóa BẮT BUỘC dùng mã LaTeX.
- Công thức trong dòng (Inline): Bọc bằng `$`. Ví dụ: `$v = s/t$`
- Công thức đứng riêng (Block): Bọc bằng `$$`. Ví dụ: `$$F = m \cdot a$$`
- LƯU Ý JSON: Phải nhân đôi dấu gạch chéo ngược cho lệnh LaTeX để không làm hỏng cú pháp JSON (Ví dụ: `\\frac{a}{b}`, `\\sqrt{x}`, `\\Delta`).

🚨 QUY TẮC CHO "spoken" (ĐỂ CHUYỂN THÀNH GIỌNG NÓI TTS):
- Chia nội dung thành các câu ngắn. Mỗi object trong mảng chỉ nên chứa 1-2 câu (khoảng 15-30 từ) để máy đọc không bị ngắt quãng.
- KHÔNG chứa mã LaTeX. Phải dịch công thức thành tiếng Việt (Ví dụ: "H hai O", "x bình phương cộng y bình phương", "căn bậc hai của x").
- Chỉ chứa chữ cái, số và dấu câu cơ bản (, . ! ?).
    """

    parts = []
    if req.fileBase64 and req.mimeType:
        clean_b64 = re.sub(r'^data:[a-zA-Z0-9/+]+;base64,', '', req.fileBase64)
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
            "temperature": 0.2, 
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json"
        }
    }

    async with httpx.AsyncClient() as client:
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
# 2. API PROXY GOOGLE TTS (ĐỌC TỪNG CÂU - ĐÃ SỬA LỖI GIẢI MÃ)
# ==========================================
@app.get("/api/tts")
async def get_tts(text: str = Query(...), lang: str = "vi"):
    if not text or not text.strip():
        return JSONResponse(status_code=400, content={"error": "Nội dung văn bản trống."})

    # Cắt chuỗi tối đa 200 ký tự để không làm sập API Google Translate TTS miễn phí
    safe_text = text[:200].strip()
    
    base_url = "https://translate.googleapis.com/translate_tts"
    # Dùng `params` giúp tự động encode các ký tự đặc biệt, xuống dòng, khoảng trắng thành chuẩn URL
    params = {
        "client": "gtx",
        "ie": "UTF-8",
        "tl": lang,
        "q": safe_text
    }
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    async with httpx.AsyncClient() as client:
        try:
            # Chuyển sang lấy toàn bộ content thay vì dùng `.stream()` mù quáng nhằm kiểm tra mã trạng thái
            resp = await client.get(base_url, params=params, headers=headers, timeout=15.0)
            
            if resp.status_code != 200:
                print(f"[TTS ERROR] Google TTS từ chối (Mã {resp.status_code}). Có thể do bị rate-limit.")
                return JSONResponse(
                    status_code=resp.status_code, 
                    content={"error": "Google TTS phản hồi lỗi. Vui lòng thử lại sau."}
                )
                
            # Trả về StreamingResponse an toàn từ BytesIO
            return StreamingResponse(io.BytesIO(resp.content), media_type="audio/mpeg")
            
        except Exception as e:
            print(f"[TTS CRITICAL LỖI]: {e}")
            return JSONResponse(status_code=500, content={"error": f"Không thể kết nối đến server TTS: {str(e)}"})

# ==========================================
# 3. API GHÉP NỐI MP3 HÀNG LOẠT (ĐÃ SỬA LỖI MÃ HÓA)
# ==========================================
class BulkTTSRequest(BaseModel):
    texts: list[str]
    lang: str = "vi"

@app.post("/api/tts/bulk")
async def bulk_tts(req: BulkTTSRequest):
    print(f"\n========== TỔNG HỢP AUDIO TỔNG ({len(req.texts)} phần tử) ==========")
    base_url = "https://translate.googleapis.com/translate_tts"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    combined_audio = bytearray()
    
    async with httpx.AsyncClient() as client:
        for text in req.texts:
            if not text or not text.strip():
                continue
            
            # Đảm bảo mỗi đoạn nhỏ gửi lên không vượt quá giới hạn ký tự
            safe_text = text[:200].strip()
            params = {
                "client": "gtx",
                "ie": "UTF-8",
                "tl": req.lang,
                "q": safe_text
            }
            try:
                resp = await client.get(base_url, params=params, headers=headers, timeout=15.0)
                if resp.status_code == 200:
                    combined_audio.extend(resp.content)
                else:
                    print(f"[WARNING] Bỏ qua đoạn do lỗi mã {resp.status_code} từ Google: '{safe_text[:30]}...'")
            except Exception as e:
                print(f"[WARNING] Bỏ qua đoạn âm thanh do lỗi kết nối: {e}")
                
    if not combined_audio:
        return JSONResponse(status_code=500, content={"error": "Không thể tải bất kỳ đoạn audio nào từ server TTS."})
        
    return StreamingResponse(
        io.BytesIO(combined_audio), 
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment; filename=Merged_OCR_AudioBook.mp3"}
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
