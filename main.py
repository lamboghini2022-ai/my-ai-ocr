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
from typing import Optional, List

# Thử tải biến môi trường từ file .env nếu chạy local
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="SaaS OCR Reader Backend")

# ==========================================
# CẤU HÌNH CORS MIDDLEWARE
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
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
        "message": "Backend OCR Reader đang chạy!",
        "endpoints": {
            "OCR & Extract": "/api/extract [POST]",
            "TTS Stream": "/api/tts [GET]",
            "TTS Bulk Download": "/api/tts/bulk [POST]"
        }
    })

# Tạo model dữ liệu cho 1 ảnh
class ImageData(BaseModel):
    fileBase64: str
    mimeType: str

# Cập nhật Model để nhận 1 mảng gồm nhiều ảnh (tối đa 5)
class ExtractRequest(BaseModel):
    images: Optional[List[ImageData]] = None  # Cấu trúc mới hỗ trợ 5 ảnh
    
    # Vẫn giữ lại cấu trúc cũ để lỡ Frontend chưa kịp cập nhật thì code không bị sập
    fileBase64: Optional[str] = None
    mimeType: Optional[str] = None
    
    rawText: Optional[str] = None

# ==========================================
# 1. API XỬ LÝ OCR & TRÍCH XUẤT QUA GEMINI (TỐI ĐA 5 ẢNH)
# ==========================================
@app.post("/api/extract") 
async def extract_text(req: ExtractRequest):
    print("\n========== BẮT ĐẦU XỬ LÝ YÊU CẦU OCR ==========")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[LỖI] Thiếu cấu hình GEMINI_API_KEY")
        return JSONResponse(
            status_code=500, 
            content={"error": "Chưa cấu hình biến môi trường GEMINI_API_KEY."}
        )

    model_name = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    # PROMPT MỚI: Ép AI giữ nguyên bố cục không gian, khoảng trắng để tạo cảm giác căn lề
    PROMPT_TEXT = r"""
Bạn là một Hệ thống Trích xuất Dữ liệu OCR chuyên nghiệp. Nhiệm vụ của bạn là số hóa nội dung từ các bức ảnh được cung cấp và BẮT BUỘC trả về định dạng JSON là một MẢNG (ARRAY) CHỨA CÁC OBJECT. 

Cấu trúc JSON bắt buộc:
[
  {
    "visual": "    CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM\n          Độc lập - Tự do - Hạnh phúc\n\n          HỢP ĐỒNG MUA BÁN CĂN HỘ...",
    "spoken": "Cộng hòa xã hội chủ nghĩa Việt Nam. Độc lập Tự do Hạnh phúc. Hợp đồng mua bán căn hộ..."
  }
]

📐 QUY TẮC SỐNG CÒN CHO "visual" (ĐỂ BẢO TOÀN BỐ CỤC ĐẸP MẮT):
1. BẢO TOÀN KHOẢNG TRẮNG: Bạn PHẢI sử dụng các khoảng trắng (space) để đẩy các Tiêu đề (như HỢP ĐỒNG, CỘNG HÒA XÃ HỘI...) ra giữa dòng, giống hệt như cách chúng được căn giữa trong bản gốc.
2. BẢO TOÀN NGẮT DÒNG: Dùng `\n` hoặc `\n\n` chính xác theo từng đoạn, từng mục của văn bản, hợp đồng. Không được tự ý nối các dòng của hợp đồng lại với nhau.
3. KHÔNG DÙNG THẺ HTML. Chỉ dùng text thô kết hợp khoảng trắng và ngắt dòng.
4. Mọi công thức Toán/Lý/Hóa BẮT BUỘC dùng mã LaTeX (Ví dụ: `$v = s/t$`). Phải nhân đôi dấu gạch chéo ngược (`\\frac{a}{b}`).

🚨 QUY TẮC CHO "spoken" (ĐỂ CHUYỂN THÀNH GIỌNG NÓI TTS):
- Dịch hoàn toàn sang text thuần túy để máy đọc. Chia câu ngắn.
- KHÔNG chứa mã LaTeX.
    """

    parts = []
    
    # XỬ LÝ NHIỀU ẢNH (TỐI ĐA 5)
    image_count = 0
    if req.images and len(req.images) > 0:
        if len(req.images) > 5:
            return JSONResponse(status_code=400, content={"error": "Chỉ hỗ trợ xử lý tối đa 5 ảnh trong một lần."})
        
        for img in req.images:
            clean_b64 = re.sub(r'^data:[a-zA-Z0-9/+]+;base64,', '', img.fileBase64)
            parts.append({"inlineData": {"mimeType": img.mimeType, "data": clean_b64}})
            image_count += 1
            
    # HỖ TRỢ NGƯỢC CODE CŨ (Nếu Frontend gửi lên 1 ảnh theo cách cũ)
    elif req.fileBase64 and req.mimeType:
        clean_b64 = re.sub(r'^data:[a-zA-Z0-9/+]+;base64,', '', req.fileBase64)
        parts.append({"inlineData": {"mimeType": req.mimeType, "data": clean_b64}})
        image_count = 1
        
    if req.rawText:
        parts.append({"text": req.rawText})
    
    if image_count == 0 and not req.rawText:
         return JSONResponse(status_code=400, content={"error": "Không có dữ liệu ảnh hoặc văn bản nào được gửi lên."})
         
    print(f"[INFO] Đã tiếp nhận {image_count} ảnh để xử lý.")
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
            "temperature": 0.1,  # Hạ thấp temperature để AI bám sát định dạng gốc hơn
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json"
        }
    }

    async with httpx.AsyncClient() as client:
        try:
            print("[INFO] Đang chuyển tiếp gói tin đến Google Gemini API...")
            # Tăng timeout lên 120s vì xử lý 5 ảnh sẽ tốn thời gian hơn 1 ảnh
            response = await client.post(url, json=payload, timeout=120.0)
            
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
                print(f"[SUCCESS] Trích xuất thành công nội dung từ {image_count} ảnh.")
                return {"result": parsed_json}
            except json.JSONDecodeError as e:
                print(f"[CRITICAL ERROR] JSON lỗi định dạng: {e}")
                return JSONResponse(status_code=500, content={"error": "AI trả về chuỗi JSON không hợp lệ.", "raw": raw_result})
            
        except httpx.ReadTimeout:
            print("[TIMEOUT] Quá thời gian chờ (120 giây).")
            return JSONResponse(status_code=504, content={"error": "Quá thời gian phản hồi do ảnh quá nhiều hoặc quá nặng."})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Lỗi nội bộ: {str(e)}"})

# ==========================================
# 2. API PROXY GOOGLE TTS (ĐỌC TỪNG CÂU)
# ==========================================
@app.get("/api/tts")
async def get_tts(text: str = Query(...), lang: str = "vi"):
    if not text or not text.strip():
        return JSONResponse(status_code=400, content={"error": "Nội dung văn bản trống."})

    safe_text = text[:200].strip()
    base_url = "https://translate.googleapis.com/translate_tts"
    params = {
        "client": "gtx",
        "ie": "UTF-8",
        "tl": lang,
        "q": safe_text
    }
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    
    async with httpx.AsyncClient() as client:
        try:
            resp = await client.get(base_url, params=params, headers=headers, timeout=15.0)
            if resp.status_code != 200:
                return JSONResponse(status_code=resp.status_code, content={"error": "Google TTS phản hồi lỗi."})
            return StreamingResponse(io.BytesIO(resp.content), media_type="audio/mpeg")
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Lỗi kết nối TTS: {str(e)}"})

# ==========================================
# 3. API GHÉP NỐI MP3 HÀNG LOẠT
# ==========================================
class BulkTTSRequest(BaseModel):
    texts: list[str]
    lang: str = "vi"

@app.post("/api/tts/bulk")
async def bulk_tts(req: BulkTTSRequest):
    base_url = "https://translate.googleapis.com/translate_tts"
    headers = {"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"}
    combined_audio = bytearray()
    
    async with httpx.AsyncClient() as client:
        for text in req.texts:
            if not text or not text.strip(): continue
            safe_text = text[:200].strip()
            params = {"client": "gtx", "ie": "UTF-8", "tl": req.lang, "q": safe_text}
            try:
                resp = await client.get(base_url, params=params, headers=headers, timeout=15.0)
                if resp.status_code == 200: combined_audio.extend(resp.content)
            except Exception:
                pass
                
    if not combined_audio:
        return JSONResponse(status_code=500, content={"error": "Lỗi tổng hợp audio."})
        
    return StreamingResponse(
        io.BytesIO(combined_audio), 
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment; filename=Merged_OCR_AudioBook.mp3"}
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
