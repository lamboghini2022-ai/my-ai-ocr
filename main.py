import os
import json
import httpx
import re
import io
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, JSONResponse
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
        "message": "Backend OCR Reader (Multi-image & Schema) đang chạy!"
    })

# ==========================================
# CÁC MODEL DỮ LIỆU ĐẦU VÀO
# ==========================================
class ImagePayload(BaseModel):
    fileBase64: str
    mimeType: str

class ExtractRequest(BaseModel):
    images: Optional[List[ImagePayload]] = None  # Hỗ trợ gửi mảng nhiều ảnh
    rawText: Optional[str] = None

class BulkTTSRequest(BaseModel):
    texts: list[str]
    lang: str = "vi"

# ==========================================
# 1. API XỬ LÝ OCR (HỖ TRỢ LATEX & BATCH)
# ==========================================
@app.post("/api/extract") 
async def extract_text(req: ExtractRequest):
    print("\n========== BẮT ĐẦU XỬ LÝ YÊU CẦU OCR (BATCH) ==========")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[LỖI] Thiếu cấu hình GEMINI_API_KEY")
        return JSONResponse(status_code=500, content={"error": "Thiếu biến môi trường GEMINI_API_KEY."})

    model_name = "gemini-2.5-flash" 
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    # PROMPT rõ ràng, không đe dọa AI để tránh lỗi sinh mảng rỗng
    PROMPT_TEXT = """
Bạn là Hệ thống Trích xuất Dữ liệu OCR (Optical Character Recognition) chuyên nghiệp.
Nhiệm vụ của bạn là đọc TOÀN BỘ văn bản từ các hình ảnh tôi cung cấp, không được bỏ sót bất kỳ chữ hay công thức nào.
Hãy chia nội dung thành các phần logic (theo từng đoạn văn, hoặc từng câu hỏi).

YÊU CẦU TRÌNH BÀY CHO TRƯỜNG "visual" (Hiển thị UI):
1. Giữ nguyên cấu trúc văn bản gốc. Dùng `\n\n` để ngắt dòng, tạo khoảng cách rộng rãi, dễ đọc.
2. TẤT CẢ công thức, ký hiệu Toán/Lý/Hóa BẮT BUỘC dùng mã LaTeX.
3. Chú ý: Vì output là JSON, bạn phải nhân đôi dấu gạch chéo ngược cho LaTeX. Ví dụ: viết `\\\\frac{a}{b}` (thay vì \frac), `\\\\sqrt{x}` (thay vì \sqrt), `\\\\Delta` (thay vì \Delta).
4. Công thức nằm trong dòng bọc bằng `$ $`, công thức đứng riêng 1 dòng bọc bằng `$$ $$`.

YÊU CẦU TRÌNH BÀY CHO TRƯỜNG "spoken" (Đọc giọng nói TTS):
1. KHÔNG chứa mã LaTeX, KHÔNG chứa ký hiệu toán học phức tạp.
2. Dịch mọi công thức ra tiếng Việt thuần túy (VD: "x bình phương", "căn bậc hai của a", "phân số a phần b").
3. Độ dài mỗi đoạn vừa phải (15 - 30 từ) để máy đọc không bị hụt hơi.
    """

    parts = []
    
    # Xử lý mảng hình ảnh
    if req.images:
        print(f"[INFO] Nhận được {len(req.images)} ảnh. Đang đưa vào AI...")
        for img in req.images:
            clean_b64 = re.sub(r'^data:[a-zA-Z0-9/+]+;base64,', '', img.fileBase64)
            parts.append({"inlineData": {"mimeType": img.mimeType, "data": clean_b64}})
            
    if req.rawText:
        parts.append({"text": req.rawText})
    
    parts.append({"text": PROMPT_TEXT})

    # Cấu hình Payload có kèm RESPONSE SCHEMA (Bắt buộc AI trả JSON chuẩn)
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
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "ARRAY",
                "description": "Danh sách các đoạn văn bản được trích xuất từ ảnh.",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "visual": {
                            "type": "STRING",
                            "description": "Văn bản hiển thị. Dùng \\n\\n ngắt dòng. Phải chứa LaTeX có double backslash (\\\\) cho Toán học."
                        },
                        "spoken": {
                            "type": "STRING",
                            "description": "Văn bản để máy đọc tiếng Việt, không chứa LaTeX, dịch công thức ra chữ."
                        }
                    },
                    "required": ["visual", "spoken"]
                }
            }
        }
    }

    async with httpx.AsyncClient() as client:
        try:
            print("[INFO] Đang gửi yêu cầu đến Google Gemini API...")
            response = await client.post(url, json=payload, timeout=120.0) 
            
            if response.status_code != 200:
                print(f"[LỖI API] {response.text}")
                return JSONResponse(status_code=response.status_code, content={"error": f"Lỗi Gemini: {response.text}"})
            
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return JSONResponse(status_code=400, content={"error": "Bị chặn bởi bộ lọc an toàn."})
                
            candidate = candidates[0]
            if "content" not in candidate:
                return JSONResponse(status_code=400, content={"error": "AI không trả về nội dung."})

            raw_result = candidate["content"]["parts"][0]["text"].strip()
            
            try:
                # Do đã có responseSchema, tỷ lệ lỗi parse JSON gần như bằng 0
                parsed_json = json.loads(raw_result, strict=False)
                
                # Kiểm tra nếu AI trả về mảng rỗng
                if not parsed_json:
                    print("[CẢNH BÁO] Trích xuất thành công nhưng mảng rỗng []!")
                    return JSONResponse(status_code=400, content={"error": "AI không tìm thấy văn bản trong ảnh hoặc ảnh quá mờ."})

                print(f"[SUCCESS] Trích xuất thành công {len(parsed_json)} đoạn văn bản.")
                return {"result": parsed_json}

            except json.JSONDecodeError as e:
                print(f"[CRITICAL ERROR] JSON lỗi định dạng: {e}")
                return JSONResponse(status_code=500, content={"error": "JSON parse error", "raw": raw_result})
            
        except httpx.ReadTimeout:
            print("[TIMEOUT] Quá thời gian 120 giây.")
            return JSONResponse(status_code=504, content={"error": "Quá thời gian phản hồi (120 giây)."})
        except Exception as e:
            print(f"[LỖI NỘI BỘ] {str(e)}")
            return JSONResponse(status_code=500, content={"error": f"Lỗi máy chủ: {str(e)}"})


# ==========================================
# 2. API PROXY GOOGLE TTS (ĐỌC TỪNG CÂU)
# ==========================================
@app.get("/api/tts")
async def get_tts(text: str = Query(...), lang: str = "vi"):
    target_url = f"https://translate.googleapis.com/translate_tts?client=gtx&ie=UTF-8&tl={lang}&q={text}"
    headers = {"User-Agent": "Mozilla/5.0"}
    
    async def stream_audio():
        async with httpx.AsyncClient() as client:
            async with client.stream("GET", target_url, headers=headers) as r:
                async for chunk in r.aiter_bytes():
                    yield chunk

    return StreamingResponse(stream_audio(), media_type="audio/mpeg")


# ==========================================
# 3. API GHÉP NỐI MP3 HÀNG LOẠT 
# ==========================================
@app.post("/api/tts/bulk")
async def bulk_tts(req: BulkTTSRequest):
    print(f"\n========== TỔNG HỢP AUDIO ({len(req.texts)} phần tử) ==========")
    headers = {"User-Agent": "Mozilla/5.0"}
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
                print(f"[WARNING] Bỏ qua âm thanh lỗi: {e}")
                
    if not combined_audio:
        return JSONResponse(status_code=500, content={"error": "Không thể tải audio từ server TTS."})
        
    return StreamingResponse(
        io.BytesIO(combined_audio), 
        media_type="audio/mpeg",
        headers={"Content-Disposition": "attachment; filename=Merged_OCR_AudioBook.mp3"}
    )

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
