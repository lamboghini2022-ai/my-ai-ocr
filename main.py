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
    images: Optional[List[ImagePayload]] = None  
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

    # Cập nhật model name chuẩn xác của Google (1.5 hoặc 2.0)
    model_name = "gemini-2.5-flash" 
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    # PROMPT ĐÃ ĐƯỢC TỐI ƯU HÓA LẠI
    PROMPT_TEXT = """
Bạn là Hệ thống Trích xuất Dữ liệu OCR chuyên nghiệp.
Nhiệm vụ: Trích xuất TOÀN BỘ văn bản từ hình ảnh, không bỏ sót chữ, số, công thức Toán, Lý, hay Hóa học nào.
Chia nội dung thành các phần logic (theo từng đoạn văn, câu hỏi, hoặc bài tập).

YÊU CẦU CHO TRƯỜNG "visual" (Hiển thị UI):
1. Giữ nguyên cấu trúc. Dùng Markdown (`**đậm**`, `*nghiêng*`) để làm nổi bật tiêu đề hoặc nhấn mạnh nếu cần.
2. Dùng `\n\n` để ngắt dòng giữa các đoạn, tạo khoảng cách rộng rãi, dễ đọc.
3. TẤT CẢ công thức, ký hiệu Toán/Lý/Hóa BẮT BUỘC dùng mã LaTeX chuẩn.
   - Trong dòng: bọc trong `$ $` (VD: $x^2 + y^2 = z^2$, $H_2SO_4$).
   - Riêng một dòng: bọc trong `$$ $$`.
   - HÓA HỌC: Trình bày rõ ràng các chỉ số trên/dưới, ký hiệu phản ứng (VD: $Fe^{3+}$, $CO_3^{2-}$, $2H_2 + O_2 \rightarrow 2H_2O$).
4. QUAN TRỌNG: KHÔNG TỰ Ý NHÂN ĐÔI DẤU GẠCH CHÉO NGƯỢC. Cứ viết cú pháp LaTeX bình thường (VD: \frac, \sqrt, \rightarrow). Hệ thống schema JSON sẽ tự động escape.

YÊU CẦU CHO TRƯỜNG "spoken" (Đọc giọng nói TTS):
1. KHÔNG chứa mã LaTeX, KHÔNG chứa ký hiệu toán/hóa học phức tạp.
2. Dịch mọi công thức ra tiếng Việt thuần túy, mô phỏng cách con người đọc:
   - Toán: "x bình phương", "căn bậc hai của a", "phân số a phần b".
   - Hóa học: Đọc rõ từng chất. VD: "Hát hai ét ô bốn" (H2SO4), "sắt ba cộng" (Fe3+), "mũi tên tạo ra" (->), "nhiệt độ" (t độ).
3. Độ dài mỗi đoạn vừa phải (15 - 40 từ), ngắt nghỉ bằng dấu phẩy và dấu chấm để máy đọc trôi chảy.
    """

    parts = []
    
    if req.images:
        print(f"[INFO] Nhận được {len(req.images)} ảnh. Đang đưa vào AI...")
        for img in req.images:
            clean_b64 = re.sub(r'^data:[a-zA-Z0-9/+]+;base64,', '', img.fileBase64)
            parts.append({"inlineData": {"mimeType": img.mimeType, "data": clean_b64}})
            
    if req.rawText:
        parts.append({"text": req.rawText})
    
    parts.append({"text": PROMPT_TEXT})

    # Cấu hình Payload có kèm RESPONSE SCHEMA
    payload = {
        "contents": [{"parts": parts}],
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ],
        "generationConfig": {
            "temperature": 0.1,  # Hạ nhiệt độ xuống 0.1 để OCR bám sát văn bản gốc nhất có thể
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
                            "description": "Văn bản hiển thị. Dùng Markdown và \\n\\n ngắt dòng. Sử dụng LaTeX chuẩn cho Toán/Lý/Hóa (không nhân đôi backslash)."
                        },
                        "spoken": {
                            "type": "STRING",
                            "description": "Văn bản để máy đọc tiếng Việt, không chứa LaTeX, phiên âm rõ ràng công thức hóa học và toán học ra chữ."
                        }
                    },
                    "required": ["visual", "spoken"]
                }
            }
        }
    }

    async with httpx.AsyncClient() as client:
        try:
            print(f"[INFO] Đang gửi yêu cầu đến {model_name} API...")
            response = await client.post(url, json=payload, timeout=120.0) 
            
            if response.status_code != 200:
                print(f"[LỖI API] {response.text}")
                return JSONResponse(status_code=response.status_code, content={"error": f"Lỗi Gemini: {response.text}"})
            
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return JSONResponse(status_code=400, content={"error": "Bị chặn bởi bộ lọc an toàn hoặc không thể trích xuất."})
                
            candidate = candidates[0]
            if "content" not in candidate:
                return JSONResponse(status_code=400, content={"error": "AI không trả về nội dung."})

            raw_result = candidate["content"]["parts"][0]["text"].strip()
            
            try:
                # Phân tích cú pháp JSON
                parsed_json = json.loads(raw_result, strict=False)
                
                # Kiểm tra nếu AI trả về mảng rỗng
                if not parsed_json:
                    print("[CẢNH BÁO] Trích xuất thành công nhưng mảng rỗng []!")
                    return JSONResponse(status_code=400, content={"error": "Không tìm thấy nội dung hợp lệ hoặc ảnh chứa dữ liệu khó đọc."})

                print(f"[SUCCESS] Trích xuất thành công {len(parsed_json)} đoạn văn bản.")
                return {"result": parsed_json}

            except json.JSONDecodeError as e:
                print(f"[CRITICAL ERROR] JSON lỗi định dạng: {e}")
                return JSONResponse(status_code=500, content={"error": "Lỗi định dạng JSON trả về từ AI.", "raw": raw_result})
            
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
    print(f"\n========== TỔNG HỢP AUDIO ({len(req.texts)} đoạn) ==========")
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
