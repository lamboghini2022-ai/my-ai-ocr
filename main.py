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

# Thử tải biến môi trường từ file .env nếu chạy ở môi trường local
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="SaaS OCR Reader Backend")

# ==========================================
# CẤU HÌNH CORS MIDDLEWARE (Quan trọng để Frontend gọi vào Render)
# ==========================================
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Cho phép tất cả các nguồn (bao gồm cả file HTML local) truy cập
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Tạo thư mục static nếu chưa tồn tại để tránh lỗi mount
if not os.path.exists("static"):
    os.makedirs("static")

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def root_endpoint():
    return JSONResponse(content={
        "status": "online",
        "message": "Backend OCR Reader (Sync với May_Doc_Sach.html) đang chạy ổn định!",
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
    print("\n========== BẮT ĐẦU XỬ LÝ YÊU CẦU OCR LỚN ==========")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        print("[LỖI] Hệ thống thiếu biến môi trường GEMINI_API_KEY")
        return JSONResponse(
            status_code=500, 
            content={"error": "Chưa cấu hình biến môi trường GEMINI_API_KEY trên hệ thống server Render."}
        )

    model_name = "gemini-2.5-flash"
    url = f"[https://generativelanguage.googleapis.com/v1beta/models/](https://generativelanguage.googleapis.com/v1beta/models/){model_name}:generateContent?key={api_key}"

    # PROMPT ĐƯỢC THIẾT KẾ ĐẶC BIỆT
    PROMPT_TEXT = r"""
Bạn là một Hệ thống Trích xuất Dữ liệu OCR chuyên nghiệp. Nhiệm vụ của bạn là số hóa nội dung từ hình ảnh hoặc văn bản thô được cung cấp, sau đó BẮT BUỘC trả về định dạng JSON là một MẢNG (ARRAY) CHỨA CÁC OBJECT.

Cấu trúc JSON bắt buộc phải tuân theo mẫu sau:
[
  {
    "visual": "Đề kiểm tra môn Vật Lý\n\nCâu 1: Tính vận tốc...",
    "spoken": "Đề kiểm tra môn Vật Lý. Câu một: Tính vận tốc..."
  }
]

📐 QUY TẮC CHO TRƯỜNG "visual" (GIAO DIỆN HIỂN THỊ TRÊN MÀN HÌNH):
- TUYỆT ĐỐI KHÔNG DÙNG THẺ HTML. Hãy dùng ký tự ngắt dòng `\n\n` (viết liền dạng chữ, không bấm phím Enter xuống dòng thực tế) để chia các phân đoạn văn bản.
- Mọi công thức Toán học, Vật Lý, Hóa học phức tạp BẮT BUỘC phải chuyển đổi thành mã LaTeX chuẩn.
- Công thức nằm cùng dòng văn bản (Inline): Bọc bằng ký tự `$`. Ví dụ: `$v = s/t$`
- Công thức đứng riêng một dòng (Block): Bọc bằng ký tự `$$`. Ví dụ: `$$F = m \cdot a$$`

🚨 QUY TẮC CHO TRƯỜNG "spoken" (DÙNG ĐỂ CHUYỂN THÀNH GIỌNG NÓI TTS):
- Hãy chia nhỏ nội dung bài học thành các câu ngắn. Mỗi object trong mảng chỉ nên chứa tối đa 1-2 câu ngắn (khoảng 15 đến 30 từ) để tránh máy đọc bị ngắt quãng hoặc hụt hơi.
- TUYỆT ĐỐI KHÔNG chứa mã LaTeX hoặc ký tự đặc biệt của Toán học. Phải dịch toàn bộ các công thức thành ngôn ngữ nói tiếng Việt tự nhiên (Ví dụ: dịch "H2O" thành "H hai O", dịch "x^2 + y^2" thành "x bình phương cộng y bình phương", dịch "\sqrt{x}" thành "căn bậc hai của x").
- Chỉ chứa chữ cái tiếng Việt có dấu, số đếm thông thường và các dấu câu cơ bản (, . ! ?).

⚠️ QUY TẮC AN TOÀN JSON TỐI CAO ĐỂ TRÁNH LỖI PHÂN TÍCH (JSON DECODE ERROR):
1. TUYỆT ĐỐI KHÔNG SỬ DỤNG HÀNH VI BẤM XUỐNG DÒNG THỰC TẾ (literal newline) bên trong chuỗi giá trị JSON. Tất cả các dấu xuống dòng của công thức ma trận, hệ phương trình hay phân đoạn văn bản bắt buộc phải được viết dưới dạng chuỗi ký tự thoát là `\n`.
2. BẮT BUỘC PHẢI NHÂN ĐÔI DẤU GẠCH CHÉO NGƯỢC (thành `\\`) cho TOÀN BỘ các lệnh điều khiển LaTeX (Ví dụ: viết là `\\frac{a}{b}`, `\\begin{cases}`, `\\Delta`, `\\rightarrow`). Nếu thiếu dấu gạch chéo kép, chuỗi JSON sẽ bị hỏng hoàn toàn.
3. Bất kỳ ký tự ngoặc kép (") nào xuất hiện bên trong nội dung văn bản hoặc mã LaTeX bắt buộc phải được đổi thành dấu ngoặc đơn (') để tránh xung đột làm đóng chuỗi JSON sớm.
    """

    parts = []
    if req.fileBase64 and req.mimeType:
        # Làm sạch chuỗi Base64 loại bỏ phần header của data-uri nếu frontend gửi lên thừa
        clean_b64 = re.sub(r'^data:[a-zA-Z0-9/+]+;base64,', '', req.fileBase64)
        parts.append({"inlineData": {"mimeType": req.mimeType, "data": clean_b64}})
        
    if req.rawText:
        parts.append({"text": req.rawText})
    
    parts.append({"text": PROMPT_TEXT})

    # SỬ DỤNG STRUCTURED OUTPUTS (responseSchema)
    payload = {
        "contents": [{"parts": parts}],
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ],
        "generationConfig": {
            "temperature": 0.1,  
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json",
            "responseSchema": {
                "type": "ARRAY",
                "items": {
                    "type": "OBJECT",
                    "properties": {
                        "visual": {"type": "STRING"},
                        "spoken": {"type": "STRING"}
                    },
                    "required": ["visual", "spoken"]
                }
            }
        }
    }

    async with httpx.AsyncClient() as client:
        try:
            print("[INFO] Đang gửi yêu cầu và đợi Google Gemini API xử lý...")
            response = await client.post(url, json=payload, timeout=60.0)
            
            if response.status_code != 200:
                print(f"[API ERROR] Google API trả về mã lỗi HTTP: {response.status_code}")
                return JSONResponse(status_code=response.status_code, content={"error": f"Lỗi từ phía Gemini: {response.text}"})
            
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return JSONResponse(status_code=400, content={"error": "Yêu cầu bị chặn bởi bộ lọc an toàn đầu vào của Google."})
                
            candidate = candidates[0]
            if "content" not in candidate:
                return JSONResponse(status_code=400, content={"error": "Nội dung phản hồi bị vi phạm bộ lọc đầu ra của AI."})

            raw_result = candidate["content"]["parts"][0]["text"].strip()
            
            # ĐÃ FIX: Tiền xử lý dọn dẹp chuỗi xóa block markdown ```json nếu có
            raw_result = re.sub(r"^```json\n?|```$", "", raw_result, flags=re.MULTILINE).strip()
            
            try:
                # Parse mảng JSON trả về từ AI
                parsed_json = json.loads(raw_result, strict=False)
                print(f"[SUCCESS] Trích xuất thành công {len(parsed_json)} đoạn văn bản.")
                return {"result": parsed_json} # Trả về đúng mảng object cho Frontend
            except json.JSONDecodeError as e:
                print(f"[CRITICAL ERROR] JSON lỗi định dạng: {e}")
                return JSONResponse(status_code=500, content={"error": "AI trả về chuỗi JSON không hợp lệ.", "raw": raw_result})
            
        except httpx.ReadTimeout:
            print("[TIMEOUT] Quá thời gian 60 giây.")
            return JSONResponse(status_code=504, content={"error": "Quá thời gian phản hồi (60 giây)."})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Lỗi nội bộ: {str(e)}"})

# ==========================================
# 2. API PROXY GOOGLE TTS (ĐỌC TỪNG CÂU)
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
# 3. API GHÉP NỐI MP3 HÀNG LOẠT 
# ==========================================
class BulkTTSRequest(BaseModel):
    texts: list[str]
    lang: str = "vi"

@app.post("/api/tts/bulk")
async def bulk_tts(req: BulkTTSRequest):
    print(f"\n========== TỔNG HỢP AUDIO TỔNG ({len(req.texts)} phần tử) ==========")
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
                print(f"[WARNING] Bỏ qua đoạn âm thanh lỗi: {e}")
                
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
