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

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI(title="SaaS OCR Reader Backend")

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
        "message": "Backend OCR Reader đang chạy (Hỗ trợ Multi-image)!"
    })

# ==========================================
# CẬP NHẬT MODEL: HỖ TRỢ NHIỀU ẢNH CÙNG LÚC
# ==========================================
class ImagePayload(BaseModel):
    fileBase64: str
    mimeType: str

class ExtractRequest(BaseModel):
    images: Optional[List[ImagePayload]] = None  # Nhận một mảng (tối đa 5-10 ảnh)
    rawText: Optional[str] = None

# ==========================================
# 1. API XỬ LÝ OCR & TRÍCH XUẤT QUA GEMINI
# ==========================================
@app.post("/api/extract") 
async def extract_text(req: ExtractRequest):
    print("\n========== BẮT ĐẦU XỬ LÝ YÊU CẦU OCR (BATCH) ==========")
    
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        return JSONResponse(status_code=500, content={"error": "Thiếu cấu hình GEMINI_API_KEY"})

    model_name = "gemini-1.5-flash" # Chú ý: Dùng 1.5-flash để ổn định, 2.5-flash hiện đang là bản preview có thể thiếu ổn định.
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

    # PROMPT MỚI: Thêm quy tắc bảo vệ JSON nghiêm ngặt
    PROMPT_TEXT = r"""
Bạn là một Hệ thống Trích xuất Dữ liệu OCR chuyên nghiệp. Tôi có gửi cho bạn một hoặc nhiều hình ảnh cùng lúc. Hãy đọc TẤT CẢ hình ảnh theo thứ tự và BẮT BUỘC trả về duy nhất một định dạng JSON là MẢNG (ARRAY) CHỨA CÁC OBJECT. 

Cấu trúc JSON bắt buộc:
[
  {
    "visual": "Nội dung để hiển thị trên màn hình...",
    "spoken": "Nội dung để đọc bằng giọng nói..."
  }
]

🚨 QUY TẮC BẢO VỆ JSON TRÁNH LỖI (CRITICAL):
- TUYỆT ĐỐI KHÔNG sử dụng ký tự ngoặc kép (") bên trong giá trị của "visual" hoặc "spoken". Nếu cần trích dẫn, hãy dùng ngoặc đơn (') để thay thế.
- TUYỆT ĐỐI KHÔNG dùng ký tự ngắt dòng nguyên thủy. Mọi ngắt dòng phải được viết là `\n\n`.
- Không giải thích gì thêm, chỉ trả về chuỗi JSON thuần túy.

📐 QUY TẮC CHO "visual":
- Dùng ký tự ngắt dòng `\n\n` để chia đoạn.
- Công thức Toán/Lý/Hóa phải dùng mã LaTeX (nhân đôi backslash, ví dụ: `\\frac{a}{b}`).
- Inline LaTeX bọc bằng `$`, Block bọc bằng `$$`.

🚨 QUY TẮC CHO "spoken":
- Chia nội dung thành các câu ngắn (15-30 từ). Mỗi object trong mảng chứa 1-2 câu.
- KHÔNG chứa mã LaTeX. Dịch công thức thành tiếng Việt (Ví dụ: "căn bậc hai của x").
    """

    parts = []
    
    # XỬ LÝ MẢNG ẢNH: Đẩy tất cả ảnh vào cùng 1 request
    if req.images:
        print(f"[INFO] Nhận được {len(req.images)} ảnh để xử lý.")
        for img in req.images:
            clean_b64 = re.sub(r'^data:[a-zA-Z0-9/+]+;base64,', '', img.fileBase64)
            parts.append({"inlineData": {"mimeType": img.mimeType, "data": clean_b64}})
            
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
            "temperature": 0.4, # Giảm temperature xuống 0.1 để AI tuân thủ form mẫu tốt hơn
            "maxOutputTokens": 8192,
            "responseMimeType": "application/json" 
        }
    }

    async with httpx.AsyncClient() as client:
        try:
            print("[INFO] Đang gửi yêu cầu gom nhóm đến Google Gemini API...")
            # Tăng timeout lên 120 giây vì xử lý nhiều ảnh tốn nhiều thời gian hơn
            response = await client.post(url, json=payload, timeout=120.0) 
            
            if response.status_code != 200:
                return JSONResponse(status_code=response.status_code, content={"error": f"Lỗi Gemini: {response.text}"})
            
            data = response.json()
            candidates = data.get("candidates", [])
            if not candidates:
                return JSONResponse(status_code=400, content={"error": "Bị chặn bởi bộ lọc an toàn."})
                
            candidate = candidates[0]
            if "content" not in candidate:
                return JSONResponse(status_code=400, content={"error": "AI vi phạm bộ lọc đầu ra."})

            raw_result = candidate["content"]["parts"][0]["text"].strip()
            
            # Xóa các thẻ markdown nếu AI lỡ sinh ra
            if raw_result.startswith("```json"):
                raw_result = raw_result[7:-3].strip()
            elif raw_result.startswith("```"):
                raw_result = raw_result[3:-3].strip()
            
            try:
                parsed_json = json.loads(raw_result, strict=False)
                print(f"[SUCCESS] Trích xuất thành công {len(parsed_json)} đoạn văn bản.")
                return {"result": parsed_json}
            except json.JSONDecodeError as e:
                print(f"[CRITICAL ERROR] JSON lỗi định dạng: {e}")
                # Hỗ trợ debug nếu còn dính lỗi
                return JSONResponse(status_code=500, content={"error": "AI trả về chuỗi JSON không hợp lệ.", "raw_head": raw_result[:200], "raw_tail": raw_result[-200:]})
            
        except httpx.ReadTimeout:
            print("[TIMEOUT] Quá thời gian 120 giây.")
            return JSONResponse(status_code=504, content={"error": "Quá thời gian phản hồi (120 giây)."})
        except Exception as e:
            return JSONResponse(status_code=500, content={"error": f"Lỗi nội bộ: {str(e)}"})

# ... (Giữ nguyên các endpoint TTS như cũ)
