import os
import json
import httpx
import re
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import StreamingResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional

# Thử load file .env (nếu bạn đang dùng file .env ở máy cá nhân)
try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

app = FastAPI()

# Đảm bảo thư mục 'static' tồn tại để không lỗi khi khởi động
if not os.path.exists("static"):
    os.makedirs("static")
    with open("static/index.html", "w", encoding="utf-8") as f:
        f.write("<h1>Trang chủ Backend</h1>")

# Cấu hình phục vụ các file tĩnh
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.get("/")
async def read_index():
    return FileResponse("static/index.html")

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
        print("[LỖI NGHIÊM TRỌNG] Chưa có GEMINI_API_KEY")
        return JSONResponse(
            status_code=500, 
            content={"error": "Chưa cấu hình biến môi trường GEMINI_API_KEY."}
        )

    model_name = "gemini-2.5-flash"
    url = f"https://generativelanguage.googleapis.com/v1beta/models/{model_name}:generateContent?key={api_key}"

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
        # BẢN VÁ QUAN TRỌNG: Xóa tiền tố 'data:image/...;base64,' nếu frontend lỡ gửi kèm
        clean_b64 = re.sub(r'^data:[a-zA-Z0-9/+]+;base64,', '', req.fileBase64)
        parts.append({"inlineData": {"mimeType": req.mimeType, "data": clean_b64}})
        print("[THÔNG TIN] Đã nhận và dọn dẹp chuỗi Base64 từ Frontend.")
        
    if req.rawText:
        parts.append({"text": req.rawText})
    
    parts.append({"text": prompt})

    payload = {
        "contents": [{"parts": parts}],
        "generationConfig": {
            "temperature": 0.1,
            "responseMimeType": "application/json" 
        },
        "safetySettings": [
            {"category": "HARM_CATEGORY_HARASSMENT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_HATE_SPEECH", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_SEXUALLY_EXPLICIT", "threshold": "BLOCK_NONE"},
            {"category": "HARM_CATEGORY_DANGEROUS_CONTENT", "threshold": "BLOCK_NONE"}
        ]
    }

    async with httpx.AsyncClient() as client:
        try:
            print("[THÔNG TIN] Đang gửi yêu cầu lên Google Gemini...")
            response = await client.post(url, json=payload, timeout=60.0)
            
            if response.status_code != 200:
                print(f"[LỖI API GOOGLE] HTTP {response.status_code} - {response.text}")
                return JSONResponse(status_code=response.status_code, content={"error": f"Lỗi Gemini API: {response.text}"})
            
            data = response.json()
            
            # KIỂM TRA LỖI PROMPT (Bị chặn từ vòng gửi xe)
            candidates = data.get("candidates", [])
            if not candidates:
                prompt_feedback = data.get("promptFeedback", {})
                print("[LỖI AN TOÀN] Bị chặn từ vòng gửi xe (Prompt Feedback):", prompt_feedback)
                return JSONResponse(
                    status_code=400, 
                    content={"error": f"API chặn yêu cầu. Prompt Feedback: {prompt_feedback}"}
                )
                
            candidate = candidates[0]
            
            # KIỂM TRA LỖI CONTENT (Bị chặn bởi Safety Filter sau khi tạo text)
            if "content" not in candidate:
                finish_reason = candidate.get("finishReason", "Lý do không xác định")
                print("[LỖI AN TOÀN] Bị chặn bởi Safety Filter. Finish Reason:", finish_reason)
                return JSONResponse(
                    status_code=400, 
                    content={"error": f"AI từ chối trả lời do Safety Filter. Lý do: {finish_reason}."}
                )

            raw_result = candidate["content"]["parts"][0]["text"].strip()
            clean_text = raw_result.replace("```json", "").replace("```", "").strip()
            
            # XỬ LÝ VÀ SỬA LỖI JSON
            try:
                parsed_json = json.loads(clean_text, strict=False)
                print("[THÀNH CÔNG] Đã trích xuất và parse JSON hoàn tất!")
                return {"result": parsed_json}
            except json.JSONDecodeError:
                print("[CẢNH BÁO] Lỗi Parse JSON lần 1, đang kích hoạt bộ sửa lỗi tự động...")
                try:
                    fixed_text = clean_text.replace('\\', '\\\\').replace('\\\\"', '\\"')
                    parsed_json = json.loads(fixed_text, strict=False)
                    print("[THÀNH CÔNG] Đã sửa lỗi backslash và parse JSON thành công!")
                    return {"result": parsed_json}
                except Exception as parse_err:
                    print(f"[LỖI ĐỊNH DẠNG] Không thể parse JSON: {parse_err}")
                    print(f"--- ĐÂY LÀ KẾT QUẢ RAW TỪ AI TRẢ VỀ ---\n{clean_text}\n-------------------")
                    return JSONResponse(
                        status_code=500, 
                        content={
                            "error": "AI trả về chuỗi nhưng không thể parse thành JSON.",
                            "chi_tiet_loi": str(parse_err),
                            "ket_qua_tho": clean_text
                        }
                    )
            
        except httpx.ReadTimeout:
            print("[LỖI MẠNG] Timeout - Gọi API Google quá 60s không có phản hồi.")
            return JSONResponse(status_code=504, content={"error": "Hết thời gian chờ từ Google API. Ảnh có thể quá nặng hoặc đường truyền chậm."})
        except Exception as e:
            import traceback
            print("[LỖI NGHIÊM TRỌNG (CRASH BE)]:")
            traceback.print_exc()
            error_detail = str(e) if str(e) else repr(e)
            return JSONResponse(
                status_code=500, 
                content={"error": f"Lỗi hệ thống hoặc định dạng: {error_detail}"}
            )

# ==========================================
# 2. API PROXY GOOGLE TTS 
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", 8000))
    uvicorn.run("main:app", host="0.0.0.0", port=port, reload=True)
