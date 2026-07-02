import os
import base64
import re
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import google.generativeai as genai
from google.generativeai.types import HarmCategory, HarmBlockThreshold

# 1. Khởi tạo ứng dụng FastAPI
app = FastAPI()

# 2. Cấu hình CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],          
    allow_credentials=True,
    allow_methods=["*"],          
    allow_headers=["*"],          
)

# 3. Khởi tạo API Key
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")

if not GEMINI_API_KEY:
    raise ValueError("LỖI: Không tìm thấy GEMINI_API_KEY!")

genai.configure(api_key=GEMINI_API_KEY)

# Sử dụng model hỗ trợ xử lý file tốt
model = genai.GenerativeModel("gemini-2.5-flash")

# 4. Model nhận dữ liệu - Thêm trường doc_type
class ScanRequest(BaseModel):
    mime_type: str
    base64_data: str
    doc_type: str = "document" # Có thể là 'exam' (đề thi) hoặc 'document' (tài liệu thường)

# 5. ĐỊNH NGHĨA CÁC LOẠI PROMPT

# Prompt dành riêng cho Đề thi (Chứa logic bảng trắc nghiệm, chống bộ lọc)
PROMPT_EXAM = r"""Bạn là hệ thống AI OCR chuyên nghiệp. Nhiệm vụ: Số hóa đề thi, định dạng HTML chuẩn xác và bọc TOÀN BỘ kết quả vào TRONG MỘT cặp thẻ `<chunk>` và `</chunk>`.

- TUYỆT ĐỐI KHÔNG sao chép các hàng dấu chấm dài (........) ở phần Họ tên, Số báo danh. Chỉ ghi `...`. 
- Chú ý giữ nguyên dấu sau chữ "Câu" (VD: Câu 1., Câu 1:).

CẤU TRÚC TRÌNH BÀY HTML:
1. Tiêu đề (Bảng 2 cột):
<table style="width:100%; border:none; margin-bottom:10px;">
  <tr>
    <td style="width:50%; text-align:left;">SỞ GD&ĐT...<br><b>TRƯỜNG THPT...</b></td>
    <td style="width:50%; text-align:center;"><b>ĐỀ KIỂM TRA...</b><br>Mã đề: ...</td>
  </tr>
</table>

2. Câu hỏi và Đáp án: 
- In đậm chữ "Câu...".
- Các đáp án trắc nghiệm BẮT BUỘC bọc trong bảng 1 hàng 4 cột (25% mỗi cột). Nếu đáp án dài, dùng bảng 2 cột (50%).
- Dùng mã LaTeX nguyên bản cho công thức toán lý hóa (chèn thêm nhóm `{}` rỗng để tránh trùng lặp dữ liệu tĩnh).
"""

# Prompt dành cho Tài liệu văn bản/Hợp đồng (Giữ nguyên cấu trúc văn bản hành chính)
PROMPT_DOCUMENT = r"""Bạn là hệ thống AI OCR chuyên nghiệp. Nhiệm vụ: Số hóa tài liệu văn bản, hợp đồng, giữ nguyên cấu trúc hành chính và bọc TOÀN BỘ kết quả vào TRONG MỘT cặp thẻ `<chunk>` và `</chunk>`.

YÊU CẦU ĐỊNH DẠNG HTML BẮT BUỘC:
1. Xử lý đoạn văn: Sử dụng thẻ `<p style="text-align: justify; margin-bottom: 8px;">` cho các đoạn văn bản thường.
2. Căn lề Quốc hiệu, Tiêu ngữ, Tên cơ quan: Sử dụng thẻ `<div style="text-align: center; font-weight: bold;">` cho các phần như "CỘNG HÒA XÃ HỘI CHỦ NGHĨA VIỆT NAM" hoặc tên thông tư.
3. Danh sách: Nếu tài liệu có các điểm a, b, c, d hoặc 1, 2, 3, hãy dùng thẻ `<ul>` và `<li>` để thục lề cho rõ ràng.
4. Chữ ký và dấu: Trình bày phần nơi nhận và chữ ký dưới dạng bảng 2 cột vô hình (`<table style="width:100%; border:none;">...`) để căn lề hai bên.
5. Lược bỏ nhiễu: TUYỆT ĐỐI KHÔNG lặp lại các chuỗi dấu chấm quá dài (........). Thay thế chúng bằng `......` (6 dấu chấm) để tránh làm sập bộ nhớ. KHÔNG bọc đầu ra bằng khối mã Markdown (```).
"""

# 6. Endpoint xử lý chính
@app.post("/api/scan")
async def scan_image(request: ScanRequest):
    try:
        # Lựa chọn Prompt dựa trên loại tài liệu
        prompt_text = PROMPT_EXAM if request.doc_type == "exam" else PROMPT_DOCUMENT

        base64_data = request.base64_data
        if "base64," in base64_data:
            base64_data = base64_data.split("base64,")[1]

        try:
            image_bytes = base64.b64decode(base64_data)
        except Exception:
            raise HTTPException(status_code=400, detail="Chuỗi dữ liệu Base64 không hợp lệ.")

        image_parts = [
            {
                "mime_type": request.mime_type,
                "data": image_bytes
            }
        ]

        safety_settings = {
            HarmCategory.HARM_CATEGORY_HARASSMENT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_HATE_SPEECH: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: HarmBlockThreshold.BLOCK_NONE,
            HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: HarmBlockThreshold.BLOCK_NONE,
        }

        response = model.generate_content(
            contents=[prompt_text, image_parts[0]],
            generation_config={
                "temperature": 0.2, # Giảm nhiệt độ xuống để OCR chính xác hơn, bớt tự bịa data
                "max_output_tokens": 8192
            },
            safety_settings=safety_settings
        )

        if response.candidates and len(response.candidates) > 0:
            candidate = response.candidates[0]
            
            if candidate.content and candidate.content.parts:
                result_text = response.text
                
                chunks = re.findall(r'<chunk>(.*?)</chunk>', result_text, re.DOTALL)
                
                if chunks:
                    html_content = "".join([f"<div class='ocr-item' style='margin-bottom: 6px; line-height: 1.5;'>{c.strip()}</div>" for c in chunks])
                    return {"status": "success", "html_content": html_content}
                else:
                    clean_html = result_text.replace("```xml", "").replace("```html", "").replace("```", "").strip()
                    return {"status": "success", "html_content": f"<div class='ocr-raw'>{clean_html}</div>"}

        finish_reason = getattr(response.candidates[0], 'finish_reason', 'UNKNOWN') if response.candidates else 'NO_CANDIDATES'
        raise HTTPException(
            status_code=500, 
            detail=f"Mô hình AI từ chối xử lý hoặc phản hồi bị ngắt do quá dài (Token Limit). Mã lỗi: {finish_reason}. Vui lòng chia nhỏ tài liệu."
        )

    except HTTPException as http_ex:
        raise http_ex
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Lỗi hệ thống máy chủ: {str(e)}")
