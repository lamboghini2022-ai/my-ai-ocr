require('dotenv').config();
const express = require('express');
const cors = require('cors');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;

// Cấu hình Middleware
app.use(cors());
// Tăng giới hạn payload lên 50MB vì Base64 của ảnh/PDF khá nặng
app.use(express.json({ limit: '50mb' })); 
// Phục vụ các file tĩnh trong thư mục 'public' (Frontend)
app.use(express.static(path.join(__dirname, 'public')));

// ==========================================
// 1. API GỌI GEMINI (Xử lý OCR)
// ==========================================
app.post('/api/extract', async (req, res) => {
    try {
        const { fileBase64, mimeType, rawText } = req.body;
        const apiKey = process.env.GEMINI_API_KEY;

        if (!apiKey) {
            return res.status(500).json({ error: "Server chưa cấu hình Gemini API Key" });
        }

        const modelName = "gemini-2.5-flash";
        const endpoint = `https://generativelanguage.googleapis.com/v1beta/models/${modelName}:generateContent?key=${apiKey}`;

        const prompt = `Bạn là trợ lý AI xử lý tài liệu. Trích xuất toàn bộ văn bản và trả về DUY NHẤT một mảng JSON.
CHÚ Ý QUAN TRỌNG ĐỂ KHÔNG BỊ LỖI AUDIO: Mỗi phần tử trong mảng JSON phải là MỘT CÂU NGẮN (tối đa 150 ký tự). Nếu câu gốc quá dài, HÃY TỰ ĐỘNG CẮT NGẮT thành nhiều phần tử liên tiếp nhau.

QUY TẮC BẮT BUỘC CHO MẢNG JSON:
- "visual": Dùng mã LaTeX bọc trong $$...$$ (đứng một mình) hoặc \\( ... \\) (trong dòng) cho TẤT CẢ công thức Toán/Hóa học để MathJax có thể vẽ. Giữ lại nguyên vẹn khoảng trắng (space) ở đầu dòng và ký tự xuống dòng (\\n) ở cuối để dựng layout như bản gốc.
- "spoken": Dịch công thức sang CHỮ TIẾNG VIỆT thuần túy để máy tính phát âm (vd: "x bình phương", "H hai O").
- Tuyệt đối không thêm văn bản ngoài mảng JSON.`;

        let parts = [];
        if (fileBase64 && mimeType) {
            parts.push({ inlineData: { mimeType: mimeType, data: fileBase64 } });
        }
        if (rawText) {
            parts.push({ text: "Dữ liệu gốc:\n" + rawText });
        }
        parts.push({ text: prompt });

        const payload = {
            contents: [{ parts: parts }],
            generationConfig: { temperature: 0.1 }
        };

        const response = await fetch(endpoint, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify(payload)
        });

        if (!response.ok) {
            const errData = await response.json();
            throw new Error(errData.error?.message || response.statusText);
        }

        const data = await response.json();
        if (!data.candidates || data.candidates.length === 0) {
            throw new Error("AI không trả về kết quả.");
        }
        
        let resultText = data.candidates[0].content.parts[0].text;
        resultText = resultText.replace(/```json/gi, '').replace(/```/g, '').trim();
        
        res.json({ result: JSON.parse(resultText) });

    } catch (error) {
        console.error("Lỗi API Gemini:", error.message);
        res.status(500).json({ error: error.message });
    }
});

// ==========================================
// 2. API PROXY GỌI GOOGLE TTS (Sửa lỗi CORS)
// ==========================================
app.get('/api/tts', async (req, res) => {
    try {
        const { text, lang } = req.query;
        if (!text) return res.status(400).send("Thiếu text");
        
        const targetUrl = `https://translate.googleapis.com/translate_tts?client=gtx&ie=UTF-8&tl=${lang || 'vi'}&q=${encodeURIComponent(text)}`;
        
        const response = await fetch(targetUrl, {
            headers: { 'User-Agent': 'Mozilla/5.0' } // Giả lập trình duyệt để tránh bị Google chặn
        });

        if (!response.ok) throw new Error("Google TTS từ chối kết nối");

        // Set header để trả file Audio về thẳng Frontend
        res.set('Content-Type', 'audio/mpeg');
        
        // Pipe data từ Google thẳng về Client
        const arrayBuffer = await response.arrayBuffer();
        const buffer = Buffer.from(arrayBuffer);
        res.send(buffer);

    } catch (error) {
        console.error("Lỗi TTS Proxy:", error.message);
        res.status(500).send("Lỗi tạo Audio");
    }
});

// Khởi động server
app.listen(PORT, () => {
    console.log(`🚀 Server đang chạy tại http://localhost:${PORT}`);
});
