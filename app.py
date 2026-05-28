import os
import io
import streamlit as nn
import streamlit as st
from google import genai
from google.genai import types

# Cấu hình giao diện trang web (Dark/Light mode tự động)
st.set_page_config(
    page_title="Gemini Text-to-Speech",
    page_icon="🎙️",
    layout="centered"
)

st.title("🎙️ Gemini Text-to-Speech")
st.caption("Ứng dụng chuyển đổi văn bản thành giọng nói sử dụng Google AI Studio API miễn phí.")

# Khởi tạo API Client từ Environment Variable (dành cho Render) hoặc nhập tay (để test cục bộ)
api_key = os.environ.get("GEMINI_API_KEY", "")

with st.sidebar:
    st.header("⚙️ Cấu hình API")
    if not api_key:
        api_key = st.text_input("Nhập Gemini API Key của bạn:", type="password")
        st.info("💡 Bạn có thể lấy key tại Google AI Studio.")
    else:
        st.success("✅ Đã tìm thấy API Key từ hệ thống (Environment Variable).")
        
    st.markdown("---")
    st.markdown("### 🎙️ Cấu hình giọng đọc")
    # Danh sách các giọng đọc mẫu hỗ trợ của Gemini TTS
    voice_option = st.selectbox(
        "Chọn giọng đọc (Voice):",
        ["Aoede", "Charon", "Puck", "Kore", "Fenrir", "Enceladus"]
    )
    
    # Cho phép người dùng tùy biến cảm xúc/ngữ điệu bằng Prompt chỉ dẫn
    style_prompt = st.text_input(
        "Tông giọng / Cảm xúc (Tùy chọn):", 
        value="cheerful and natural", 
        placeholder="Ví dụ: cheerful, serious, whispering..."
    )

# Giao diện chính để nhập nội dung
text_input = st.text_area(
    "Nhập văn bản cần chuyển thành giọng nói:",
    height=200,
    placeholder="Nhập đoạn văn bản tại đây..."
)

if st.button("🔊 Chuyển thành Giọng Nói", type="primary"):
    if not api_key:
        st.error("Vui lòng nhập hoặc cấu hình Gemini API Key trước khi thực hiện!")
    elif not text_input.strip():
        st.warning("Vui lòng không để trống phần văn bản.")
    else:
        with st.spinner("Đang xử lý âm thanh từ Gemini..."):
            try:
                # Khởi tạo Client GenAI
                client = genai.Client(api_key=api_key)
                
                # Cấu hình prompt điều hướng kèm theo text
                full_content = f"Read the following text. Style guideline: {style_prompt}. Text: {text_input}"
                
                # Gọi API bằng model chuyên dụng cho Speech của Gemini
                response = client.models.generate_content(
                    model="gemini-2.5-flash", # Hoặc gemini-3.1-flash-tts-preview
                    contents=full_content,
                    config=types.GenerateContentConfig(
                        response_modalities=["AUDIO"],
                        speech_config=types.SpeechConfig(
                            voice_config=types.VoiceConfig(
                                prebuilt_voice_config=types.PrebuiltVoiceConfig(
                                    voice_name=voice_option
                                )
                            )
                        )
                    )
                )
                
                # Trích xuất dữ liệu âm thanh trả về từ API (Định dạng mặc định là PCM/L16 phát được dưới dạng file raw/wav)
                audio_parts = response.candidates[0].content.parts
                audio_bytes = None
                
                for part in audio_parts:
                    if part.inline_data:
                        audio_bytes = part.inline_data.data
                        break
                
                if audio_bytes:
                    st.success("🎉 Chuyển đổi thành công!")
                    
                    # Hiển thị trình phát nhạc ngay trên giao diện web
                    # Lưu ý: Thư viện phát mặc định đọc luồng dữ liệu từ API
                    st.audio(audio_bytes, format="audio/wav")
                    
                    # Nút tải file xuống máy
                    st.download_button(
                        label="📥 Tải file âm thanh (.wav)",
                        data=audio_bytes,
                        file_name="gemini_tts_output.wav",
                        mime="audio/wav"
                    )
                else:
                    st.error("Không nhận được dữ liệu âm thanh hợp lệ từ mô hình.")
                    
            except Exception as e:
                st.error(f"Đã xảy ra lỗi trong quá trình gọi API: {str(e)}")
                                      
