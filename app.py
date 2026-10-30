
import os
import base64
import streamlit as st
from openai import OpenAI

# OpenAI 클라이언트 생성
client = OpenAI()

# 페이지 설정
st.set_page_config(page_title="AI 영어 통역기", page_icon="🌐")
st.title("🌐 AI 영어 통역기")

# 텍스트 입력
text = st.text_area(
    "텍스트를 입력하면 영어로 통역해 드립니다.",
    "이 곳에 영어로 통역할 문장을 입력하세요. 어떤 언어든 상관이 없습니다."
)

# 버튼
if st.button("영어로 통역하기"):
    if text.strip() == "":
        st.warning("번역할 텍스트를 입력해주세요.")
    else:
        # 영문 번역
        prompt = f"""
        다음 문장을 영문으로 번역한 후 번역된 문장만 출력하세요.
        ---
        "{text}"
        """

        response = client.responses.create(
            model="gpt-4o-mini",
            input=prompt
        )

        st.markdown("### 💬 번역 결과")
        st.write(response.output_text)

        # TTS 생성
        speech = client.audio.speech.create(
            model="tts-1",
            voice="alloy",
            input=response.output_text
        )

        audio_bytes = speech.content

        # 음성 출력
        st.markdown("### 🔊 음성 재생")
        st.audio(audio_bytes, format="audio/mp3")
