

import os, json, smtplib, re   # ★ re 추가 (음성용 텍스트 정리)
import pandas as pd
import streamlit as st
from openai import OpenAI
from email.mime.text import MIMEText

# ── 기본 설정 ─────────────────────────────────────
# 키는 환경변수에서 자동
client = OpenAI()

# 이메일 요청 시 받을 주소 (음성 인식 오류를 피하려고 코드에 고정)
MY_EMAIL = "spider379@naver.com"

# ★ [화면] 에이블 대표 이미지 경로 (app.py와 같은 폴더에 둔 파일 이름)
IMAGE_PATH = "able.png"

# ★ [추가 기능 2] 대화 기록 최대 개수 (질문 1 + 답변 1 = 2개 → 10이면 최근 5번의 대화)
MAX_HISTORY = 10

# ★ [추가 기능 3] 음성용 텍스트 정리: 마크다운 기호를 빼야 TTS가 매끄럽게 읽음
def to_speech_text(text):
    text = re.sub(r"[#*`>|]", "", text)              # 제목·굵게 등 기호 제거
    text = re.sub(r"^\s*-\s*", "", text, flags=re.M)  # 줄 앞 목록 기호 '-' 제거
    return text[:4000]                                # TTS 입력 길이 한도

# 에이블 역할
sys_role = """
당신은 '김 대리'의 개인 비서 '에이블'입니다.
김 대리가 회의에 참석하지 못했을 때, 회의 내용을 정리해 업무를 놓치지 않도록 돕습니다.

[주요 업무]
1. 회의록을 분석하고 핵심 내용을 요약합니다.
2. 회의록에서 김 대리에게 배정된 업무 지시사항을 찾아 자세히 정리합니다.
3. 김 대리의 다양한 질문에 답변합니다.

[답변 규칙]
- 한국어로 정중하고 간결하게 답변합니다.
- 회의록에 없는 내용은 지어내지 않습니다.
- 요약을 주제 → 논의 내용 → 결정 사항 순서로 구역을 나눠서 진행합니다.
- 김 대리 담당 업무는 요약과 별도로 정리하고, 업무마다 업무명, 세부 내용, 기한을 함께 적습니다. 기한이 언급되지 않은 업무는 '기한 미정'으로 표시합니다.
- 결정 사항에는 예산, 일정, 채널처럼 구체적인 수치와 항목을 포함합니다.
- 업무 세부 내용에는 회의에서 언급된 방향과 조건을 포함합니다.
- 기한은 회의에서 해당 업무에 대해 명시적으로 언급된 경우에만 적습니다. 추측하지 않습니다.
- 이메일은 김 대리가 요청한 경우에만 보내고 이메일 본문은 마크다운 기호 없이 작성합니다.
- 이메일 요청이 있으면 반드시 send_email 도구를 사용해 실제로 보냅니다. 도구를 쓰지 않고 보냈다고 말하지 않습니다.
- 받는 사람 주소는 [김 대리 이메일 주소]에 적힌 값만 사용합니다.
"""

# ── 이메일 발송 함수 (미션 1) ──
def send_email(to, subject, body):
    sender = os.getenv("GMAIL_ADDRESS")
    password = os.getenv("GMAIL_APP_PASSWORD")

    # Secrets에 값이 없으면 연결 전에 바로 안내
    if not sender or not password:
        return {"status": "failed",
                "error": "GMAIL_ADDRESS 또는 GMAIL_APP_PASSWORD가 설정되지 않았습니다."}

    msg = MIMEText(body, "plain", "utf-8")   # 한글 본문 깨짐 방지
    msg["Subject"] = subject
    msg["From"] = sender
    msg["To"] = to

    try:
        # 465 포트: 처음부터 SSL로 연결 (클라우드 서버에서 더 안정적)
        with smtplib.SMTP_SSL("smtp.gmail.com", 465, timeout=20) as s:
            s.login(sender, password)
            s.sendmail(sender, to, msg.as_string())
        return {"status": "success"}
    except Exception as e:
        return {"status": "failed", "error": str(e)}

# ── 도구 정의 (미션 7)  ──
# description: LLM이 도구를 언제·어떻게 쓸지 판단하는 기준
tools = [{
    "type": "function",
    "name": "send_email",
    "description": "이메일을 발송합니다. 요청받은 내용은 한 통의 이메일로 모아서 보냅니다.",
    "parameters": {
        "type": "object",
        "properties": {
            "to": {"type": "string",
                   "description": "이메일 수신자. 예: 'test@gmail.com'"},
            "subject": {"type": "string",
                        "description": "이메일 제목. 예: '마케팅 전략 초안'"},
            "body": {"type": "string",
                     "description": "이메일 본문. 회의 요약과 김 대리 담당 업무를 모두 포함하고, 마크다운 기호(#, *, -) 없이 일반 텍스트로 작성. 예: '안녕하세요, 김 대리님. 비서 에이블입니다.'"}
        },
        "required": ["to", "subject", "body"],
        "additionalProperties": False
    },
    "strict": True
}]

# ── 업무 우선순위 분석용 출력 스키마 (05 실습 JSON 구조화) ──
priority_schema = {
    "type": "json_schema",
    "name": "priority",
    "schema": {
        "type": "object",
        "properties": {
            "업무목록": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "업무명": {"type": "string"},
                        "기한": {"type": ["string", "null"]},
                        "긴급도": {"type": "string", "enum": ["높음", "낮음"]},
                        "중요도": {"type": "string", "enum": ["높음", "낮음"]},
                        "근거": {"type": "string"}
                    },
                    "required": ["업무명", "기한", "긴급도", "중요도", "근거"],
                    "additionalProperties": False
                }
            },
            "참석자별업무": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "이름": {"type": "string"},
                        "업무수": {"type": "integer"}
                    },
                    "required": ["이름", "업무수"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["업무목록", "참석자별업무"],
        "additionalProperties": False
    }
}

# 우선순위 분석 프롬프트 템플릿
priority_template = """
다음 회의록을 분석하세요.

1. 김 대리에게 배정된 업무를 모두 추출하세요.
   - 기한은 회의록에서 해당 업무에 명시된 표현 그대로 적고, 없으면 null로 하세요.
   - 긴급도: 기한이 가깝거나 회의에서 서둘러야 한다고 언급되면 '높음', 아니면 '낮음'
   - 중요도: 신제품 출시와 매출에 직접 영향을 주면 '높음', 보조 업무면 '낮음'
   - 근거: 판단 이유를 회의록 내용에 근거해 한 문장으로 쓰세요.
2. 회의 참석자(김 대리 포함)별로 새로 배정된 업무 수를 세세요.
   업무를 전달하거나 회의록을 정리하는 일은 업무로 세지 않습니다.

회의록에 없는 내용은 지어내지 마세요.

[회의록]
{meeting}
"""

# 긴급도 × 중요도 → 분류 (파이썬으로 정해서 결과 일관되게)
def classify(urgent, important):
    if urgent == "높음" and important == "높음":
        return "즉시 착수"
    elif important == "높음":
        return "순차 진행"
    else:
        return "위임 검토"

# ── 화면: 제목 ──
st.set_page_config(page_title="AI 개인비서 에이블", page_icon="🤖")
st.title("🤖 AI 개인비서 에이블")
st.caption("음성으로 질문하시면 답변해 드립니다.")

# ★ [화면] 소개 영역: 왼쪽 이미지 / 오른쪽 역할·핵심 기능
left, right = st.columns([1, 2])
with left:
    if os.path.exists(IMAGE_PATH):
        st.image(IMAGE_PATH, width="stretch")   # 칸 너비에 맞춤
    else:
        st.info("🖼️ 이미지 준비 중\n\n(IMAGE_PATH 경로에 파일을 넣으세요)")
with right:
    st.markdown("""
#### 김 대리의 AI 개인비서, 에이블
- 📝 회의록을 분석해 **요약하고 내 업무를 찾아** 드립니다.
- 🎙️ **음성으로 질문**하면 글과 음성으로 답하고, 요청 시 **이메일**로 보냅니다.
- 📅 업무 **우선순위**를 정리하고 마감일을 **캘린더**에 등록합니다.
""")

st.divider()

# ── 세션 상태 초기화 (S04: 처음 한 번만) ──
if "messages" not in st.session_state:
    st.session_state.messages = []        # 대화 히스토리
if "meeting_text" not in st.session_state:
    st.session_state.meeting_text = ""    # 업로드한 회의록 (탭 전환 시 유지)

# ★ [화면] 기능별 탭 (제목을 누르면 해당 기능만 보임)
tab1, tab2, tab3, tab4, tab5 = st.tabs(["🎙️ AI 비서 (회의록·음성 질문)", "📊 업무 우선순위 분석", "📅 마감 캘린더",
                                        "📨 팔로업 메일", "🎧 출근길 브리핑"])

# ════════════════ 탭 1: 회의록 업로드 · 음성 질문 · AI 답변 · 대화 기록 ════════════════
with tab1:
    # ── 화면: 회의록 업로드  ──
    uploaded_file = st.file_uploader("회의록 파일 업로드 (txt)", type=["txt"])

    # 새 파일이 올라오면 세션 상태에 저장 → 다른 탭에 갔다 와도 유지됨
    if uploaded_file is not None:
        st.session_state.meeting_text = uploaded_file.read().decode("utf-8")

    # 아래 코드는 모두 meeting_text 하나로 사용
    meeting_text = st.session_state.meeting_text

    if meeting_text == "":
        st.info("회의록 파일을 업로드하면 회의 내용을 바탕으로 답변합니다.")
    else:
        st.success("회의록이 준비되었습니다.")
        st.text_area("회의록 내용", meeting_text, height=150)

    # ── 화면: 음성 녹음 + 실행 버튼 ──
    audio_value = st.audio_input("무엇을 도와드릴까요?")
    voice_on = st.toggle("🔊 답변을 음성으로 듣기", value=True)   # ★ [추가 기능 3] 음성 켜기/끄기

    if st.button("실행"):
        if audio_value is None:
            st.warning("먼저 마이크 버튼을 눌러 질문을 녹음해 주세요.")
        else:
            try:
                # 시간이 걸리는 작업 동안 스피너 표시
                with st.spinner("답변을 준비하고 있습니다.."):

                    # STT: 녹음은 파일이 아니라 메모리 데이터 → (파일명, 바이트)로 전달
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=("recording.wav", audio_value.getvalue())
                    )

                    # 프롬프트: 음성 요청 + 고정 이메일 주소 + 회의록
                    prompt = f"""
{transcript.text}

[김 대리 이메일 주소]
{MY_EMAIL}

[미팅 내용]
{meeting_text}
"""

                    # 음성 요청에 '메일'이 들어 있으면 이메일 도구를 반드시 사용
                    # ('이 메일', '이메일' 모두 잡히도록 '메일'로 검사)
                    wants_email = "메일" in transcript.text
                    if wants_email:
                        choice = {"type": "function", "name": "send_email"}
                    else:
                        choice = "auto"

                    # 1단계: 질문 + 도구 정의 전달
                    # parallel False: 메일 한 통만
                    response = client.responses.create(
                        model="gpt-4o-mini",
                        # ★ [추가 기능 1] 멀티턴: 역할 → 이전 대화 → 이번 질문 순서로 전달
                        input=[{"role": "system", "content": sys_role}]
                              + st.session_state.messages
                              + [{"role": "user", "content": prompt}],
                        tools=tools,
                        tool_choice=choice,
                        parallel_tool_calls=False,
                        temperature=0
                    )

                    # 도구 호출 요청이 있는지 확인
                    tool_calls = [item for item in response.output
                                  if item.type == "function_call"]

                    if not tool_calls:
                        answer = response.output_text
                    else:
                        # 2단계: LLM이 정한 인자(to, subject, body)로 함수 실행
                        tool_outputs = []
                        for item in tool_calls:
                            if item.name == "send_email":
                                args = json.loads(item.arguments)
                                result = send_email(**args)
                                tool_outputs.append({
                                    "type": "function_call_output",
                                    "call_id": item.call_id,
                                    "output": json.dumps(result, ensure_ascii=False)
                                })

                        # 3단계: 실행 결과를 LLM에 돌려주고 최종 답변 받기
                        final_response = client.responses.create(
                            model="gpt-4o-mini",
                            previous_response_id=response.id,
                            input=tool_outputs,
                            temperature=0
                        )
                        answer = final_response.output_text

                # ── 결과 화면 ──────────────────────────
                st.markdown("#### 🎙️ 인식된 질문")
                st.info(transcript.text)   # 음성 인식이 정확한지 확인용

                if tool_calls:   # 메일을 보냈을 때만 발송 결과 표시
                    if result["status"] == "success":
                        st.success(f"📧 {args['to']}(으)로 이메일을 보냈습니다.")
                    else:
                        st.error(f"이메일 발송 실패: {result.get('error')}")

                st.divider()
                st.markdown("### AI 답변")
                st.markdown(answer)   # 마크다운(###, **)을 서식으로 변환해 표시

                # ★ [추가 기능 3] LLM 최종 응답을 음성으로 즉시 재생 (미션 5·6 TTS)
                if voice_on:
                    with st.spinner("음성을 준비하고 있습니다.."):
                        speech = client.audio.speech.create(
                            model="gpt-4o-mini-tts",
                            voice="nova",
                            input=to_speech_text(answer),
                            instructions="비서가 담당자에게 얘기하는 정중한 말투로 빠뜨리지 말고 또박또박 읽어주세요."
                        )
                    st.audio(speech.content, format="audio/mp3", autoplay=True)

                # 대화 히스토리 저장 (질문·답변 한 쌍씩, 최근 10개만 유지)
                st.session_state.messages.append({"role": "user", "content": transcript.text})
                st.session_state.messages.append({"role": "assistant", "content": answer})
                st.session_state.messages = st.session_state.messages[-MAX_HISTORY:]   # ★ 숫자를 변수로

            except Exception as e:
                # API·네트워크 오류 시 앱이 멈추지 않고 안내만 표시 (종합실습6)
                st.error(f"오류가 발생했습니다: {e}")

    # ── 화면: 대화 기록 (종합실습6 st.chat_message) ──
    if st.session_state.messages:
        with st.expander(f"💬 대화 기록 ({len(st.session_state.messages) // 2}건)"):
            for message in st.session_state.messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])
        # ★ [추가 기능 2] 대화 기록 지우기 (새 회의록으로 바꿀 때 이전 대화가 섞이지 않게)
        if st.button("🗑️ 대화 기록 지우기"):
            st.session_state.messages = []
            st.rerun()   # 화면을 바로 다시 그려 기록을 지움

# ════════════════ 탭 2: 업무 우선순위 분석 ════════════════
with tab2:
    st.subheader("📊 업무 우선순위 분석")
    st.caption("업로드한 회의록에서 김 대리 업무를 긴급도 × 중요도로 분류합니다.")

    if st.button("우선순위 분석"):
        if meeting_text == "":
            st.warning("먼저 회의록 파일을 업로드해 주세요.")
        else:
            try:
                with st.spinner("업무를 분석하고 있습니다.."):
                    response = client.responses.create(
                        model="gpt-4o-mini",
                        input=[{"role": "user",
                                "content": priority_template.format(meeting=meeting_text)}],
                        text={"format": priority_schema},   # 정해진 JSON 형식으로만 답변
                        temperature=0
                    )
                    data = json.loads(response.output_text)   # JSON 문자열 → 딕셔너리

                # ── ① 업무 집중도 (R&R 밸런스) ──────────
                df_people = pd.DataFrame(data["참석자별업무"])
                kim = df_people[df_people["이름"].str.contains("김")]["업무수"].sum()
                total = df_people["업무수"].sum()

                st.markdown("#### ⚖️ 업무 집중도")
                col1, col2 = st.columns(2)
                col1.metric("김 대리 배정 업무", f"{kim}건")
                col2.metric("전체 배정 업무 중 비중",
                            f"{kim / total * 100:.0f}%" if total else "-")

                # 김 대리 업무가 3건 이상이거나 절반 이상이면 경보
                if kim >= 3 or (total and kim / total >= 0.5):
                    st.error("🚨 김 대리에게 업무가 집중되어 있습니다. 우선순위와 위임을 검토하세요.")
                else:
                    st.success("업무 배분이 적정한 수준입니다.")

                st.bar_chart(df_people.set_index("이름")["업무수"])

                # ── ② 우선순위 매트릭스 ──────────────────
                df_task = pd.DataFrame(data["업무목록"])
                df_task["기한"] = df_task["기한"].fillna("기한 미정").replace("null", "기한 미정")
                df_task["분류"] = df_task.apply(
                    lambda r: classify(r["긴급도"], r["중요도"]), axis=1)

                st.markdown("#### 🗂️ 우선순위 가이드")
                cols = st.columns(3)
                for col, group, icon in zip(cols,
                                            ["즉시 착수", "순차 진행", "위임 검토"],
                                            ["🔴", "🟡", "🔵"]):
                    col.markdown(f"**{icon} {group}**")
                    tasks = df_task[df_task["분류"] == group]
                    if tasks.empty:
                        col.caption("해당 업무 없음")
                    for _, t in tasks.iterrows():
                        col.info(f"**{t['업무명']}**\n\n⏰ {t['기한']}")

                # ── ③ 판단 근거 표 ───────────────────────
                st.markdown("#### 📋 판단 근거")
                st.dataframe(df_task[["분류", "업무명", "기한", "긴급도", "중요도", "근거"]],
                             hide_index=True)
                st.caption("※ 중요도는 회의 내용을 바탕으로 한 에이블의 판단입니다.")

            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

# ══════════════════════════════════════════════════════════════
# 추가 기능 2: 마감 기한 역산 캘린더 (Action Scheduler)
#  - 이 셀은 app.py 끝에 "이어 붙이기"(-a) 합니다.
#  - ⚠️ app.py 셀(%%writefile app.py)을 실행한 직후, 이 셀을 한 번만 실행하세요.
# ══════════════════════════════════════════════════════════════
from datetime import date, timedelta
import re

# ── ① 일정 추출용 출력 스키마 (05 실습 JSON 구조화) ──
schedule_schema = {
    "type": "json_schema",
    "name": "schedule",
    "schema": {
        "type": "object",
        "properties": {
            "김대리업무": {                      # 김 대리 업무 목록 (기한 없어도 포함)
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "일정명": {"type": "string"},
                        "기한표현": {"type": ["string", "null"]},
                        "기한일수": {"type": ["integer", "null"]}
                    },
                    "required": ["일정명", "기한표현", "기한일수"],
                    "additionalProperties": False
                }
            },
            "주요일정": {                        # 출시일 같은 전체 일정
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "일정명": {"type": "string"},
                        "기한표현": {"type": ["string", "null"]},
                        "기한일수": {"type": ["integer", "null"]}
                    },
                    "required": ["일정명", "기한표현", "기한일수"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["김대리업무", "주요일정"],
        "additionalProperties": False
    }
}

schedule_template = """
다음 회의록에서 일정을 추출하세요.

[김대리업무]
- 김 대리에게 배정된 업무를 업무 하나당 한 항목씩 빠짐없이 모두 적으세요.
- 기한이 없는 업무도 반드시 포함하세요. (기한표현, 기한일수는 null)
- 일정명은 회의록의 구체적인 업무명으로 쓰세요.
- 보고서 제출처럼 업무의 결과물 기한은 별도 항목으로 만들지 말고, 그 업무의 기한으로 적으세요.

[주요일정]
- 신제품 출시일처럼 회의에서 시점이 정해진 전체 일정만 적으세요.
- 다음 회의, 진행 상황 점검처럼 회의 운영 일정은 제외하세요.
- 김대리업무에 적은 업무는 다시 적지 마세요.

[기한 규칙]
- 기한표현: 회의록에 나온 시한 표현을 그대로 적고, 없으면 null로 하세요.
- 기한일수: 회의 날로부터 며칠 뒤인지 정수로 적으세요.
  '다음 주까지'는 7, '2주 안에'는 14처럼 회의 날 기준으로 정할 수 있을 때만 적습니다.
  '다음 달 초'처럼 정하기 어렵거나, '출시 2주 전'처럼 다른 일정을 기준으로 한 시점이거나,
  기한이 없으면 null로 하세요.

회의록에 없는 내용은 지어내지 마세요.

[회의록]
{meeting}
"""

# ── ② 날짜 계산 (계산은 파이썬이 담당 → 결과가 항상 일관됨) ──
def calc_due(base, expr, days):
    """기준일 + 기한일수로 마감일 계산. '다음 달'이면 다음 달 1일. 못 정하면 None"""
    if days is not None:
        return base + timedelta(days=days)
    if expr and "다음 달" in expr:
        return (base.replace(day=1) + timedelta(days=32)).replace(day=1)
    return None

def make_milestones(name, due, base):
    """마감일 기준 D-14, D-7, D-1, D-day 생성 (기준일보다 이전 날짜는 제외)"""
    rows = []
    for label, before in [("D-14", 14), ("D-7", 7), ("D-1", 1), ("D-day", 0)]:
        day = due - timedelta(days=before)
        if day >= base:
            rows.append({"날짜": day, "단계": label, "일정명": name,
                         "제목": f"[{label}] {name}"})
    return rows

# ── ③ 캘린더 파일(.ics) 만들기 → 어떤 캘린더에서도 열면 바로 등록 ──
def make_ics(rows):
    lines = ["BEGIN:VCALENDAR", "VERSION:2.0", "PRODID:-//Able//Action Scheduler//KO"]
    for i, r in enumerate(rows):
        lines += ["BEGIN:VEVENT",
                  f"UID:able-{r['날짜']:%Y%m%d}-{i}@able",
                  f"DTSTAMP:{date.today():%Y%m%d}T000000Z",
                  f"DTSTART;VALUE=DATE:{r['날짜']:%Y%m%d}",
                  f"DTEND;VALUE=DATE:{r['날짜'] + timedelta(days=1):%Y%m%d}",
                  f"SUMMARY:{r['제목']}",
                  "END:VEVENT"]
    lines.append("END:VCALENDAR")
    return "\r\n".join(lines)

# ── ④ Google 캘린더 연결 (서비스 계정 + 내 캘린더 공유) ──
# 환경변수 GOOGLE_TOKEN: service_account.json 내용 / GOOGLE_CALENDAR_ID: 내 캘린더 ID
#   - Colab: api_key.txt 셀에서 등록
#   - Streamlit Cloud: Secrets에 등록
def get_calendar_service():
    from googleapiclient.discovery import build
    info = json.loads(os.getenv("GOOGLE_TOKEN"))
    if info.get("type") == "service_account":      # 서비스 계정 키를 쓴 경우
        from google.oauth2.service_account import Credentials
        creds = Credentials.from_service_account_info(
            info, scopes=["https://www.googleapis.com/auth/calendar"])
    else:                                           # OAuth로 받은 token.json
        from google.oauth2.credentials import Credentials
        creds = Credentials.from_authorized_user_info(info)
    return build("calendar", "v3", credentials=creds)

def add_to_google(rows):
    service = get_calendar_service()
    calendar_id = os.getenv("GOOGLE_CALENDAR_ID", "primary")   # 기본: 본인 캘린더
    for r in rows:
        event = {
            "summary": r["제목"],
            "description": "AI 개인비서 에이블이 회의록 기한을 역산해 등록한 일정입니다.",
            "start": {"date": r["날짜"].isoformat()},
            "end": {"date": (r["날짜"] + timedelta(days=1)).isoformat()},   # 종일 일정
        }
        service.events().insert(calendarId=calendar_id, body=event).execute()
    return len(rows)

# ── ⑤ 화면: 탭 3 안에 표시 (탭은 app.py 셀에서 만듦) ──
with tab3:
    # ── ⑤ 화면 ───────────────────────────────────────
    st.subheader("📅 마감 기한 역산 캘린더")
    st.caption("회의록의 시한을 날짜로 계산해 D-14, D-7, D-1 마일스톤으로 등록합니다.")

    if "schedule" not in st.session_state:
        st.session_state.schedule = []      # 추출한 일정 (버튼을 눌러도 유지)

    base_day = st.date_input("회의 날짜 (기한 계산 기준일)", value=date.today())

    if st.button("일정 추출"):
        if meeting_text == "":
            st.warning("먼저 회의록 파일을 업로드해 주세요.")
        else:
            try:
                with st.spinner("회의록에서 일정을 찾고 있습니다.."):
                    response = client.responses.create(
                        model="gpt-4o-mini",
                        input=[{"role": "user",
                                "content": schedule_template.format(meeting=meeting_text)}],
                        text={"format": schedule_schema},
                        temperature=0
                    )
                    data = json.loads(response.output_text)

                # 이전에 수정했던 체크박스·날짜 값 지우기 (새 결과로 다시 채우기 위해)
                for key in list(st.session_state.keys()):
                    if key.startswith(("use_", "due_")):
                        del st.session_state[key]

                # 두 목록을 하나로 합치면서 구분 표시 (김 대리 업무가 먼저 나오게)
                schedule = []
                for item in data["김대리업무"]:
                    schedule.append({**item, "구분": "김 대리 업무"})
                for item in data["주요일정"]:
                    schedule.append({**item, "구분": "주요 일정"})
                st.session_state.schedule = schedule
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

    # 추출 결과가 있으면: 확인·수정 → 마일스톤 → 등록
    if st.session_state.schedule:
        st.markdown("#### ✅ 일정 확인 및 수정")
        st.caption("기한이 없거나 애매한 일정은 날짜를 직접 정한 뒤 체크하세요.")

        all_rows = []
        for i, item in enumerate(st.session_state.schedule):
            # 구분 이름만 있는 빈 항목은 건너뛰기 (AI 출력 검증)
            if item["일정명"] in ("김 대리 업무", "주요 일정"):
                continue
            # "null" 글자도 기한 없음으로 처리
            expr = item["기한표현"] if item["기한표현"] not in (None, "", "null") else None

            days = item["기한일수"]
            # '출시 2주 전'처럼 다른 일정 기준 시점은 AI가 계산한 일수를 쓰지 않음
            if expr and re.search(r"\d+\s*(주|일)\s*전", expr):
                days = None
            due = calc_due(base_day, expr, days)
            expr = expr or "기한 미정"

            c1, c2, c3 = st.columns([1, 3, 2])
            use = c1.checkbox("등록", value=due is not None, key=f"use_{i}")
            c2.markdown(f"**{item['일정명']}**  \n{item['구분']} · 회의 표현: {expr}")
            due = c3.date_input("마감일", value=due or base_day + timedelta(days=7),
                                key=f"due_{i}")

            if use:
                all_rows += make_milestones(item["일정명"], due, base_day)

        if all_rows:
            all_rows.sort(key=lambda r: r["날짜"])
            st.markdown("#### 🗓️ 등록될 마일스톤")
            st.dataframe(pd.DataFrame(all_rows)[["날짜", "단계", "일정명"]], hide_index=True)

            col1, col2 = st.columns(2)

            # 방법 1: Google 캘린더 자동 등록 (토큰이 있을 때만 버튼 활성화)
            has_token = bool(os.getenv("GOOGLE_TOKEN"))
            if col1.button("Google 캘린더에 등록", disabled=not has_token):
                try:
                    with st.spinner("Google 캘린더에 등록하고 있습니다.."):
                        count = add_to_google(all_rows)
                    st.success(f"📅 {count}개 일정을 Google 캘린더에 등록했습니다.")
                except Exception as e:
                    st.error(f"캘린더 등록 실패: {e}")
            if not has_token:
                col1.caption("GOOGLE_TOKEN이 없어 비활성화됨")

            # 방법 2: .ics 파일 다운로드 (어떤 캘린더든 열면 등록됨)
            col2.download_button("📥 캘린더 파일(.ics) 다운로드",
                                 data=make_ics(all_rows),
                                 file_name="able_schedule.ics",
                                 mime="text/calendar")
        else:
            st.info("등록할 일정을 체크해 주세요.")

# ══════════════════════════════════════════════════════════════
# 추가 기능 3: 팔로업 메일 초안 자동 생성 & 원클릭 승인 (탭 4)
#  - ⚠️ app.py 셀 → 캘린더 셀 다음에, 이 셀을 한 번만 실행하세요.
# ══════════════════════════════════════════════════════════════

# ── ① 팔로업 요청 추출 스키마 (05 실습 JSON 구조화) ──
followup_schema = {
    "type": "json_schema",
    "name": "followup",
    "schema": {
        "type": "object",
        "properties": {
            "요청목록": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "담당자": {"type": "string"},      # 메일 받을 회의 참석자
                        "요청자료": {"type": "string"},    # 요청할 자료·정보
                        "관련업무": {"type": "string"},    # 김 대리의 어떤 업무에 필요한지
                        "근거": {"type": "string"},        # 회의록 속 근거 발언
                        "메일제목": {"type": "string"},
                        "메일본문": {"type": "string"}
                    },
                    "required": ["담당자", "요청자료", "관련업무", "근거", "메일제목", "메일본문"],
                    "additionalProperties": False
                }
            }
        },
        "required": ["요청목록"],
        "additionalProperties": False
    }
}

followup_template = """
당신은 김 대리의 비서 에이블입니다. 회의록을 읽고, 김 대리가 배정된 업무를 시작하기 위해
다른 참석자에게 받아야 할 선행 자료나 정보를 찾아 요청 메일 초안을 작성하세요.

[찾는 기준]
- 참석자가 김 대리에게 넘겨주기로 한 자료, 또는 회의에서 그 참석자가 언급해 가장 잘 알고 있는 정보
- 김 대리 업무 수행에 실제로 필요한 것만, 최대 3건
- 근거에는 회의록에서 그 참석자가 한 발언을 그대로 적으세요.
- 회의록에 근거가 없는 요청은 만들지 마세요.

[메일 작성 규칙]
- 김 대리가 해당 참석자에게 보내는 메일입니다. 받는 사람 직함에 '님'을 붙여 정중하게 쓰세요.
- 무엇이, 왜(어떤 업무에 필요한지), 언제까지 필요한지를 포함하세요. 기한이 회의록에 없으면 '가능한 일정'을 여쭙는 표현으로 쓰세요.
- 마크다운 기호 없이 일반 텍스트로, 5~8문장 이내로 쓰세요.
- 마지막은 '김 대리 드림'으로 끝내세요.

[회의록]
{meeting}
"""

# ── ② 화면: 탭 4 ──
with tab4:
    st.subheader("📨 팔로업 메일 초안")
    st.caption("회의에서 동료에게 받아야 할 자료를 찾아 요청 메일 초안을 만들고, 승인하면 발송합니다.")

    if "followups" not in st.session_state:
        st.session_state.followups = []      # 생성한 초안 목록 (버튼을 눌러도 유지)

    if st.button("초안 만들기", key="fu_make"):
        if meeting_text == "":
            st.warning("먼저 'AI 비서' 탭에서 회의록 파일을 업로드해 주세요.")
        else:
            try:
                with st.spinner("필요한 자료를 찾고 메일 초안을 쓰고 있습니다.."):
                    response = client.responses.create(
                        model="gpt-4o-mini",
                        input=[{"role": "user",
                                "content": followup_template.format(meeting=meeting_text)}],
                        text={"format": followup_schema},
                        temperature=0
                    )
                    data = json.loads(response.output_text)

                # 이전 초안의 입력칸·발송 기록 지우기 (새 초안으로 다시 채우기 위해)
                for key in list(st.session_state.keys()):
                    if key.startswith(("fu_to_", "fu_sub_", "fu_body_", "fu_sent_")):
                        del st.session_state[key]

                st.session_state.followups = data["요청목록"]
                if not data["요청목록"]:
                    st.info("회의록에서 요청할 선행 자료를 찾지 못했습니다.")
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

    # 초안마다: 근거 확인 → 수정 → 승인 버튼
    for i, d in enumerate(st.session_state.followups):
        st.divider()
        st.markdown(f"**📌 {d['담당자']}님께 '{d['요청자료']}' 요청 메일 초안을 작성했습니다. 발송할까요?**")
        st.caption(f"관련 업무: {d['관련업무']}  \n회의 근거: \"{d['근거']}\"")

        # 동료 실제 주소가 없으므로 기본값은 시연용으로 김 대리 주소 (직접 수정 가능)
        to = st.text_input("받는 사람", value=MY_EMAIL, key=f"fu_to_{i}",
                           help="시연용으로 김 대리 주소가 들어 있습니다. 실제 동료 주소로 바꿔 보내세요.")
        subject = st.text_input("제목", value=d["메일제목"], key=f"fu_sub_{i}")
        body = st.text_area("본문 (수정 가능)", value=d["메일본문"], height=200, key=f"fu_body_{i}")

        if st.session_state.get(f"fu_sent_{i}"):
            st.success(f"📧 {st.session_state[f'fu_sent_{i}']}(으)로 발송했습니다.")
        elif st.button("✅ 승인하고 발송", key=f"fu_send_{i}"):
            if "@" not in to:
                st.warning("받는 사람 이메일 주소를 확인해 주세요.")
            else:
                with st.spinner("메일을 보내고 있습니다.."):
                    result = send_email(to, subject, body)   # 미션 1 함수 재사용
                if result["status"] == "success":
                    st.session_state[f"fu_sent_{i}"] = to      # 중복 발송 방지
                    st.rerun()
                else:
                    st.error(f"이메일 발송 실패: {result.get('error')}")

# ══════════════════════════════════════════════════════════════
# 추가 기능 4: 1분 출근길 오디오 브리핑 (탭 5)
#  - ⚠️ 팔로업 메일 셀 다음에, 이 셀을 한 번만 실행하세요.
# ══════════════════════════════════════════════════════════════

briefing_role = """
당신은 김 대리의 비서 에이블이며, 출근길에 이어폰으로 듣는 1분 라디오 브리핑을 진행합니다.

[형식]
- 뉴스 앵커처럼 또렷하고 경쾌한 말투로, 말하듯 자연스러운 문장으로 씁니다.
- 마크다운 기호, 목록, 제목, 괄호를 쓰지 않습니다.
- 전체 250~300자 이내로 씁니다. (읽으면 약 1분)

[순서]
1. 짧은 인사: "김 대리님, 출근길 브리핑입니다."
2. 회의 결론 한두 문장: 무엇을 하기로 했는지
3. 먼저 챙길 일: 김 대리 업무를 기한이 가까운 순서로, 업무 개수를 먼저 말하고 '첫째, 둘째'로
4. 짧은 마무리 인사

[주의]
- 회의록에 없는 내용은 지어내지 않습니다.
- 기한은 회의에서 명시된 표현 그대로 말하고, 없으면 '기한은 정해지지 않았습니다'라고 말합니다.
"""

# ── 화면: 탭 5 ──
with tab5:
    st.subheader("🎧 1분 출근길 브리핑")
    st.caption("회의 결론과 먼저 챙길 일만 1분 안팎의 음성으로 들려 드립니다.")

    if "briefing" not in st.session_state:
        st.session_state.briefing = None      # {"script": 원고, "audio": 음성} 저장

    voice = st.selectbox("목소리", ["nova", "alloy", "shimmer", "onyx"], key="br_voice")

    if st.button("브리핑 만들기", key="br_make"):
        if meeting_text == "":
            st.warning("먼저 'AI 비서' 탭에서 회의록 파일을 업로드해 주세요.")
        else:
            try:
                with st.spinner("브리핑 원고를 쓰고 있습니다.."):
                    response = client.responses.create(
                        model="gpt-4o-mini",
                        input=[{"role": "system", "content": briefing_role},
                               {"role": "user", "content": f"[회의록]\n{meeting_text}"}],
                        temperature=0
                    )
                    script = response.output_text

                with st.spinner("음성으로 변환하고 있습니다.."):
                    speech = client.audio.speech.create(
                        model="gpt-4o-mini-tts",
                        voice=voice,
                        input=script,
                        instructions="라디오 뉴스 앵커처럼 또렷하고 경쾌하게, 조금 빠른 속도로 읽어 주세요."
                    )

                st.session_state.briefing = {"script": script, "audio": speech.content}
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

    # 만든 브리핑이 있으면 재생 + 다운로드 + 원고
    if st.session_state.briefing:
        b = st.session_state.briefing
        st.audio(b["audio"], format="audio/mp3")
        sec = round(len(b["script"]) / 5.5)   # 한국어 약 5~6자/초 기준 예상 길이
        st.caption(f"원고 {len(b['script'])}자 · 예상 재생 시간 약 {sec}초")

        st.download_button("📥 MP3로 저장 (이어폰으로 듣기)", data=b["audio"],
                           file_name="able_briefing.mp3", mime="audio/mpeg")
        with st.expander("📄 브리핑 원고 보기"):
            st.write(b["script"])
