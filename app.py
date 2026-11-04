import os
import json
import math
import hashlib
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from openai import OpenAI

# ============================================================
# 1. 기본 설정
# ============================================================

st.set_page_config(
    page_title="에이블 AI 업무 비서",
    page_icon="🤖",
    layout="wide"
)

KST = ZoneInfo("Asia/Seoul")
TODAY = datetime.now(KST).date()


def get_config(name, default=None):
    """환경변수 또는 Streamlit Secrets에서 설정 읽기"""
    value = os.getenv(name)
    if value:
        return value

    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


api_key = get_config("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY가 설정되지 않았습니다.")
    st.stop()

client = OpenAI(api_key=api_key)

SYSTEM_PROMPT = """
당신은 김 대리의 개인 AI 업무 비서 '에이블'입니다.

[역할]
1. 회의록의 주요 내용을 요약합니다.
2. 논의 내용과 결정 사항을 구분합니다.
3. 담당자별 업무(R&R)를 정리합니다.
4. 김 대리에게 배정된 업무를 자세히 설명합니다.
5. 긴급도와 중요도를 기준으로 업무 우선순위를 분석합니다.
6. 마감일과 업무량을 고려해 실행 계획을 제안합니다.

[답변 원칙]
- 모든 답변은 한국어로 작성합니다.
- 회의록에 없는 사실을 임의로 생성하지 않습니다.
- 마감일이 없으면 '기한 미정'이라고 표시합니다.
- 담당자가 불명확하면 '담당자 미정'이라고 표시합니다.
- 예상 작업시간과 우선순위 판단은 추정임을 명시합니다.
- Google Calendar 등록은 반드시 사용자 승인 후 진행합니다.
"""

# ============================================================
# 2. 세션 상태 초기화
# ============================================================

DEFAULT_STATE = {
    "messages": [],
    "meeting_text": "",
    "meeting_signature": "",
    "tasks": [],
    "summary": "",
    "transcript": "",
    "audio_answer": None,
    "registered_events": {},
}

for key, value in DEFAULT_STATE.items():
    if key not in st.session_state:
        st.session_state[key] = value


def reset_state(full=False):
    if full:
        for key in DEFAULT_STATE:
            st.session_state[key] = (
                [] if key in ("messages", "tasks")
                else {} if key == "registered_events"
                else None if key == "audio_answer"
                else ""
            )
    else:
        st.session_state.messages = []


# ============================================================
# 3. AI 대화
# ============================================================

def ask_ai(question):
    context = (
        SYSTEM_PROMPT
        + f"\n현재 날짜: {TODAY.isoformat()}"
        + "\n[회의록]\n"
        + (
            st.session_state.meeting_text
            if st.session_state.meeting_text
            else "회의록 없음"
        )
    )

    messages = [
        {"role": "system", "content": context}
    ]

    messages.extend(st.session_state.messages[-20:])

    messages.append({
        "role": "user",
        "content": question
    })

    response = client.responses.create(
        model="gpt-4o-mini",
        input=messages,
        temperature=0
    )

    answer = response.output_text

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    return answer


# ============================================================
# 4. 업무 추출
# ============================================================

TASK_TOOL = [{
    "type": "function",
    "name": "extract_tasks",
    "description": "회의록에서 업무와 담당자, 마감일, 중요도를 추출합니다.",
    "parameters": {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "task_name": {"type": "string"},
                        "description": {"type": "string"},
                        "owner": {"type": "string"},
                        "deadline": {
                            "type": ["string", "null"]
                        },
                        "urgent": {"type": "boolean"},
                        "important": {"type": "boolean"},
                        "reason": {"type": "string"},
                        "estimated_hours": {
                            "type": ["number", "null"]
                        }
                    },
                    "required": [
                        "task_name",
                        "description",
                        "owner",
                        "deadline",
                        "urgent",
                        "important",
                        "reason",
                        "estimated_hours"
                    ],
                    "additionalProperties": False
                }
            }
        },
        "required": ["tasks"],
        "additionalProperties": False
    },
    "strict": True
}]


def extract_tasks(meeting_text):
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": (
                    SYSTEM_PROMPT
                    + f"\n현재 날짜: {TODAY.isoformat()}"
                )
            },
            {
                "role": "user",
                "content": (
                    "다음 회의록에서 명시된 업무를 추출하세요.\n"
                    "담당자나 마감일을 추측하지 마세요.\n"
                    "마감일은 명확히 확인되는 경우에만 "
                    "YYYY-MM-DD로 작성하세요.\n"
                    "소요시간이 명시되지 않았다면 null로 작성하세요.\n"
                    "긴급도와 중요도는 판단 근거를 포함하세요.\n\n"
                    + meeting_text
                )
            }
        ],
        tools=TASK_TOOL,
        tool_choice={
            "type": "function",
            "name": "extract_tasks"
        },
        temperature=0
    )

    for item in response.output:
        if (
            item.type == "function_call"
            and item.name == "extract_tasks"
        ):
            return json.loads(item.arguments).get("tasks", [])

    return []


# ============================================================
# 5. 아이젠하워 매트릭스
# ============================================================

def classify_task(task):
    urgent = task["urgent"]
    important = task["important"]

    if urgent and important:
        return "Q1"
    elif important:
        return "Q2"
    elif urgent:
        return "Q3"
    else:
        return "Q4"


QUADRANTS = {
    "Q1": "🔴 Q1 긴급하고 중요한 업무",
    "Q2": "🔵 Q2 중요하지만 긴급하지 않은 업무",
    "Q3": "🟠 Q3 긴급하지만 중요하지 않은 업무",
    "Q4": "⚪ Q4 긴급하지도 중요하지도 않은 업무"
}


# ============================================================
# 6. 일정 역산
# ============================================================

def parse_deadline(value):
    if not value:
        return None

    try:
        return date.fromisoformat(str(value))
    except (ValueError, TypeError):
        return None


def working_days(start, end):
    days = []
    current = start

    while current <= end:
        if current.weekday() < 5:
            days.append(current)
        current += timedelta(days=1)

    return days


def calculate_schedule(task, daily_capacity):
    deadline = parse_deadline(task.get("deadline"))
    hours = task.get("estimated_hours")

    if deadline is None or hours is None:
        return None

    hours = float(hours)

    if hours <= 0:
        return None

    days_needed = math.ceil(hours / daily_capacity)

    # 주말 마감일이면 직전 평일을 마지막 작업일로 설정
    last_workday = deadline

    while last_workday.weekday() >= 5:
        last_workday -= timedelta(days=1)

    start = last_workday
    counted = 1

    while counted < days_needed:
        start -= timedelta(days=1)

        if start.weekday() < 5:
            counted += 1

    return start, deadline


# ============================================================
# 7. 업무 과부하 분석
# ============================================================

def analyze_workload(tasks, daily_capacity):
    loads = {}
    warnings = []
    unknown = []

    for task in tasks:
        owner = task.get("owner", "")

        if not ("김" in owner and "대리" in owner):
            continue

        schedule = calculate_schedule(task, daily_capacity)

        if schedule is None:
            unknown.append(task["task_name"])
            continue

        start, deadline = schedule

        if deadline < TODAY:
            warnings.append(
                f"'{task['task_name']}'의 마감일이 지났습니다."
            )
            continue

        days = working_days(max(start, TODAY), deadline)

        if not days:
            warnings.append(
                f"'{task['task_name']}'의 남은 평일 작업일이 없습니다."
            )
            continue

        hours_per_day = (
            float(task["estimated_hours"]) / len(days)
        )

        for day in days:
            loads[day] = loads.get(day, 0) + hours_per_day

    for day, hours in sorted(loads.items()):
        if hours > daily_capacity:
            warnings.append(
                f"{day}: 예상 {hours:.1f}시간 / "
                f"가능 {daily_capacity}시간"
            )

    return loads, warnings, unknown


# ============================================================
# 8. Google Calendar
# ============================================================

def get_calendar_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    service_file = get_config("GOOGLE_SERVICE_ACCOUNT_FILE")

    if service_file:
        credentials = (
            service_account.Credentials.from_service_account_file(
                service_file,
                scopes=[
                    "https://www.googleapis.com/auth/calendar.events"
                ]
            )
        )
    else:
        info = get_config("gcp_service_account")

        if not info:
            raise RuntimeError(
                "Google Calendar 서비스 계정이 설정되지 않았습니다."
            )

        credentials = (
            service_account.Credentials.from_service_account_info(
                dict(info),
                scopes=[
                    "https://www.googleapis.com/auth/calendar.events"
                ]
            )
        )

    return build(
        "calendar",
        "v3",
        credentials=credentials,
        cache_discovery=False
    )


def register_calendar(task, start, deadline):
    calendar_id = get_config("calendar_id") or get_config(
        "GOOGLE_CALENDAR_ID"
    )

    if not calendar_id:
        raise RuntimeError("Google Calendar ID가 없습니다.")

    body = {
        "summary": "[에이블] " + task["task_name"],
        "description": (
            f"담당자: {task['owner']}\n"
            f"업무 내용: {task['description']}\n"
            f"우선순위: {classify_task(task)}\n"
            f"예상 소요시간: {task['estimated_hours']}시간\n"
            f"마감일: {deadline}\n"
            "권장 시작일은 예상 업무시간을 기준으로 계산했습니다."
        ),
        "start": {
            "date": start.isoformat()
        },
        "end": {
            "date": (deadline + timedelta(days=1)).isoformat()
        }
    }

    service = get_calendar_service()

    return service.events().insert(
        calendarId=calendar_id,
        body=body
    ).execute()


# ============================================================
# 9. 사이드바
# ============================================================

with st.sidebar:
    st.header("⚙️ 업무 관리 설정")

    daily_capacity = st.slider(
        "하루 업무 가능 시간",
        min_value=1,
        max_value=12,
        value=4
    )

    st.divider()

    st.write("💬 저장된 대화 수")
    st.write(len(st.session_state.messages))

    if st.button("🔄 대화 초기화"):
        reset_state(full=False)
        st.rerun()

    if st.button("🗑️ 전체 초기화"):
        reset_state(full=True)
        st.rerun()


# ============================================================
# 10. 메인 화면
# ============================================================

st.title("🤖 에이블 AI 업무 비서")

st.write(
    "음성으로 회의 요약, 업무 관리, 일정 등록을 요청하세요."
)

tab1, tab2, tab3, tab4 = st.tabs([
    "🎤 음성 AI",
    "💬 멀티턴 대화",
    "📊 업무 우선순위",
    "🗓️ 일정 관리"
])


# ============================================================
# TAB 1. 음성 AI
# ============================================================

with tab1:
    st.header("🎤 음성 AI 서비스")

    with st.container(border=True):
        uploaded_file = st.file_uploader(
            "회의록 파일 업로드",
            type=["txt"],
            key="meeting_upload"
        )

        audio_value = st.audio_input(
            "무엇을 도와드릴까요?",
            key="voice_recording"
        )

        run_button = st.button(
            "🚀 실행",
            key="run_voice_ai"
        )

    if run_button:
        if uploaded_file is not None:
            new_meeting = uploaded_file.getvalue().decode(
                "utf-8-sig"
            )

            if new_meeting != st.session_state.meeting_text:
                st.session_state.meeting_text = new_meeting
                st.session_state.tasks = []
                st.session_state.meeting_signature = ""
                st.session_state.registered_events = {}

        if not st.session_state.meeting_text:
            st.warning("회의록 파일을 먼저 업로드하세요.")

        else:
            try:
                if audio_value is not None:
                    with st.spinner("🎤 음성을 인식하고 있습니다..."):
                        transcript = client.audio.transcriptions.create(
                            model="whisper-1",
                            file=(
                                "audio.wav",
                                audio_value.getvalue(),
                                "audio/wav"
                            )
                        )

                        question = transcript.text
                        st.session_state.transcript = question
                else:
                    question = (
                        "에이블, 김 대리입니다. "
                        "다음 미팅 내용을 요약해 주세요. "
                        "나에게 배정된 업무가 있다면 "
                        "자세히 알려주세요."
                    )
                    st.session_state.transcript = ""

                with st.spinner("🤖 회의록을 분석하고 있습니다..."):
                    answer = ask_ai(question)
                    st.session_state.summary = answer

                    signature = hashlib.sha256(
                        st.session_state.meeting_text.encode("utf-8")
                    ).hexdigest()

                    if signature != st.session_state.meeting_signature:
                        tasks = extract_tasks(
                            st.session_state.meeting_text
                        )

                        st.session_state.tasks = tasks
                        st.session_state.meeting_signature = signature

                # 음성 답변 생성
                try:
                    with st.spinner("🔊 음성 답변 생성 중..."):
                        speech = client.audio.speech.create(
                            model="tts-1",
                            voice="alloy",
                            input=answer[:4000]
                        )

                        st.session_state.audio_answer = (
                            speech.read()
                        )
                except Exception as tts_error:
                    st.session_state.audio_answer = None
                    st.warning(
                        f"음성 답변 생성 실패: {tts_error}"
                    )

                st.success("회의록 분석이 완료되었습니다.")

            except Exception as error:
                st.error(f"분석 중 오류 발생: {error}")

    if st.session_state.transcript:
        st.subheader("🎤 음성 인식 결과")
        st.write(st.session_state.transcript)

    if st.session_state.summary:
        st.subheader("📋 회의록 요약 결과")
        st.markdown(st.session_state.summary)

    if st.session_state.audio_answer:
        st.subheader("🔊 AI 음성 답변")
        st.audio(
            st.session_state.audio_answer,
            format="audio/mp3"
        )


# ============================================================
# TAB 2. 멀티턴 대화
# ============================================================

with tab2:
    st.header("💬 멀티턴 AI 대화")

    st.caption(
        "이전 대화와 업로드된 회의록을 기억하고 답변합니다."
    )

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    prompt = st.chat_input(
        "에이블에게 질문하세요.",
        key="chat_input"
    )

    if prompt:
        try:
            with st.spinner("에이블이 답변을 준비하고 있습니다..."):
                ask_ai(prompt)
            st.rerun()

        except Exception as error:
            st.error(f"AI 응답 오류: {error}")


# ============================================================
# TAB 3. 업무 우선순위
# ============================================================

with tab3:
    st.header("📊 업무 우선순위 분석")

    tasks = st.session_state.tasks

    if not tasks:
        st.info(
            "먼저 음성 AI 탭에서 회의록 분석을 실행하세요."
        )

    else:
        st.subheader("📌 아이젠하워 매트릭스")

        col1, col2 = st.columns(2)

        for idx, quadrant in enumerate(["Q1", "Q2", "Q3", "Q4"]):
            column = col1 if idx % 2 == 0 else col2

            with column:
                with st.container(border=True):
                    st.markdown(
                        f"**{QUADRANTS[quadrant]}**"
                    )

                    matched = [
                        task for task in tasks
                        if classify_task(task) == quadrant
                    ]

                    if matched:
                        for task in matched:
                            st.write("•", task["task_name"])
                    else:
                        st.caption("해당 업무 없음")

        st.divider()

        st.subheader("👥 R&R 업무 분석")

        for idx, task in enumerate(tasks):
            with st.expander(
                f"{classify_task(task)} | "
                f"{task['task_name']}"
            ):
                st.write("**담당자:**", task["owner"])
                st.write("**업무 내용:**", task["description"])
                st.write(
                    "**마감일:**",
                    task["deadline"] or "기한 미정"
                )
                st.write(
                    "**우선순위 판단 근거:**",
                    task["reason"]
                )

                hours = task.get("estimated_hours")

                new_hours = st.number_input(
                    "예상 소요시간 (시간)",
                    min_value=0.0,
                    max_value=1000.0,
                    value=float(hours) if hours is not None else 0.0,
                    step=0.5,
                    key=f"hours_{idx}"
                )

                if hours != new_hours:
                    task["estimated_hours"] = (
                        new_hours if new_hours > 0 else None
                    )

        st.divider()

        st.subheader("🚨 김 대리 업무 과부하 분석")

        loads, warnings, unknown = analyze_workload(
            tasks,
            daily_capacity
        )

        for warning in warnings:
            st.warning(warning)

        if unknown:
            st.info(
                "업무시간 또는 마감일 미확인: "
                + ", ".join(unknown)
            )

        if loads:
            df = pd.DataFrame([
                {
                    "날짜": str(day),
                    "예상 업무시간": round(hours, 2)
                }
                for day, hours in sorted(loads.items())
            ])

            st.bar_chart(
                df.set_index("날짜")
            )

            if not warnings:
                st.success(
                    "입력된 예상 업무시간 기준으로 "
                    "일별 업무 과부하가 감지되지 않았습니다."
                )

        else:
            st.caption(
                "분석 가능한 김 대리의 업무 일정이 없습니다."
            )


# ============================================================
# TAB 4. 일정 관리
# ============================================================

with tab4:
    st.header("🗓️ 일정 관리")

    st.write(
        "업무 마감일을 기준으로 시작일을 역산하고 "
        "Google Calendar에 등록할 수 있습니다."
    )

    st.caption(
        "역산은 평일 기준이며 공휴일과 기존 캘린더 일정은 "
        "반영하지 않습니다. 캘린더 등록은 시작일부터 "
        "마감일까지 하나의 종일 일정으로 생성됩니다."
    )

    tasks = st.session_state.tasks

    if not tasks:
        st.info(
            "먼저 회의록을 분석해 업무 목록을 생성하세요."
        )

    else:
        for idx, task in enumerate(tasks):
            with st.container(border=True):
                st.subheader(task["task_name"])

                st.write("**담당자:**", task["owner"])
                st.write(
                    "**마감일:**",
                    task["deadline"] or "기한 미정"
                )

                schedule = calculate_schedule(
                    task,
                    daily_capacity
                )

                if schedule is None:
                    st.warning(
                        "마감일 또는 예상 소요시간이 없어 "
                        "자동 역산할 수 없습니다."
                    )
                    continue

                start, deadline = schedule

                col1, col2, col3 = st.columns(3)

                with col1:
                    st.metric(
                        "권장 시작일",
                        start.isoformat()
                    )

                with col2:
                    st.metric(
                        "마감일",
                        deadline.isoformat()
                    )

                with col3:
                    st.metric(
                        "예상 소요시간",
                        f"{task['estimated_hours']}시간"
                    )

                if deadline < TODAY:
                    st.error("이미 마감일이 지난 업무입니다.")
                    continue

                if start < TODAY:
                    st.warning(
                        "권장 시작일이 이미 지났습니다. "
                        "업무량이나 일정을 조정해야 합니다."
                    )

                event_key = (
                    f"{task['task_name']}|"
                    f"{task['owner']}|"
                    f"{start}|{deadline}"
                )

                if event_key in st.session_state.registered_events:
                    st.success(
                        "현재 세션에서 캘린더 등록 완료"
                    )

                    event_url = (
                        st.session_state.registered_events[event_key]
                    )

                    if event_url:
                        st.link_button(
                            "📅 Google Calendar에서 보기",
                            event_url
                        )

                else:
                    if st.button(
                        "📅 Google Calendar 등록 승인",
                        key=f"register_{idx}"
                    ):
                        try:
                            with st.spinner("일정 등록 중..."):
                                event = register_calendar(
                                    task,
                                    start,
                                    deadline
                                )

                            st.session_state.registered_events[
                                event_key
                            ] = event.get("htmlLink", "")

                            st.success(
                                "Google Calendar 등록 완료!"
                            )

                            st.rerun()

                        except Exception as error:
                            st.error(
                                f"일정 등록 실패: {error}"
                            )

st.divider()
st.caption(
    "에이블 AI 업무 비서 | "
    "회의록 분석 · 멀티턴 대화 · R&R · 업무 우선순위 · 일정 관리"
)
