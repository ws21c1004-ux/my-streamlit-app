
import os
import json
import math
import base64
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from openai import OpenAI

# ==========================================
# 1. 기본 설정
# ==========================================

client = OpenAI()
KST = ZoneInfo("Asia/Seoul")

st.set_page_config(
    page_title="에이블 AI 업무 비서",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 에이블 AI 업무 비서")
st.write("음성으로 회의 요약, 업무 관리, 일정 등록을 요청하세요.")

# ==========================================
# 2. 역할 부여
# ==========================================

sys_role = """
당신은 '김 대리'의 개인 비서 '에이블'입니다.
김 대리가 요청한 업무를 최선을 다해 처리합니다.

[주요 업무]
1. 회의록 핵심 내용 요약
2. 김 대리에게 배정된 업무 추출
3. R&R 업무 분장 정리
4. Eisenhower Matrix 우선순위 분류
5. 업무 과부하 분석
6. 마감일 역산 및 실행 일정 제안
7. 김 대리의 후속 질문에 답변

[답변 규칙]
- 한국어로 정중하고 간결하게 답변합니다.
- 회의록에 없는 내용은 지어내지 않습니다.
- 요약은 주제 → 논의 내용 → 결정 사항 순서로 작성합니다.
- 담당 업무는 업무명, 세부 내용, 기한을 정리합니다.
- 기한이 없으면 '기한 미정'으로 표시합니다.
- 결정 사항의 예산, 일정, 채널 등 구체적인 정보를 포함합니다.
- 업무 기한은 추측하지 않습니다.
- 예상 소요시간은 추정치임을 명시합니다.
- 이전 대화 내용을 참고하여 후속 질문에 답변합니다.
- 캘린더 등록은 사용자의 승인을 받아야 합니다.
"""

# ==========================================
# 3. 세션 상태 초기화 (멀티턴)
# ==========================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "meeting_text" not in st.session_state:
    st.session_state.meeting_text = ""

if "tasks" not in st.session_state:
    st.session_state.tasks = []

if "summary" not in st.session_state:
    st.session_state.summary = ""

if "registered_events" not in st.session_state:
    st.session_state.registered_events = {}

# ==========================================
# 4. Function Calling 도구 정의
# ==========================================

tools = [{
    "type": "function",
    "name": "extract_tasks",
    "description": "회의록에서 업무와 담당자, 기한, 우선순위를 추출합니다.",
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
                            "type": ["string", "null"],
                            "description": "YYYY-MM-DD 또는 null"
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

# ==========================================
# 5. 아이젠하워 매트릭스
# ==========================================

def classify_task(task):

    urgent = task["urgent"]
    important = task["important"]

    if urgent and important:
        return "Q1 긴급 + 중요"

    elif not urgent and important:
        return "Q2 비긴급 + 중요"

    elif urgent and not important:
        return "Q3 긴급 + 비중요"

    else:
        return "Q4 비긴급 + 비중요"


# ==========================================
# 6. 마감일 역산
# ==========================================

def calculate_schedule(task, daily_capacity=4):

    deadline = task.get("deadline")
    hours = task.get("estimated_hours")

    if not deadline or hours is None or hours <= 0:
        return None

    try:
        end_date = datetime.strptime(
            deadline, "%Y-%m-%d"
        ).date()
    except ValueError:
        return None

    required_days = math.ceil(hours / daily_capacity)

    start_date = end_date
    remaining = required_days - 1

    while remaining > 0:
        start_date -= timedelta(days=1)

        if start_date.weekday() < 5:
            remaining -= 1

    return start_date


# ==========================================
# 7. 업무 과부하 분석
# ==========================================

def analyze_workload(tasks, daily_capacity=4):

    today = datetime.now(KST).date()

    daily_load = {}
    warnings = []

    for task in tasks:

        deadline = task.get("deadline")
        hours = task.get("estimated_hours")

        if not deadline or hours is None or hours <= 0:
            continue

        try:
            end_date = datetime.strptime(
                deadline, "%Y-%m-%d"
            ).date()
        except ValueError:
            continue

        if end_date < today:
            warnings.append(
                f"마감 기한 경과: {task['task_name']}"
            )
            continue

        start_date = calculate_schedule(
            task, daily_capacity
        )

        if start_date is None:
            continue

        if start_date < today:
            warnings.append(
                f"일정 재조정 필요: {task['task_name']}"
            )

        start_date = max(start_date, today)

        workdays = []
        current = start_date

        while current <= end_date:

            if current.weekday() < 5:
                workdays.append(current)

            current += timedelta(days=1)

        if not workdays:
            warnings.append(
                f"작업 가능한 평일 없음: {task['task_name']}"
            )
            continue

        hours_per_day = hours / len(workdays)

        for day in workdays:
            daily_load[day] = (
                daily_load.get(day, 0) + hours_per_day
            )

    for day, total in sorted(daily_load.items()):

        if total > daily_capacity:
            warnings.append(
                f"{day}: 예상 {total:.1f}시간 "
                f"(기준 {daily_capacity}시간 초과)"
            )

    return warnings, daily_load


# ==========================================
# 8. Google Calendar 연동
# ==========================================

def get_calendar_service():

    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    info = dict(st.secrets["gcp_service_account"])

    credentials = (
        service_account.Credentials.from_service_account_info(
            info,
            scopes=[
                "https://www.googleapis.com/auth/calendar.events"
            ]
        )
    )

    service = build(
        "calendar",
        "v3",
        credentials=credentials,
        cache_discovery=False
    )

    return service


def register_calendar(task, start_date):

    service = get_calendar_service()

    deadline = datetime.strptime(
        task["deadline"], "%Y-%m-%d"
    ).date()

    event = {
        "summary": f"[에이블] {task['task_name']}",
        "description": (
            f"담당자: {task['owner']}\n"
            f"업무 내용: {task['description']}\n"
            f"우선순위: {classify_task(task)}\n"
            f"예상 소요시간: {task['estimated_hours']}\n"
            f"마감일: {task['deadline']}"
        ),
        "start": {
            "date": start_date.isoformat()
        },
        "end": {
            "date": (
                deadline + timedelta(days=1)
            ).isoformat()
        }
    }

    result = service.events().insert(
        calendarId=st.secrets["calendar_id"],
        body=event
    ).execute()

    return result


# ==========================================
# 9. 멀티턴 대화 함수
# ==========================================

def chat_with_ai(user_input):

    # 사용자 메시지 저장
    st.session_state.messages.append({
        "role": "user",
        "content": user_input
    })

    # 최근 20개 메시지 사용
    limited_history = st.session_state.messages[-20:]

    # 시스템 메시지
    system_content = sys_role

    # 회의록 내용 유지
    meeting_text = st.session_state.meeting_text

    if meeting_text:
        system_content += f"""

[미팅 내용]
{meeting_text}
"""

    # LLM 호출
    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": system_content},
            *limited_history
        ],
        temperature=0
    )

    # 최종 응답
    answer = response.output_text

    # AI 응답 저장
    st.session_state.messages.append({
        "role": "assistant",
        "content": answer
    })

    return answer


# ==========================================
# 10. 업무 추출 함수 (Function Calling)
# ==========================================

def extract_tasks(meeting_text):

    if not meeting_text.strip():
        return []

    prompt = f"""
현재 날짜: {datetime.now(KST).date()}

다음 회의록에서 실행 업무를 추출하세요.

[회의록]
{meeting_text}

[추출 규칙]
- 업무명과 세부 내용을 정리합니다.
- 담당자가 없으면 '담당자 미정'으로 표시합니다.
- 기한이 없으면 null로 표시합니다.
- 긴급도와 중요도를 판단하고 근거를 작성합니다.
- 예상 소요시간에 근거가 없으면 null로 표시합니다.
- 상대적 날짜는 현재 날짜를 기준으로 해석합니다.
"""

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {"role": "system", "content": sys_role},
            {"role": "user", "content": prompt}
        ],
        tools=tools,
        tool_choice={
            "type": "function",
            "name": "extract_tasks"
        },
        temperature=0
    )

    for item in response.output:

        if item.type == "function_call":
            if item.name == "extract_tasks":

                arguments = json.loads(item.arguments)

                return arguments.get("tasks", [])

    return []


# ==========================================
# 11. 사이드바 설정
# ==========================================

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
        st.session_state.messages = []
        st.rerun()

    if st.button("🗑️ 전체 초기화"):

        st.session_state.messages = []
        st.session_state.meeting_text = ""
        st.session_state.tasks = []
        st.session_state.summary = ""
        st.session_state.registered_events = {}

        st.rerun()


# ==========================================
# 12. 탭 구성
# ==========================================

tab1, tab2, tab3, tab4 = st.tabs([
    "🎤 음성 AI",
    "💬 멀티턴 대화",
    "📊 업무 우선순위",
    "📅 일정 관리"
])


# ==========================================
# TAB 1. 음성 AI 및 회의록 요약
# ==========================================

with tab1:

    st.subheader("🎤 음성 AI 서비스")

    with st.form("task_form"):

        uploaded_file = st.file_uploader(
            "회의록 파일 업로드",
            type=["txt"]
        )

        audio_value = st.audio_input(
            "무엇을 도와드릴까요?"
        )

        submit = st.form_submit_button("🚀 실행")

    if submit:

        # 회의록 저장
        if uploaded_file is not None:

            meeting_text = (
                uploaded_file.getvalue()
                .decode("utf-8-sig")
            )

            st.session_state.meeting_text = meeting_text

        else:
            meeting_text = st.session_state.meeting_text

        if audio_value is None:
            st.warning("음성을 녹음해주세요.")

        else:

            # STT
            with st.spinner("음성을 인식하고 있습니다..."):

                transcript = client.audio.transcriptions.create(
                    model="whisper-1",
                    file=(
                        "audio.wav",
                        audio_value.getvalue(),
                        "audio/wav"
                    )
                )

            st.write("🎤 음성 인식 결과")
            st.write(transcript.text)

            # 멀티턴 LLM 호출
            with st.spinner("에이블이 답변을 준비하고 있습니다..."):

                answer = chat_with_ai(transcript.text)

            st.subheader("🤖 AI 답변")
            st.write(answer)

            # 회의록 업무 추출
            if meeting_text:

                with st.spinner("담당 업무를 분석하고 있습니다..."):

                    st.session_state.tasks = extract_tasks(
                        meeting_text
                    )

                    st.session_state.summary = answer

                st.success("회의록 및 담당 업무 분석 완료!")

            # TTS 음성 변환
            with st.spinner("음성을 생성하고 있습니다..."):

                speech_response = client.audio.speech.create(
                    model="tts-1",
                    voice="alloy",
                    input=answer
                )

                audio_bytes = speech_response.content

                audio_base64 = base64.b64encode(
                    audio_bytes
                ).decode()

                st.audio(
                    audio_bytes,
                    format="audio/mp3"
                )

    if st.session_state.meeting_text:
        st.info("현재 회의록이 세션에 저장되어 있습니다.")


# ==========================================
# TAB 2. 멀티턴 대화
# ==========================================

with tab2:

    st.subheader("💬 에이블과 대화하기")

    st.write("이전 대화를 기억하여 후속 질문에 답변합니다.")

    # 기존 대화 출력
    for message in st.session_state.messages:

        with st.chat_message(message["role"]):
            st.write(message["content"])

    # 새로운 질문
    user_input = st.chat_input(
        "에이블에게 추가 질문을 입력하세요."
    )

    if user_input:

        with st.chat_message("user"):
            st.write(user_input)

        with st.spinner("에이블이 답변을 준비하고 있습니다..."):

            answer = chat_with_ai(user_input)

        with st.chat_message("assistant"):
            st.write(answer)


# ==========================================
# TAB 3. R&R 및 Eisenhower Matrix
# ==========================================

with tab3:

    st.subheader("📊 업무 우선순위 분석")

    tasks = st.session_state.tasks

    if not tasks:
        st.info("먼저 회의록을 업로드하고 음성 분석을 실행하세요.")

    else:

        col1, col2 = st.columns(2)

        with col1:

            st.error("🔴 Q1 긴급 + 중요")

            for task in tasks:
                if classify_task(task).startswith("Q1"):
                    st.write(task["task_name"])

            st.warning("🟠 Q3 긴급 + 비중요")

            for task in tasks:
                if classify_task(task).startswith("Q3"):
                    st.write(task["task_name"])

        with col2:

            st.info("🔵 Q2 비긴급 + 중요")

            for task in tasks:
                if classify_task(task).startswith("Q2"):
                    st.write(task["task_name"])

            st.success("🟢 Q4 비긴급 + 비중요")

            for task in tasks:
                if classify_task(task).startswith("Q4"):
                    st.write(task["task_name"])

        st.divider()

        st.subheader("👥 R&R 업무 분장")

        for task in tasks:

            with st.expander(
                f"{classify_task(task)} | {task['task_name']}"
            ):

                st.write("담당자:", task["owner"])
                st.write("세부 내용:", task["description"])
                st.write(
                    "마감일:",
                    task["deadline"] or "기한 미정"
                )
                st.write("판단 근거:", task["reason"])

                hours = task["estimated_hours"]

                st.write(
                    "예상 소요시간:",
                    f"{hours}시간 (추정)"
                    if hours is not None
                    else "추정 불가"
                )

        st.divider()

        # 김 대리 담당 업무
        my_tasks = [
            task for task in tasks
            if "김" in task["owner"]
            and "대리" in task["owner"]
        ]

        st.subheader("🚨 업무 과부하 분석")

        warnings, daily_load = analyze_workload(
            my_tasks,
            daily_capacity
        )

        if warnings:

            for warning in warnings:
                st.warning(warning)

        elif daily_load:
            st.success("현재 예상 업무량은 설정한 기준 이내입니다.")

        else:
            st.info(
                "업무량 분석에 필요한 기한 또는 "
                "예상 소요시간이 부족합니다."
            )

        if daily_load:

            workload_df = pd.DataFrame([
                {
                    "날짜": str(day),
                    "예상 업무시간": hours
                }
                for day, hours in sorted(daily_load.items())
            ])

            st.bar_chart(
                workload_df.set_index("날짜")
            )


# ==========================================
# TAB 4. Action Scheduler
# ==========================================

with tab4:

    st.subheader("📅 Action Scheduler")

    tasks = st.session_state.tasks

    if not tasks:

        st.info("등록할 업무가 없습니다.")

    else:

        for i, task in enumerate(tasks):

            start_date = calculate_schedule(
                task,
                daily_capacity
            )

            with st.container(border=True):

                st.write(f"### {task['task_name']}")

                st.write("담당자:", task["owner"])
                st.write(
                    "우선순위:",
                    classify_task(task)
                )

                if start_date is None:

                    st.info(
                        "기한 또는 예상 소요시간이 없어 "
                        "일정 역산이 불가능합니다."
                    )

                    continue

                st.write("권장 시작일:", start_date)
                st.write("마감일:", task["deadline"])

                if start_date < datetime.now(KST).date():
                    st.warning(
                        "권장 시작일이 지났습니다. "
                        "일정 재조정이 필요합니다."
                    )

                event_key = (
                    f"{task['task_name']}_"
                    f"{task['deadline']}_{i}"
                )

                if event_key in st.session_state.registered_events:

                    st.success("이미 캘린더에 등록된 업무입니다.")

                    event_url = (
                        st.session_state.registered_events[event_key]
                    )

                    if event_url:
                        st.markdown(
                            f"[등록된 일정 보기]({event_url})"
                        )

                else:

                    if st.button(
                        f"📅 캘린더 등록: {task['task_name']}",
                        key=f"calendar_{i}"
                    ):

                        try:

                            event = register_calendar(
                                task,
                                start_date
                            )

                            event_url = event.get(
                                "htmlLink", ""
                            )

                            st.session_state.registered_events[
                                event_key
                            ] = event_url

                            st.success("캘린더 등록 완료!")

                            if event_url:
                                st.markdown(
                                    f"[일정 확인]({event_url})"
                                )

                        except Exception as e:

                            st.error(
                                f"캘린더 등록 실패: {e}"
                            )
