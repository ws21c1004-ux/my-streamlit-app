import os
import json
import math
import hashlib
from datetime import datetime, date, time, timedelta
from zoneinfo import ZoneInfo

import pandas as pd
import streamlit as st
from openai import OpenAI

# ============================================================
# 1. 기본 설정
# ============================================================

st.set_page_config(
    page_title="에이블 AI 업무 비서 Pro",
    page_icon="🤖",
    layout="wide"
)

KST = ZoneInfo("Asia/Seoul")
TODAY = datetime.now(KST).date()


def setting(name, default=None):
    value = os.getenv(name)
    if value:
        return value
    try:
        return st.secrets.get(name, default)
    except Exception:
        return default


api_key = setting("OPENAI_API_KEY")

if not api_key:
    st.error("OPENAI_API_KEY를 설정해 주세요.")
    st.stop()

client = OpenAI(api_key=api_key)


# ============================================================
# 2. 세션 상태
# ============================================================

INITIAL = {
    "messages": [],
    "meeting_text": "",
    "meeting_hash": "",
    "summary": "",
    "transcript": "",
    "audio_bytes": None,
    "tasks": [],
    "subtasks": {},
    "calendar_proposals": {},
    "registered": {},
    "voice_enabled": True,
    "autoplay": False,
}

for key, value in INITIAL.items():
    if key not in st.session_state:
        st.session_state[key] = (
            value.copy() if isinstance(value, (list, dict)) else value
        )


def reset_all():
    for key, value in INITIAL.items():
        st.session_state[key] = (
            value.copy() if isinstance(value, (list, dict)) else value
        )


# ============================================================
# 3. AI 역할
# ============================================================

ROLE = """
당신은 김 대리의 개인 AI 업무 비서 '에이블'입니다.

[역할]
1. 회의록을 요약합니다.
2. 주요 논의 및 결정 사항을 정리합니다.
3. 담당자별 업무(R&R)를 분석합니다.
4. 김 대리의 담당 업무를 자세히 설명합니다.
5. 업무 우선순위를 분석합니다.
6. 업무를 실행 가능한 단계로 세분화합니다.
7. 마감일과 예상 작업시간을 고려해 일정을 제안합니다.

[원칙]
- 한국어로 친절하고 명료하게 답변합니다.
- 회의록에 없는 사실을 만들어내지 않습니다.
- 담당자와 마감일을 추측하지 않습니다.
- AI의 판단과 회의록의 사실을 구분합니다.
- 캘린더 등록 전 반드시 사용자 승인을 받습니다.
"""


# ============================================================
# 4. 멀티턴 AI 대화
# ============================================================

def chat(question):
    messages = [
        {
            "role": "system",
            "content": (
                ROLE
                + f"\n오늘 날짜: {TODAY}"
                + "\n[회의록]\n"
                + (
                    st.session_state.meeting_text
                    or "업로드된 회의록 없음"
                )
            )
        }
    ]

    messages.extend(st.session_state.messages[-16:])

    messages.append({
        "role": "user",
        "content": question
    })

    response = client.responses.create(
        model="gpt-4o-mini",
        input=messages
    )

    answer = response.output_text

    st.session_state.messages.extend([
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer}
    ])

    return answer


# ============================================================
# 5. 회의록 업무 추출
# ============================================================

TASK_TOOL = [{
    "type": "function",
    "name": "extract_tasks",
    "description": "회의록에서 담당 업무를 추출합니다.",
    "parameters": {
        "type": "object",
        "properties": {
            "tasks": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "name": {"type": "string"},
                        "description": {"type": "string"},
                        "owner": {"type": "string"},
                        "deadline": {
                            "type": ["string", "null"]
                        },
                        "urgent": {"type": "boolean"},
                        "important": {"type": "boolean"},
                        "reason": {"type": "string"},
                        "hours": {
                            "type": ["number", "null"]
                        }
                    },
                    "required": [
                        "name",
                        "description",
                        "owner",
                        "deadline",
                        "urgent",
                        "important",
                        "reason",
                        "hours"
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
                "content": ROLE + f"\n오늘 날짜: {TODAY}"
            },
            {
                "role": "user",
                "content": (
                    "다음 회의록에서 실제 언급된 업무만 추출하세요.\n"
                    "기한이 명확하면 YYYY-MM-DD 형식으로 작성하세요.\n"
                    "기한이 불명확하면 null로 작성하세요.\n"
                    "소요시간이 명시되지 않으면 null로 작성하세요.\n"
                    "긴급도와 중요도 판단 근거를 포함하세요.\n\n"
                    + meeting_text
                )
            }
        ],
        tools=TASK_TOOL,
        tool_choice={
            "type": "function",
            "name": "extract_tasks"
        }
    )

    for item in response.output:
        if (
            item.type == "function_call"
            and item.name == "extract_tasks"
        ):
            tasks = json.loads(item.arguments)["tasks"]

            for idx, task in enumerate(tasks):
                task["id"] = idx
                task["status"] = "예정"

            return tasks

    return []


# ============================================================
# 6. AI 업무 세분화
# ============================================================

SUBTASK_TOOL = [{
    "type": "function",
    "name": "split_task",
    "description": "업무를 실행 가능한 세부 단계로 분할합니다.",
    "parameters": {
        "type": "object",
        "properties": {
            "steps": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "title": {"type": "string"},
                        "details": {"type": "string"},
                        "hours": {"type": "number"}
                    },
                    "required": [
                        "title",
                        "details",
                        "hours"
                    ],
                    "additionalProperties": False
                }
            }
        },
        "required": ["steps"],
        "additionalProperties": False
    },
    "strict": True
}]


def split_task(task):
    total_hours = task.get("hours")

    if not total_hours or total_hours <= 0:
        raise ValueError(
            "AI 업무 세분화 전에 예상 작업시간을 입력하세요."
        )

    response = client.responses.create(
        model="gpt-4o-mini",
        input=[
            {
                "role": "system",
                "content": (
                    "당신은 업무 계획 전문가입니다. "
                    "업무를 2~6개의 실행 가능한 단계로 분할하세요. "
                    "각 단계의 예상 작업시간을 배분하세요."
                )
            },
            {
                "role": "user",
                "content": json.dumps({
                    "task": task["name"],
                    "description": task["description"],
                    "total_hours": total_hours
                }, ensure_ascii=False)
            }
        ],
        tools=SUBTASK_TOOL,
        tool_choice={
            "type": "function",
            "name": "split_task"
        }
    )

    for item in response.output:
        if (
            item.type == "function_call"
            and item.name == "split_task"
        ):
            steps = json.loads(item.arguments)["steps"]

            steps = [
                step for step in steps
                if step["hours"] > 0
            ]

            if not steps:
                raise ValueError("세부 작업이 생성되지 않았습니다.")

            hours_sum = sum(step["hours"] for step in steps)

            for step in steps:
                step["hours"] = round(
                    total_hours * step["hours"] / hours_sum,
                    2
                )

            steps[-1]["hours"] = round(
                total_hours
                - sum(step["hours"] for step in steps[:-1]),
                2
            )

            return steps

    return []


# ============================================================
# 7. 업무 분석
# ============================================================

def quadrant(task):
    urgent = task["urgent"]
    important = task["important"]

    if urgent and important:
        return "Q1"
    if important:
        return "Q2"
    if urgent:
        return "Q3"

    return "Q4"


def parse_date(value):
    try:
        return date.fromisoformat(str(value)) if value else None
    except (ValueError, TypeError):
        return None


def is_my_task(task):
    owner = task.get("owner", "")
    return "김" in owner and "대리" in owner


def analyze_workload(tasks, capacity):
    loads = {}

    for task in tasks:
        if not is_my_task(task):
            continue

        if task["status"] == "완료":
            continue

        hours = task.get("hours")
        deadline = parse_date(task.get("deadline"))

        if not hours or not deadline or deadline < TODAY:
            continue

        days = []
        current = TODAY

        while current <= deadline:
            if current.weekday() < 5:
                days.append(current)
            current += timedelta(days=1)

        if not days:
            continue

        per_day = float(hours) / len(days)

        for day in days:
            loads[day] = loads.get(day, 0) + per_day

    return loads


# ============================================================
# 8. Google Calendar 연결
# ============================================================

def calendar_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    scopes = [
        "https://www.googleapis.com/auth/calendar.events",
        "https://www.googleapis.com/auth/calendar.readonly"
    ]

    service_file = setting("GOOGLE_SERVICE_ACCOUNT_FILE")
    service_info = setting("gcp_service_account")

    if service_file:
        credentials = (
            service_account.Credentials.from_service_account_file(
                service_file,
                scopes=scopes
            )
        )

    elif service_info:
        credentials = (
            service_account.Credentials.from_service_account_info(
                dict(service_info),
                scopes=scopes
            )
        )

    else:
        raise RuntimeError(
            "Google Calendar 서비스 계정이 설정되지 않았습니다."
        )

    return build(
        "calendar",
        "v3",
        credentials=credentials,
        cache_discovery=False
    )


def calendar_id():
    value = (
        setting("GOOGLE_CALENDAR_ID")
        or setting("calendar_id")
    )

    if not value:
        raise RuntimeError("Google Calendar ID가 없습니다.")

    return value


def calendar_busy(start_day, end_day):
    service = calendar_service()
    cid = calendar_id()

    start_dt = datetime.combine(
        start_day, time.min, tzinfo=KST
    )

    end_dt = datetime.combine(
        end_day + timedelta(days=1),
        time.min,
        tzinfo=KST
    )

    result = service.freebusy().query(
        body={
            "timeMin": start_dt.isoformat(),
            "timeMax": end_dt.isoformat(),
            "timeZone": "Asia/Seoul",
            "items": [{"id": cid}]
        }
    ).execute()

    entry = result["calendars"][cid]

    if entry.get("errors"):
        raise RuntimeError(str(entry["errors"]))

    return [
        (
            datetime.fromisoformat(
                item["start"].replace("Z", "+00:00")
            ).astimezone(KST),
            datetime.fromisoformat(
                item["end"].replace("Z", "+00:00")
            ).astimezone(KST)
        )
        for item in entry.get("busy", [])
    ]


def overlaps(start, end, busy):
    return any(
        start < busy_end and end > busy_start
        for busy_start, busy_end in busy
    )


# ============================================================
# 9. 빈 시간 추천
# ============================================================

def propose_slots(task, capacity, busy, reserved):
    deadline = parse_date(task.get("deadline"))
    hours = task.get("hours")

    if deadline is None or not hours or hours <= 0:
        raise ValueError(
            "유효한 마감일과 예상 작업시간이 필요합니다."
        )

    if deadline < TODAY:
        raise ValueError("마감일이 지났습니다.")

    remaining = math.ceil(float(hours) * 2) * 30
    daily_limit = int(capacity * 60)

    slots = []
    current_day = TODAY
    now = datetime.now(KST)

    while current_day <= deadline and remaining > 0:
        if current_day.weekday() < 5:
            used = 0

            current = datetime.combine(
                current_day,
                time(9),
                tzinfo=KST
            )

            closing = datetime.combine(
                current_day,
                time(18),
                tzinfo=KST
            )

            while (
                current + timedelta(minutes=30) <= closing
                and remaining > 0
            ):
                next_time = current + timedelta(minutes=30)

                if (
                    current >= now
                    and used + 30 <= daily_limit
                    and not overlaps(
                        current,
                        next_time,
                        busy + reserved
                    )
                ):
                    slots.append((current, next_time))
                    used += 30
                    remaining -= 30

                current = next_time

        current_day += timedelta(days=1)

    return slots, remaining


def insert_calendar_slots(task, slots):
    service = calendar_service()
    cid = calendar_id()

    created = []

    for idx, (start, end) in enumerate(slots, 1):
        body = {
            "summary": (
                f"[에이블] {task['name']} "
                f"({idx}/{len(slots)})"
            ),
            "description": (
                f"담당자: {task['owner']}\n"
                f"업무 내용: {task['description']}\n"
                f"마감일: {task['deadline']}"
            ),
            "start": {
                "dateTime": start.isoformat(),
                "timeZone": "Asia/Seoul"
            },
            "end": {
                "dateTime": end.isoformat(),
                "timeZone": "Asia/Seoul"
            }
        }

        try:
            event = service.events().insert(
                calendarId=cid,
                body=body
            ).execute()

            created.append(event)

        except Exception as error:
            raise RuntimeError(
                f"{len(created)}개 등록 후 실패했습니다. "
                f"중복 등록 여부를 확인하세요: {error}"
            ) from error

    return created


# ============================================================
# 10. 보고서 생성
# ============================================================

def make_task_df(tasks):
    return pd.DataFrame([
        {
            "업무명": task["name"],
            "담당자": task["owner"],
            "마감일": task.get("deadline") or "미정",
            "우선순위": quadrant(task),
            "예상시간": task.get("hours"),
            "진행상태": task["status"],
            "업무내용": task["description"],
            "판단근거": task["reason"]
        }
        for task in tasks
    ])


def report_markdown():
    lines = [
        "# 에이블 AI 업무 보고서",
        f"생성일: {TODAY}",
        "",
        "## 회의 요약",
        st.session_state.summary or "요약 없음",
        "",
        "## 업무 목록"
    ]

    for task in st.session_state.tasks:
        lines.extend([
            f"### {task['name']}",
            f"- 담당자: {task['owner']}",
            f"- 마감일: {task.get('deadline') or '미정'}",
            f"- 우선순위: {quadrant(task)}",
            f"- 판단 근거: {task['reason']}",
            f"- 예상시간: {task.get('hours') or '미정'}",
            f"- 진행상태: {task['status']}",
            f"- 상세내용: {task['description']}"
        ])

        for step in st.session_state.subtasks.get(
            task["id"], []
        ):
            lines.append(
                f"  - {step['title']} "
                f"({step['hours']}시간): {step['details']}"
            )

        lines.append("")

    lines.append(
        "※ AI가 생성한 우선순위 및 예상 작업시간은 "
        "추정치이므로 검토가 필요합니다."
    )

    return "\n".join(lines)


# ============================================================
# 11. CSS 디자인
# ============================================================

st.markdown("""
<style>
.stApp {
    background-color: #FFFFFF;
    color: #202638;
}

section[data-testid="stSidebar"] {
    background-color: #F0F2F6;
    border-right: 1px solid #E4E7ED;
}

h1, h2, h3 {
    color: #24283A !important;
    letter-spacing: -0.7px;
}

.stButton > button[kind="primary"],
button[kind="primary"] {
    background-color: #FF4B50 !important;
    border-color: #FF4B50 !important;
    color: white !important;
    border-radius: 9px;
}

.stButton > button {
    border-radius: 9px;
    min-height: 39px;
}

[data-testid="stMetric"] {
    border: 1px solid #E9EBF0;
    border-radius: 12px;
    padding: 12px;
    background: white;
}

.robot-shell {
    height: 340px;
    display: flex;
    align-items: center;
    justify-content: center;
    flex-direction: column;
}

.robot-shell svg {
    width: min(300px, 100%);
    height: 290px;
    filter: drop-shadow(
        0 14px 16px rgba(50,64,100,0.12)
    );
}

.robot-caption {
    font-weight: 700;
    color: #397394;
}

.assistant-note {
    background: #FFF8F8;
    border: 1px solid #FFDBDC;
    padding: 14px 16px;
    border-radius: 12px;
    line-height: 1.65;
}
</style>
""", unsafe_allow_html=True)


# ============================================================
# 12. 미래지향적 AI 로봇 캐릭터
# ============================================================

ROBOT_SVG = """
<div class="robot-shell">
<svg viewBox="0 0 320 310"
     xmlns="http://www.w3.org/2000/svg"
     role="img"
     aria-label="에이블 AI 로봇">

<defs>
<linearGradient id="head" x1="0" x2="1" y1="0" y2="1">
<stop stop-color="#F4F8FF"/>
<stop offset="1" stop-color="#C7D5ED"/>
</linearGradient>

<linearGradient id="body" x1="0" x2="1">
<stop stop-color="#DCE8FA"/>
<stop offset="1" stop-color="#9AB4D9"/>
</linearGradient>

<linearGradient id="screen" x1="0" x2="1">
<stop stop-color="#26344E"/>
<stop offset="1" stop-color="#17233C"/>
</linearGradient>
</defs>

<ellipse cx="160" cy="290" rx="91" ry="12"
fill="#DCE2ED" opacity=".6"/>

<path d="M160 57V29"
stroke="#687F9F" stroke-width="8"
stroke-linecap="round"/>

<circle cx="160" cy="23" r="12" fill="#FF5B62"/>

<rect x="62" y="65" width="196" height="142"
rx="58" fill="url(#head)"
stroke="#8199BD" stroke-width="5"/>

<rect x="78" y="83" width="164" height="107"
rx="42" fill="url(#screen)"/>

<ellipse cx="119" cy="129" rx="16" ry="20"
fill="#64E2FF"/>

<ellipse cx="201" cy="129" rx="16" ry="20"
fill="#64E2FF"/>

<circle cx="114" cy="123" r="5" fill="white"/>
<circle cx="196" cy="123" r="5" fill="white"/>

<path d="M137 157Q160 177 183 157"
fill="none" stroke="#7AF0EF"
stroke-width="7" stroke-linecap="round"/>

<circle cx="61" cy="140" r="15"
fill="#A4B7D6" stroke="#8199BD" stroke-width="4"/>

<circle cx="259" cy="140" r="15"
fill="#A4B7D6" stroke="#8199BD" stroke-width="4"/>

<rect x="101" y="205" width="118" height="75"
rx="31" fill="url(#body)"
stroke="#8199BD" stroke-width="5"/>

<rect x="134" y="223" width="52" height="31"
rx="13" fill="#E8F8FF"
stroke="#8199BD" stroke-width="3"/>

<path d="M151 239h18m-9-9v18"
stroke="#FF5B62" stroke-width="4"
stroke-linecap="round"/>

<path d="M99 220Q70 222 72 252"
stroke="#9BB3D6" stroke-width="19"
stroke-linecap="round"/>

<path d="M221 220Q250 220 252 191"
stroke="#9BB3D6" stroke-width="19"
stroke-linecap="round"/>

<circle cx="72" cy="253" r="13"
fill="#E6F1FF" stroke="#8199BD" stroke-width="4"/>

<circle cx="252" cy="190" r="13"
fill="#E6F1FF" stroke="#8199BD" stroke-width="4"/>

<path d="M129 279v10m62-10v10"
stroke="#8199BD" stroke-width="13"
stroke-linecap="round"/>

</svg>
<div class="robot-caption">
에이블 · AI 업무 파트너
</div>
</div>
"""


# ============================================================
# 13. 사이드바
# ============================================================

with st.sidebar:
    st.header("⚙️ 에이블 설정")

    daily_capacity = st.slider(
        "하루 업무 가능 시간",
        min_value=1,
        max_value=9,
        value=4
    )

    st.caption(
        "일정 추천: 평일 09:00~18:00 / 30분 단위"
    )

    st.divider()

    st.subheader("📋 회의록 상태")

    if st.session_state.meeting_text:
        st.success("회의록 준비됨")
    else:
        st.info("회의록 미업로드")

    st.metric(
        "💬 저장된 대화",
        len(st.session_state.messages) // 2
    )

    st.metric(
        "📌 추출된 업무",
        len(st.session_state.tasks)
    )

    with st.expander("🧠 응답 설정"):
        st.caption(
            "회의록 사실과 AI 제안을 구분하여 답변합니다."
        )

    with st.expander("🔊 음성 설정"):
        st.checkbox(
            "음성 답변 생성",
            key="voice_enabled"
        )

        st.checkbox(
            "음성 자동 재생",
            key="autoplay"
        )

    with st.expander("📅 캘린더 연결 안내"):
        st.caption(
            "Google 서비스 계정 및 Calendar ID 설정 필요"
        )

    st.divider()

    if st.button(
        "🔄 대화 초기화",
        use_container_width=True
    ):
        st.session_state.messages = []
        st.rerun()

    if st.button(
        "🗑️ 전체 초기화",
        use_container_width=True
    ):
        reset_all()
        st.rerun()


# ============================================================
# 14. 메인 화면
# ============================================================

st.title("🤖 에이블 | 나의 AI 업무 비서")

st.caption(
    "회의를 이해하고, 업무를 계획하고, "
    "일정까지 연결하는 나만의 AI 파트너"
)

home, dashboard, schedule, reports = st.tabs([
    "💬 AI 비서",
    "📊 업무 대시보드",
    "📅 일정 관리",
    "📥 보고서"
])


# ============================================================
# TAB 1. AI 비서
# ============================================================

with home:
    left, right = st.columns(
        [1, 1.15],
        gap="large"
    )

    with left:
        st.markdown(
            ROBOT_SVG,
            unsafe_allow_html=True
        )

        st.markdown("""
        <div class="assistant-note">
        🤖 안녕하세요! 저는 에이블이에요.<br>
        회의록을 올리고 음성이나 텍스트로 요청해 주세요.<br>
        업무 추출부터 일정 계획까지 도와드릴게요!
        </div>
        """, unsafe_allow_html=True)

        with st.expander(
            "📄 회의록 업로드 / 변경",
            expanded=not bool(st.session_state.meeting_text)
        ):
            uploaded = st.file_uploader(
                "회의록 TXT",
                type=["txt"],
                key="meeting_file"
            )

            if uploaded is not None:
                meeting_text = uploaded.getvalue().decode(
                    "utf-8-sig"
                )

                digest = hashlib.sha256(
                    meeting_text.encode("utf-8")
                ).hexdigest()

                if digest != st.session_state.meeting_hash:
                    st.session_state.meeting_text = meeting_text
                    st.session_state.meeting_hash = digest
                    st.session_state.tasks = []
                    st.session_state.subtasks = {}
                    st.session_state.calendar_proposals = {}
                    st.session_state.registered = {}
                    st.session_state.summary = ""

            if st.session_state.meeting_text:
                st.text_area(
                    "회의록 미리보기",
                    st.session_state.meeting_text,
                    height=120,
                    disabled=True
                )

                if st.button(
                    "✨ 회의록 요약 및 업무 추출",
                    type="primary",
                    use_container_width=True
                ):
                    try:
                        with st.spinner(
                            "회의록을 분석하는 중..."
                        ):
                            summary = chat(
                                "회의록을 주요 논의, 결정 사항, "
                                "김 대리 담당 업무, 마감일 중심으로 "
                                "정리해줘."
                            )

                            tasks = extract_tasks(
                                st.session_state.meeting_text
                            )

                            st.session_state.summary = summary
                            st.session_state.tasks = tasks

                        st.success(
                            f"분석 완료: {len(tasks)}개 업무 추출"
                        )

                        st.rerun()

                    except Exception as error:
                        st.error(
                            f"회의록 분석 실패: {error}"
                        )

        with st.expander("🎙️ 음성으로 말하기"):
            recording = st.audio_input(
                "에이블에게 음성으로 요청하기"
            )

            if st.button(
                "🎤 음성 인식 후 질문",
                use_container_width=True,
                disabled=recording is None
            ):
                try:
                    with st.spinner(
                        "음성 인식 및 답변 생성 중..."
                    ):
                        transcript = (
                            client.audio.transcriptions.create(
                                model="whisper-1",
                                file=(
                                    "voice.wav",
                                    recording.getvalue(),
                                    "audio/wav"
                                )
                            )
                        )

                        st.session_state.transcript = (
                            transcript.text
                        )

                        answer = chat(transcript.text)

                        if st.session_state.voice_enabled:
                            speech = (
                                client.audio.speech.create(
                                    model="tts-1",
                                    voice="alloy",
                                    input=answer[:4000]
                                )
                            )

                            st.session_state.audio_bytes = (
                                speech.read()
                            )

                        else:
                            st.session_state.audio_bytes = None

                    st.rerun()

                except Exception as error:
                    st.error(
                        f"음성 처리 실패: {error}"
                    )

            if st.session_state.transcript:
                st.caption(
                    "최근 음성 인식: "
                    + st.session_state.transcript
                )

            if st.session_state.audio_bytes:
                st.audio(
                    st.session_state.audio_bytes,
                    format="audio/mp3",
                    autoplay=st.session_state.autoplay
                )

    with right:
        st.subheader("💬 에이블과 실시간 대화")

        with st.container(
            border=True,
            height=440
        ):
            if not st.session_state.messages:
                st.write(
                    "🤖 안녕하세요! 저는 에이블이에요. "
                    "회의록을 올리고 필요한 업무를 물어보세요!"
                )

            for message in st.session_state.messages:
                with st.chat_message(
                    message["role"],
                    avatar=(
                        "🤖"
                        if message["role"] == "assistant"
                        else "👤"
                    )
                ):
                    st.markdown(message["content"])

        with st.form(
            "message_form",
            clear_on_submit=True
        ):
            question = st.text_area(
                "에이블에게 메시지 보내기",
                placeholder="어떤 업무를 도와드릴까요?",
                height=95
            )

            submitted = st.form_submit_button(
                "💬 에이블에게 보내기",
                type="primary",
                use_container_width=True
            )

        if submitted and question.strip():
            try:
                with st.spinner(
                    "에이블이 답변하는 중..."
                ):
                    chat(question.strip())

                st.rerun()

            except Exception as error:
                st.error(
                    f"대화 오류: {error}"
                )

        if st.session_state.summary:
            with st.expander("📝 최근 회의 요약"):
                st.markdown(
                    st.session_state.summary
                )


# ============================================================
# TAB 2. 업무 대시보드
# ============================================================

with dashboard:
    st.subheader("📊 업무 대시보드")

    tasks = st.session_state.tasks

    if not tasks:
        st.info(
            "AI 비서 탭에서 회의록을 업로드하고 "
            "업무 추출을 실행하세요."
        )

    else:
        total = len(tasks)

        completed = sum(
            task["status"] == "완료"
            for task in tasks
        )

        ongoing = sum(
            task["status"] == "진행 중"
            for task in tasks
        )

        overdue = sum(
            parse_date(task.get("deadline")) is not None
            and parse_date(task.get("deadline")) < TODAY
            and task["status"] != "완료"
            for task in tasks
        )

        cols = st.columns(4)

        for col, label, value in zip(
            cols,
            [
                "전체 업무",
                "완료",
                "진행 중",
                "기한 초과"
            ],
            [
                total,
                completed,
                ongoing,
                overdue
            ]
        ):
            col.metric(label, value)

        progress = completed / total

        st.progress(
            progress,
            text=f"전체 진행률 {progress:.0%}"
        )

        st.subheader("🎯 아이젠하워 매트릭스")

        cols = st.columns(2)

        quadrant_labels = {
            "Q1": "🔴 긴급·중요",
            "Q2": "🔵 중요·비긴급",
            "Q3": "🟠 긴급·비중요",
            "Q4": "⚪ 비긴급·비중요"
        }

        for idx, q in enumerate(
            ["Q1", "Q2", "Q3", "Q4"]
        ):
            with cols[idx % 2]:
                with st.container(border=True):
                    st.markdown(
                        f"**{quadrant_labels[q]}**"
                    )

                    selected = [
                        task for task in tasks
                        if quadrant(task) == q
                    ]

                    for task in selected:
                        st.write("•", task["name"])

                    if not selected:
                        st.caption("해당 업무 없음")

        st.subheader("👥 R&R 및 AI 업무 세분화")

        for task in tasks:
            with st.expander(
                f"{quadrant(task)} | "
                f"{task['name']} — {task['owner']}"
            ):
                st.write(
                    "**업무 내용:**",
                    task["description"]
                )

                st.write(
                    "**마감일:**",
                    task.get("deadline") or "기한 미정"
                )

                st.caption(
                    "분류 근거: " + task["reason"]
                )

                col1, col2 = st.columns(2)

                with col1:
                    statuses = [
                        "예정",
                        "진행 중",
                        "완료",
                        "보류"
                    ]

                    task["status"] = st.selectbox(
                        "진행 상태",
                        statuses,
                        index=statuses.index(
                            task.get("status", "예정")
                        ),
                        key=f"status_{task['id']}"
                    )

                with col2:
                    hours = st.number_input(
                        "예상 작업시간 (시간)",
                        min_value=0.0,
                        max_value=500.0,
                        value=float(
                            task.get("hours") or 0
                        ),
                        step=0.5,
                        key=f"hours_{task['id']}"
                    )

                    if hours != float(
                        task.get("hours") or 0
                    ):
                        task["hours"] = (
                            hours if hours > 0 else None
                        )

                        st.session_state.calendar_proposals.pop(
                            task["id"],
                            None
                        )

                if st.button(
                    "🧠 AI 세부 작업 생성",
                    key=f"split_{task['id']}"
                ):
                    try:
                        with st.spinner(
                            "업무를 단계별로 분해하고 있습니다..."
                        ):
                            steps = split_task(task)

                            st.session_state.subtasks[
                                task["id"]
                            ] = steps

                        st.rerun()

                    except Exception as error:
                        st.error(str(error))

                steps = st.session_state.subtasks.get(
                    task["id"], []
                )

                for idx, step in enumerate(steps, 1):
                    st.write(
                        f"{idx}. **{step['title']}** "
                        f"— {step['hours']}시간"
                    )

                    st.caption(step["details"])

        st.subheader("🚨 업무 과부하 점검")

        loads = analyze_workload(
            tasks,
            daily_capacity
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

            for day, hours in sorted(loads.items()):
                if hours > daily_capacity:
                    st.warning(
                        f"{day}: 예상 {hours:.1f}시간 / "
                        f"가능 {daily_capacity}시간 초과"
                    )

        else:
            st.caption(
                "분석 가능한 업무가 없거나 "
                "예상시간과 마감일이 미정입니다."
            )


# ============================================================
# TAB 3. 일정 관리
# ============================================================

with schedule:
    st.subheader(
        "📅 Google Calendar 충돌 감지 및 일정 추천"
    )

    st.caption(
        "기존 캘린더의 바쁜 시간을 조회하고, "
        "빈 시간에 30분 단위로 작업을 제안합니다. "
        "일정 등록은 사용자 승인 후 진행합니다."
    )

    if not st.session_state.tasks:
        st.info(
            "먼저 회의록에서 업무를 추출하세요."
        )

    else:
        for task in st.session_state.tasks:
            with st.expander(
                f"📌 {task['name']} | "
                f"{task.get('deadline') or '마감일 미정'}"
            ):
                st.write(
                    "담당자:",
                    task["owner"]
                )

                st.write(
                    "예상시간:",
                    task.get("hours") or "미정"
                )

                if task["status"] == "완료":
                    st.success("완료된 업무입니다.")
                    continue

                if st.button(
                    "🔎 캘린더 충돌 확인 및 시간 추천",
                    key=f"propose_{task['id']}"
                ):
                    try:
                        deadline = parse_date(
                            task.get("deadline")
                        )

                        if deadline is None or deadline < TODAY:
                            raise ValueError(
                                "유효한 미래 마감일이 필요합니다."
                            )

                        if (deadline - TODAY).days > 90:
                            raise ValueError(
                                "90일 이내 마감 업무만 지원합니다."
                            )

                        with st.spinner(
                            "Google Calendar 확인 중..."
                        ):
                            busy = calendar_busy(
                                TODAY,
                                deadline
                            )

                            reserved = []

                            for other_id, proposal in (
                                st.session_state.calendar_proposals.items()
                            ):
                                if other_id != task["id"]:
                                    reserved.extend(
                                        proposal["slots"]
                                    )

                            slots, remaining = propose_slots(
                                task,
                                daily_capacity,
                                busy,
                                reserved
                            )

                            st.session_state.calendar_proposals[
                                task["id"]
                            ] = {
                                "slots": slots,
                                "remaining": remaining,
                                "hours": task.get("hours"),
                                "deadline": task.get("deadline")
                            }

                    except Exception as error:
                        st.error(
                            f"캘린더 조회 실패: {error}"
                        )

                proposal = (
                    st.session_state.calendar_proposals.get(
                        task["id"]
                    )
                )

                if proposal:
                    if (
                        proposal["hours"] != task.get("hours")
                        or proposal["deadline"] != task.get("deadline")
                    ):
                        st.warning(
                            "업무 조건이 변경되었습니다. "
                            "다시 추천해 주세요."
                        )
                        continue

                    slots = proposal["slots"]
                    remaining = proposal["remaining"]

                    st.write(
                        f"추천 작업 블록: {len(slots)}개 / "
                        f"미배정 {remaining / 60:.1f}시간"
                    )

                    if slots:
                        df = pd.DataFrame([
                            {
                                "날짜": start.strftime("%Y-%m-%d"),
                                "시작": start.strftime("%H:%M"),
                                "종료": end.strftime("%H:%M")
                            }
                            for start, end in slots
                        ])

                        st.dataframe(
                            df,
                            hide_index=True,
                            use_container_width=True
                        )

                    if remaining:
                        st.warning(
                            "마감일까지 충분한 시간이 없습니다. "
                            "전체 업무를 배치할 수 없어 "
                            "일정을 등록하지 않습니다."
                        )

                    elif slots:
                        signature = hashlib.sha256(
                            (
                                str(task["id"])
                                + str(slots)
                                + str(task.get("hours"))
                            ).encode()
                        ).hexdigest()

                        if signature in st.session_state.registered:
                            st.success(
                                "현재 세션에서 이미 등록한 일정입니다."
                            )

                        elif st.button(
                            "✅ 추천 일정 Google Calendar 등록 승인",
                            type="primary",
                            key=f"commit_{task['id']}"
                        ):
                            try:
                                fresh = calendar_busy(
                                    TODAY,
                                    parse_date(task["deadline"])
                                )

                                if any(
                                    overlaps(start, end, fresh)
                                    for start, end in slots
                                ):
                                    st.session_state.calendar_proposals.pop(
                                        task["id"],
                                        None
                                    )

                                    st.error(
                                        "새로운 일정이 감지되었습니다. "
                                        "다시 추천해 주세요."
                                    )

                                else:
                                    created = insert_calendar_slots(
                                        task,
                                        slots
                                    )

                                    st.session_state.registered[
                                        signature
                                    ] = [
                                        event.get("htmlLink", "")
                                        for event in created
                                    ]

                                    st.success(
                                        f"{len(created)}개 작업 블록 "
                                        "등록 완료!"
                                    )

                            except Exception as error:
                                st.error(
                                    f"일정 등록 실패: {error}"
                                )


# ============================================================
# TAB 4. 보고서 다운로드
# ============================================================

with reports:
    st.subheader("📥 업무 보고서 다운로드")

    st.caption(
        "회의 요약, 담당자, 마감일, AI 세부 작업, "
        "업무 진행 상태를 포함합니다."
    )

    if (
        not st.session_state.meeting_text
        and not st.session_state.tasks
    ):
        st.info(
            "보고서에 포함할 회의록이나 업무가 없습니다."
        )

    else:
        report = report_markdown()

        st.download_button(
            "📄 Markdown 업무 보고서 다운로드",
            data=report,
            file_name=f"aivle_report_{TODAY}.md",
            mime="text/markdown",
            use_container_width=True
        )

        if st.session_state.tasks:
            csv_data = make_task_df(
                st.session_state.tasks
            ).to_csv(
                index=False
            ).encode("utf-8-sig")

            st.download_button(
                "📊 CSV 업무 목록 다운로드",
                data=csv_data,
                file_name=f"aivle_tasks_{TODAY}.csv",
                mime="text/csv",
                use_container_width=True
            )

        with st.expander("보고서 미리보기"):
            st.markdown(report)


# ============================================================
# 15. 하단 안내
# ============================================================

st.divider()

st.caption(
    "🤖 에이블 AI 업무 비서 Pro | "
    "회의록 분석 · 실시간 대화 · AI 업무 세분화 · "
    "업무 대시보드 · Google Calendar · 보고서 다운로드"
)
