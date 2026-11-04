import os
import json
import math
from datetime import datetime, date, timedelta
from zoneinfo import ZoneInfo

import streamlit as st
from openai import OpenAI

KST = ZoneInfo('Asia/Seoul')
client = OpenAI()

st.set_page_config(page_title='에이블 AI 업무 비서', page_icon='🤖', layout='wide')
st.title('🤖 에이블 AI 업무 비서')
st.caption('음성 회의 요약 · R&R · Eisenhower Matrix · 업무 과부하 · Action Scheduler')

sys_role = '''당신은 김 대리의 개인 비서 '에이블'입니다.
김 대리가 회의에 참석하지 못했을 때 회의 내용을 정리해 업무를 놓치지 않도록 돕습니다.
[답변 규칙]
- 한국어로 정중하고 간결하게 답변합니다.
- 회의록에 없는 사실을 지어내지 않습니다.
- 요약은 주제 → 논의 내용 → 결정 사항 순서로 작성합니다.
- 김 대리 담당 업무는 요약과 별도로 업무명, 세부 내용, 기한을 적습니다.
- 기한이 명시되지 않았다면 '기한 미정'으로 표시합니다.
- 결정 사항에는 예산, 일정, 채널 등 구체적인 수치와 항목을 포함합니다.
- 업무의 방향과 조건을 포함하고, 기한을 추측하지 않습니다.
- 예상 작업 시간과 우선순위는 추정/판단임을 구분해 설명합니다.
- 캘린더에 등록하기 전 반드시 사용자 승인을 받습니다.'''

for key, default in {'messages': [], 'meeting_text': '', 'tasks': [], 'answer': '', 'transcript': '', 'calendar_registered': {}, 'meeting_signature': ''}.items():
    if key not in st.session_state:
        st.session_state[key] = default

TASK_TOOL = [{
    'type': 'function', 'name': 'extract_tasks',
    'description': '회의록에 명시된 업무를 추출하고 담당자, 기한, 우선순위 판단 근거를 구조화합니다.',
    'parameters': {
        'type': 'object', 'properties': {'tasks': {
            'type': 'array', 'items': {
                'type': 'object', 'properties': {
                    'task_name': {'type': 'string'},
                    'description': {'type': 'string'},
                    'owner': {'type': 'string'},
                    'deadline': {'type': ['string', 'null'], 'description': '명시적으로 확인된 YYYY-MM-DD 기한. 모호하거나 없으면 null'},
                    'urgent': {'type': 'boolean'},
                    'important': {'type': 'boolean'},
                    'reason': {'type': 'string'},
                    'estimated_hours': {'type': ['number', 'null'], 'description': '명시된 소요시간만 기록. 없으면 null'}
                },
                'required': ['task_name', 'description', 'owner', 'deadline', 'urgent', 'important', 'reason', 'estimated_hours'],
                'additionalProperties': False
            }
        }}, 'required': ['tasks'], 'additionalProperties': False
    }, 'strict': True
}]


def quadrant(task):
    u, i = task['urgent'], task['important']
    if u and i:
        return 'Q1 긴급·중요 — 즉시 실행'
    if i:
        return 'Q2 비긴급·중요 — 일정 계획'
    if u:
        return 'Q3 긴급·비중요 — 위임 검토'
    return 'Q4 비긴급·비중요 — 보류 검토'


def parse_deadline(value):
    if not value:
        return None
    try:
        return date.fromisoformat(value)
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


def plan_task(task, capacity):
    end = parse_deadline(task['deadline'])
    hours = task['estimated_hours']
    if end is None or hours is None or hours <= 0:
        return None
    days_needed = math.ceil(hours / capacity)
    start = end
    # Start at the deadline and count backward through weekdays.
    counted = 1 if start.weekday() < 5 else 0
    while counted < days_needed:
        start -= timedelta(days=1)
        if start.weekday() < 5:
            counted += 1
    return start, end


def workload_report(tasks, capacity):
    today = datetime.now(KST).date()
    loads, alerts, unknown = {}, [], []
    for task in tasks:
        if not ('김' in task['owner'] and '대리' in task['owner']):
            continue
        schedule = plan_task(task, capacity)
        if not schedule:
            unknown.append(task['task_name'])
            continue
        start, end = schedule
        if end < today:
            alerts.append(f"'{task['task_name']}' 마감일이 지났습니다 ({end}).")
            continue
        days = working_days(max(today, start), end)
        if not days:
            alerts.append(f"'{task['task_name']}'에 남은 평일 작업일이 없습니다.")
            continue
        per_day = task['estimated_hours'] / len(days)
        for day in days:
            loads[day] = loads.get(day, 0) + per_day
    for day, hours in sorted(loads.items()):
        if hours > capacity:
            alerts.append(f'{day}: 예상 {hours:.1f}시간 / 가용 {capacity}시간 초과')
    return loads, alerts, unknown


def get_calendar_service():
    from google.oauth2 import service_account
    from googleapiclient.discovery import build
    info = dict(st.secrets['gcp_service_account'])
    credentials = service_account.Credentials.from_service_account_info(
        info, scopes=['https://www.googleapis.com/auth/calendar.events']
    )
    return build('calendar', 'v3', credentials=credentials, cache_discovery=False)


def register_calendar(task, start, end):
    # All-day planning span. The end date is exclusive in Google Calendar.
    body = {
        'summary': '[에이블] ' + task['task_name'],
        'description': (
            f"담당자: {task['owner']}\n업무: {task['description']}\n"
            f"분류: {quadrant(task)}\n예상 작업 시간: {task['estimated_hours']}시간\n"
            f"마감일: {end}\n시작일은 가용시간에 따른 계산값이며 확정 일정이 아닙니다."
        ),
        'start': {'date': start.isoformat()},
        'end': {'date': (end + timedelta(days=1)).isoformat()},
    }
    return get_calendar_service().events().insert(
        calendarId=st.secrets['calendar_id'], body=body
    ).execute()

with st.sidebar:
    st.header('⚙️ 분석 설정')
    daily_capacity = st.slider('하루 업무 가능 시간', 1, 12, 4)
    st.caption('마감 역산은 평일 기준입니다. 공휴일·기존 일정은 반영하지 않습니다.')
    st.caption('Google Calendar 등록은 서비스 계정 설정 후 사용 가능합니다.')
    if st.button('세션 초기화'):
        for key in ('messages', 'tasks', 'answer', 'transcript', 'meeting_text', 'calendar_registered', 'meeting_signature'):
            st.session_state[key] = {} if key == 'calendar_registered' else ([] if key in ('messages', 'tasks') else '')
        st.rerun()

with st.form('task_form'):
    uploaded_file = st.file_uploader('회의록 텍스트 파일 업로드', type=['txt'])
    audio_value = st.audio_input('무엇을 도와드릴까요?')
    submit = st.form_submit_button('🚀 실행')

meeting_text = st.session_state.meeting_text

if meeting_text:
    with st.expander('📄 저장된 회의록 확인'):
        st.text_area('회의록', meeting_text, height=180, disabled=True)

# 텍스트 후속 질문도 허용: 이전 대화를 함께 전달하여 멀티턴 구현
chat_text = st.chat_input('이전 회의에 관해 추가로 질문하세요 (예: 제 업무만 알려줘)')

if uploaded_file is not None:
    new_text = uploaded_file.getvalue().decode('utf-8-sig')
    if new_text != st.session_state.meeting_text:
        st.session_state.meeting_text = new_text
        st.session_state.tasks = []
        st.session_state.meeting_signature = ''
meeting_text = st.session_state.meeting_text

user_request = None
if submit:
    if audio_value is None:
        st.warning('음성을 녹음해주세요.')
    else:
        with st.spinner('음성을 텍스트로 변환 중...'):
            transcript = client.audio.transcriptions.create(
                model='whisper-1', file=('audio.wav', audio_value.getvalue(), 'audio/wav')
            )
        st.session_state.transcript = transcript.text
        user_request = transcript.text
elif chat_text:
    user_request = chat_text

if user_request:
    # 회의록은 시스템 문맥으로 매 턴 유지하고, 대화는 최근 20개 메시지까지 전달
    system_content = (
        sys_role + f"\n현재 날짜(한국): {datetime.now(KST).date()}"
        + f"\n[저장된 회의록]\n{meeting_text if meeting_text else '제공되지 않음'}"
    )
    input_messages = [{'role': 'system', 'content': system_content}]
    input_messages.extend(st.session_state.messages[-20:])
    input_messages.append({'role': 'user', 'content': user_request})

    with st.spinner('답변을 준비하고 있습니다...'):
        response = client.responses.create(
            model='gpt-4o-mini', input=input_messages, temperature=0
        )
        answer = response.output_text
        st.session_state.answer = answer
        st.session_state.messages.append({'role': 'user', 'content': user_request})
        st.session_state.messages.append({'role': 'assistant', 'content': answer})

        # 회의록이 새로 업로드된 경우에만 업무를 재추출해 기존 분석 결과 보존
        if meeting_text and meeting_text != st.session_state.meeting_signature:
            task_response = client.responses.create(
                model='gpt-4o-mini',
                input=[{'role': 'system', 'content': system_content},
                       {'role': 'user', 'content':
                        '저장된 회의록에서 실제로 명시된 업무만 추출하세요. '
                        '담당자 미정이면 담당자 미정, 불명확한 기한과 소요시간은 null로 기록하세요. '
                        '긴급도와 중요도 판단 근거를 포함하세요.'}],
                tools=TASK_TOOL,
                tool_choice={'type': 'function', 'name': 'extract_tasks'},
                temperature=0
            )
            tasks = []
            for item in task_response.output:
                if item.type == 'function_call' and item.name == 'extract_tasks':
                    tasks = json.loads(item.arguments).get('tasks', [])
            st.session_state.tasks = tasks
            st.session_state.meeting_signature = meeting_text

if st.session_state.transcript:
    st.subheader('🎤 음성 인식 결과')
    st.write(st.session_state.transcript)

if st.session_state.answer:
    st.subheader('📋 AI 답변')
    st.write(st.session_state.answer)

if st.session_state.tasks:
    tasks = st.session_state.tasks
    st.subheader('📊 Eisenhower Matrix')
    columns = st.columns(2)
    for idx, (prefix, title) in enumerate([
        ('Q1', '🔴 Q1 긴급·중요'), ('Q2', '🔵 Q2 비긴급·중요'),
        ('Q3', '🟠 Q3 긴급·비중요'), ('Q4', '⚪ Q4 비긴급·비중요')
    ]):
        with columns[idx % 2]:
            st.markdown(f'**{title}**')
            matched = [t for t in tasks if quadrant(t).startswith(prefix)]
            if matched:
                for task in matched:
                    st.write('•', task['task_name'])
            else:
                st.caption('해당 업무 없음')

    st.subheader('👥 R&R 및 업무 상세')
    for idx, task in enumerate(tasks):
        with st.expander(f"{quadrant(task)} | {task['task_name']}"):
            st.write('담당자:', task['owner'])
            st.write('세부 내용:', task['description'])
            st.write('기한:', task['deadline'] or '기한 미정')
            st.write('분류 근거:', task['reason'])
            st.write('예상 소요시간:',
                     f"{task['estimated_hours']}시간" if task['estimated_hours'] is not None
                     else '미정 (사용자 입력 필요)')

    st.subheader('🚨 김 대리 업무 과부하 분석')
    loads, alerts, unknown = workload_report(tasks, daily_capacity)
    for alert in alerts:
        st.warning(alert)
    if unknown:
        st.info('업무량 계산 제외(기한/시간 미확인): ' + ', '.join(unknown))
    if not alerts and loads:
        st.success('입력된 예상 소요시간 기준, 일별 업무량 초과가 감지되지 않았습니다.')
    if loads:
        import pandas as pd
        df = pd.DataFrame([
            {'날짜': str(day), '예상 업무시간': hours}
            for day, hours in sorted(loads.items())
        ]).set_index('날짜')
        st.bar_chart(df)

    st.subheader('📅 Action Scheduler')
    st.caption('등록 버튼을 누르면 표시된 업무 일정이 Google Calendar에 생성됩니다.')
    for idx, task in enumerate(tasks):
        schedule = plan_task(task, daily_capacity)
        if schedule is None:
            st.info(f"{task['task_name']}: 마감일 또는 작업시간이 없어 자동 역산 불가")
            continue
        start, end = schedule
        st.write(f"**{task['task_name']}** | 권장 시작일: {start} | 마감일: {end}")
        if start < datetime.now(KST).date():
            st.warning('권장 시작일이 이미 지났습니다. 일정 조정이 필요합니다.')
        key = f"{task['task_name']}|{task['owner']}|{end}"
        if key in st.session_state.calendar_registered:
            st.success('이 세션에서 등록 완료')
            st.link_button('캘린더에서 보기', st.session_state.calendar_registered[key])
        elif st.button(f"📅 캘린더 등록 승인: {task['task_name']}", key=f'calendar_{idx}'):
            try:
                event = register_calendar(task, start, end)
                st.session_state.calendar_registered[key] = event.get('htmlLink', '')
                st.success('Google Calendar에 등록했습니다.')
                st.rerun()
            except Exception as exc:
                st.error(f'캘린더 등록 실패: {exc}')

with st.expander('💬 대화 기록'):
    for message in st.session_state.messages:
        st.markdown(f"**{'김 대리' if message['role'] == 'user' else '에이블'}:** {message['content']}")
