
import os, json, smtplib, re
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import calendar as pycal

# 업무 고유 ID 생성 및 유사 업무 비교
import uuid
import hashlib
from difflib import SequenceMatcher

# ============================================================
# 다양한 업무 문서 파일 형식 지원
# ============================================================
# TXT  : 일반 텍스트
# PDF  : PDF 문서
# Excel: XLSX, XLS, CSV
# Word : DOCX
# 한글 : HWPX, HWP
# 음성 : MP3, WAV, M4A
# ============================================================

import io
import zipfile
import tempfile
import subprocess
import xml.etree.ElementTree as ET

from pathlib import Path
from pypdf import PdfReader
from docx import Document

import pandas as pd
import streamlit as st
from openai import OpenAI
from email.mime.text import MIMEText

# ── 기본 설정 ─────────────────────────────────────
# 키는 환경변수에서 자동
client = OpenAI()

# 이메일 요청 시 받을 주소 (음성 인식 오류를 피하려고 코드에 고정)
MY_EMAIL = "soulsky0218@gmail.com" #수정필요 (기본수신 이메일)

# 에이블 대표 이미지 경로 (app.py와 같은 폴더에 둔 파일 이름)
IMAGE_PATH = "able.png"

# 사이드바 메뉴 (아래 셀들도 이 이름으로 화면을 나눔)
PAGES = ["🎙️ AI 비서", "📊 업무 우선순위 분석", "📅 마감 캘린더", "📨 팔로업 메일", "🎧 출근길 브리핑",
         "🍃 오늘의 할 일", "🗂️ 회의 자료실"]

# 대화 기록 최대 개수 (질문 1 + 답변 1 = 2개 → 10이면 최근 5번의 대화)
MAX_HISTORY = 10

# 주석 추가 ~~~
def kst_today():
    return datetime.now(ZoneInfo("Asia/Seoul")).date()

def kst_now_str():
    return datetime.now(ZoneInfo("Asia/Seoul")).strftime("%Y-%m-%d %H:%M:%S")


# 음성용 텍스트 정리: 마크다운 기호를 빼야 TTS가 매끄럽게 읽음
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
    # Streamlit Secrets에서 Gmail 발송 계정 정보 가져오기
    sender = st.secrets.get("GMAIL_ADDRESS", os.getenv("GMAIL_ADDRESS"))
    password = st.secrets.get("GMAIL_APP_PASSWORD", os.getenv("GMAIL_APP_PASSWORD"))

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

# ── 이메일 요청인지 판단 ('메일' + '보내다/전송/발송'이 함께 있어야 요청으로 봄) ──
def wants_email(question):
    q = question.replace(" ", "")                        # '이 메일로'처럼 띄어 써도 인식
    has_mail = "메일" in q or "email" in q.lower()
    has_send = re.search(r"보내|전송|발송|송부", q) is not None
    return has_mail and has_send

# ── 에이블에게 질문하기 (음성·텍스트 대화 공통) ──
def ask_able(question, current_meeting):
    active_doc_name = st.session_state.active_doc
    history = st.session_state.messages_dict.get(active_doc_name, [])
    """질문 → LLM 호출(이메일 도구 포함) → (답변, 이메일 발송 결과) 반환"""
    # 프롬프트: 질문 + 고정 이메일 주소 + 회의록
    prompt = f"""
{question}

[김 대리 이메일 주소]
{MY_EMAIL}

[미팅 내용: {active_doc_name}]
{current_meeting}
"""
    # 1단계: 역할 → 이전 대화(멀티턴) → 이번 질문
    request = {
        "model": "gpt-4o-mini",
        "input": [{"role": "system", "content": sys_role}]
                 + history
                 + [{"role": "user", "content": prompt}],
        "temperature": 0
    }
    # 이메일 도구는 '메일을 보내 달라'는 요청이 확실할 때만 전달 (파이썬이 판단)
    #  - 요청이 없으면 도구 자체를 주지 않아서, 이전 대화에 메일 요청이 있어도 다시 보내지 않음
    if wants_email(question):
        request["tools"] = tools
        request["tool_choice"] = {"type": "function", "name": "send_email"}
        request["parallel_tool_calls"] = False     # 메일 한 통만
    response = client.responses.create(**request)

    tool_calls = [item for item in response.output if item.type == "function_call"]
    if not tool_calls:
        return response.output_text, None

    # 2단계: LLM이 정한 인자(to, subject, body)로 이메일 함수 실행
    tool_outputs = []
    email = None
    for item in tool_calls:
        if item.name == "send_email":
            args = json.loads(item.arguments)
            result = send_email(**args)
            email = {"to": args["to"], **result}
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
    return final_response.output_text, email

# ══════════════════════════════════════════════════
# 화면 공통: 제목 · 소개 · 세션 상태 · 사이드바 메뉴
# ══════════════════════════════════════════════════
st.set_page_config(page_title="AI 개인비서 에이블", page_icon="🤖", layout="wide")
st.title("🤖 AI 개인비서 에이블")
st.caption("다중 회의록/업무자료 분석, 스마트 캘린더, 실시간 대화 및 지출/목표 대시보드를 제공합니다.")

# 소개 영역: 왼쪽 이미지 / 오른쪽 역할·핵심 기능
left, right = st.columns([1, 5])
with left:
    if os.path.exists(IMAGE_PATH):
        st.image(IMAGE_PATH, width=180)
    else:
        st.info("🖼️ 이미지 준비 중")
with right:
    st.markdown("""
#### 김 대리의 AI 개인비서, 에이블
- 📝 회의록을 분석해 **요약하고 내 업무를 찾아** 드립니다.
- 🎙️ **음성이나 대화로 질문**하면 글과 음성으로 답하고, 요청 시 **이메일**로 보냅니다.
- 📅 업무 **우선순위**를 정리하고 마감일을 **캘린더**에 표시합니다.
""")
st.divider()

# 세션 상태 초기화 (S04: 처음 한 번만)
if "messages" not in st.session_state:
    st.session_state.messages = []        # 대화 히스토리 (음성·텍스트 대화 공통)
if "meeting_text" not in st.session_state:
    st.session_state.meeting_text = ""    # 업로드한 회의록 (메뉴를 옮겨도 유지)
if "notice" not in st.session_state:
    st.session_state.notice = None        # 최근 처리 결과 (인식된 질문, 이메일 발송 결과)
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None    # 최근 답변 음성
    st.session_state.play_audio = False   # 새 답변일 때만 자동 재생

# 문서 업로드 및 문서별 대화 기록 초기화 (사이드바 실행 전에 필요)
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = {}       # {문서명: 본문내용}
# ============================================================
# 업로드 위젯 및 삭제 확인 상태 관리
# ============================================================

# 파일 업로드 위젯 초기화 버전
if "upload_widget_version" not in st.session_state:
    st.session_state.upload_widget_version = 0

# 삭제 확인 대기 문서
if "sidebar_delete_target" not in st.session_state:
    st.session_state.sidebar_delete_target = None
if "active_doc" not in st.session_state:
    st.session_state.active_doc = ""          # 현재 선택한 문서
if "archived_meetings" not in st.session_state:
    st.session_state.archived_meetings = []   # 삭제한 문서도 보존하는 아카이브
# ============================================================
# 문서 삭제 2단계 확인 상태 관리
# ============================================================
# None : 삭제 요청 없음
# 문서 ID : 해당 문서의 삭제 확인 대기
# ============================================================
if "delete_target" not in st.session_state:
    st.session_state.delete_target = None
if "messages_dict" not in st.session_state:
    st.session_state.messages_dict = {}       # 문서별 대화 기록


# 동일 문서 중복 저장 방지
# 파일명만 비교하지 않고 실제 문서 내용을 기준으로 확인
import hashlib

def get_document_hash(content):
    return hashlib.sha256(
        content.encode("utf-8")
    ).hexdigest()

# ============================================================
# 다양한 파일 형식의 내용을 텍스트로 변환
# ============================================================
# 목적:
# 사용자가 여러 종류의 마케팅 업무자료를 업로드하면
# 파일 형식에 맞게 내용을 추출하여 AI 분석에 활용
#
# 지원:
# TXT, CSV, PDF, XLSX, XLS, DOCX, HWPX
# MP3, WAV, M4A
#
# HWP는 별도의 변환 도구가 필요할 수 있음
# ============================================================

def extract_file_content(uploaded_file):

    # 1. 파일 확장자 확인
    filename = uploaded_file.name
    ext = Path(filename).suffix.lower()

    # 2. 파일을 바이트 데이터로 읽기
    file_bytes = uploaded_file.getvalue()

    # ========================================================
    # TXT 파일
    # ========================================================
    if ext == ".txt":

        for encoding in ["utf-8-sig", "cp949", "euc-kr"]:

            try:
                return file_bytes.decode(encoding)

            except UnicodeDecodeError:
                continue

        raise ValueError("TXT 파일의 문자 인코딩을 확인해 주세요.")

    # ========================================================
    # CSV 파일
    # ========================================================
    elif ext == ".csv":

        for encoding in ["utf-8-sig", "cp949", "euc-kr"]:

            try:
                df = pd.read_csv(
                    io.BytesIO(file_bytes),
                    encoding=encoding
                )
                return df.to_csv(index=False)

            except UnicodeDecodeError:
                continue

        raise ValueError("CSV 파일의 문자 인코딩을 확인해 주세요.")

    # ========================================================
    # PDF 파일
    # ========================================================
    elif ext == ".pdf":

        reader = PdfReader(io.BytesIO(file_bytes))

        pages = []

        for i, page in enumerate(reader.pages):

            page_text = page.extract_text() or ""

            pages.append(
                f"[페이지 {i + 1}]\n{page_text}"
            )

        return "\n\n".join(pages)

    # ========================================================
    # Excel 파일
    # ========================================================
    elif ext in [".xlsx", ".xls"]:

        sheets = pd.read_excel(
            io.BytesIO(file_bytes),
            sheet_name=None
        )

        result = []

        for sheet_name, df in sheets.items():

            result.append(
                f"[시트: {sheet_name}]\n"
                + df.to_csv(index=False)
            )

        return "\n\n".join(result)

    # ========================================================
    # Word 파일
    # ========================================================
    elif ext == ".docx":

        doc = Document(io.BytesIO(file_bytes))

        paragraphs = [
            p.text
            for p in doc.paragraphs
            if p.text.strip()
        ]

        # Word 문서의 표도 함께 읽기
        for table in doc.tables:

            for row in table.rows:

                paragraphs.append(
                    " | ".join(
                        cell.text
                        for cell in row.cells
                    )
                )

        return "\n".join(paragraphs)

    # ========================================================
    # 한글 HWPX 파일
    # ========================================================
    elif ext == ".hwpx":

        result = []

        with zipfile.ZipFile(io.BytesIO(file_bytes)) as z:

            section_files = sorted(
                name
                for name in z.namelist()
                if name.startswith("Contents/section")
                and name.endswith(".xml")
            )

            for section_name in section_files:

                root = ET.fromstring(z.read(section_name))

                texts = [
                    element.text
                    for element in root.iter()
                    if element.tag.split("}")[-1] == "t"
                    and element.text
                ]

                result.append("\n".join(texts))

        if not result:
            raise ValueError("HWPX 문서에서 본문을 찾지 못했습니다.")

        return "\n\n".join(result)

    # ========================================================
    # 한글 HWP 파일
    # ========================================================
    elif ext == ".hwp":

        # HWP 바이너리 파일은 HWPX와 내부 구조가 다름
        # pyhwp의 hwp5txt 도구를 이용하여 텍스트 변환
        with tempfile.TemporaryDirectory() as tmpdir:

            path = Path(tmpdir) / "uploaded.hwp"
            path.write_bytes(file_bytes)

            try:
                result = subprocess.run(
                    ["hwp5txt", str(path)],
                    capture_output=True,
                    timeout=30,
                    check=True
                )

            except FileNotFoundError:
                raise ValueError(
                    "HWP 변환 도구가 설치되지 않았습니다. "
                    "pyhwp 설치를 확인하거나 HWPX로 변환해 주세요."
                )

            except subprocess.CalledProcessError:
                raise ValueError(
                    "HWP 파일 변환에 실패했습니다. "
                    "파일 형식을 확인해 주세요."
                )

            return result.stdout.decode(
                "utf-8",
                errors="replace"
            )

    # ========================================================
    # 음성 녹음 파일
    # ========================================================
    elif ext in [".mp3", ".wav", ".m4a"]:

        # OpenAI Speech-to-Text API 사용
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=(filename, file_bytes)
        )

        return transcript.text

    # ========================================================
    # 지원하지 않는 파일 (파일별 처리 한계 : 구형 word, doc 파일은 지원x, .hwp-pyhwp 변환 도구가 정상 설치된 환경에서만 사용 가능, 녹음파일-OPENAI API의 파일 크기 제한이 있음)
    # ========================================================
    else:

        raise ValueError(
            f"지원하지 않는 파일 형식입니다: {ext}"
        )

# 사이드바 메뉴: 누른 메뉴의 기능만 화면에 표시
with st.sidebar:
    st.header("🤖 에이블 메뉴")
    page = st.radio("기능 선택", PAGES, label_visibility="collapsed")
    st.divider()

    # 회의록 업로드 (사이드바에 있어서 어느 메뉴에서든 올리고 바꿀 수 있음)
    st.subheader("📄 문서 업로드")
        # 지원하는 마케팅 업무자료 파일 형식
    uploaded_files = st.file_uploader(
        "업무자료 업로드",
        type=[
            "txt", "csv",
            "pdf",
            "xlsx", "xls",
            "docx",
            "hwp", "hwpx",
            "mp3", "wav", "m4a"
        ],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"work_file_upload_{st.session_state.upload_widget_version}"
    )

    # ============================================================
    # 업로드한 문서의 내용 추출 및 작업 목록 등록
    # ============================================================
    # 1. 파일 형식별 텍스트 추출
    # 2. 동일 파일 재등록 방지
    # 3. 현재 작업 대상 문서 자동 선택
    # 4. 추출 실패 시 오류 메시지 표시
    # ============================================================

    if uploaded_files:

        loaded_count = 0

        for uf in uploaded_files:

            # 기존 작업 목록에 없는 파일만 추가
            if uf.name not in st.session_state.uploaded_docs:

                try:

                    # 파일 형식별 텍스트 추출
                    content = extract_file_content(uf)

                    # 추출된 내용이 없는 경우
                    if not content.strip():

                        st.warning(
                            f"'{uf.name}' 파일에서 "
                            "분석할 텍스트를 찾지 못했습니다."
                        )

                        continue

                    # 기존 작업 목록에 등록
                    st.session_state.uploaded_docs[
                        uf.name
                    ] = content

                    # 현재 작업 대상으로 선택
                    st.session_state.active_doc = uf.name

                    loaded_count += 1

                except Exception as e:

                    st.error(
                        f"'{uf.name}' 파일 처리 오류: {e}"
                    )

        if loaded_count > 0:

            st.success(
                f"{loaded_count}개 파일 로드 완료!"
            )

        st.success(f"{len(uploaded_files)}개 파일 로드 완료!")

    # ============================================================
    # 현재 작업 대상 문서 선택 및 삭제
    # ============================================================
    # 기능:
    # 1. 업로드한 문서 중 작업 대상 선택
    # 2. 다른 문서로 변경
    # 3. 선택한 문서 삭제 요청
    # 4. 삭제 취소 또는 최종 삭제
    #
    # 주의:
    # 작업 목록에서 삭제해도 회의 자료실의
    # 보관 기록은 유지합니다.
    # ============================================================

    if st.session_state.uploaded_docs:

        st.divider()

        st.subheader("🎯 현재 작업 대상 문서")

        doc_names = list(
            st.session_state.uploaded_docs.keys()
        )

        # 현재 선택한 문서가 없으면 첫 번째 문서를 선택
        if st.session_state.active_doc not in doc_names:
            st.session_state.active_doc = doc_names[0]

        # 1. 작업 대상 문서 선택
        selected_doc = st.selectbox(
            "작업할 문서를 선택하세요.",
            doc_names,
            index=doc_names.index(
                st.session_state.active_doc
            ),
            key="active_document_selector"
        )

        # 선택 문서 변경
        if selected_doc != st.session_state.active_doc:

            st.session_state.active_doc = selected_doc

            # 다른 문서를 선택하면 삭제 요청 취소
            st.session_state.sidebar_delete_target = None

            st.rerun()

        # 2. 삭제 요청 버튼
        if (
            st.session_state.sidebar_delete_target
            != selected_doc
        ):

            if st.button(
                "🗑️ 선택한 문서 삭제",
                key="sidebar_delete_request",
                use_container_width=True
            ):

                st.session_state.sidebar_delete_target = (
                    selected_doc
                )

                st.rerun()

        # 3. 삭제 확인 단계
        else:

            st.warning(
                f"⚠️ '{selected_doc}' 문서를 삭제하시겠습니까?"
            )

            col_cancel, col_delete = st.columns(2)

            # 삭제 취소
            with col_cancel:

                if st.button(
                    "취소",
                    key="sidebar_delete_cancel",
                    use_container_width=True
                ):

                    st.session_state.sidebar_delete_target = None

                    st.rerun()

            # 최종 삭제
            with col_delete:

                if st.button(
                    "최종 삭제",
                    key="sidebar_delete_confirm",
                    use_container_width=True
                ):

                    # 작업 목록에서 선택 문서 제거
                    st.session_state.uploaded_docs.pop(
                        selected_doc,
                        None
                    )

                    # 문서별 대화 기록도 작업 목록에서 제거
                    st.session_state.messages_dict.pop(
                        selected_doc,
                        None
                    )

                    # 남은 문서 중 첫 번째 문서 선택
                    remaining_docs = list(
                        st.session_state.uploaded_docs.keys()
                    )

                    st.session_state.active_doc = (
                        remaining_docs[0]
                        if remaining_docs
                        else ""
                    )

                    # 삭제 확인 상태 초기화
                    st.session_state.sidebar_delete_target = None

                    # 업로드 위젯 초기화
                    # 삭제한 파일이 다시 자동 등록되는 현상 방지
                    st.session_state.upload_widget_version += 1

                    st.rerun()

    else:

        st.info("현재 등록된 작업 문서가 없습니다.")

    st.divider()
    active_history = st.session_state.messages_dict.get(st.session_state.active_doc, [])
    st.subheader("📋 히스토리 요약")
    st.markdown(f"**현재 문서:** `{st.session_state.active_doc or '선택 없음'}`")
    st.markdown(f"**대화 기록:** {len(active_history) // 2}건 (최대 {MAX_HISTORY//2}회)")

current_meeting = st.session_state.uploaded_docs.get(st.session_state.active_doc, "")


# ══════════════════════════════════════════════════
# 메뉴 1: AI 비서 (회의록 · 음성 요청 · 실시간 대화)
# ══════════════════════════════════════════════════
if page == PAGES[0]:
    col_left, col_right = st.columns([1, 1], gap="large")

    # ── 왼쪽: 회의록 업로드 + 음성 요청 ──
    with col_left:
        st.subheader("📄 문서 업로드")
        if not current_meeting:
            st.info("사이드바에서 문서 파일을 업로드하면 여기서 내용을 볼 수 있습니다.")
        else:
            with st.expander("문서 원문 미리보기", expanded=True):     # 눌러야 내용이 펼쳐짐
                st.text_area("문서 내용", current_meeting, height=230, label_visibility="collapsed")

        st.divider()
        st.subheader("🎙️ 음성으로 요청하기")
        audio_value = st.audio_input("무엇을 도와드릴까요?")
        voice_on = st.toggle("🔊 답변을 음성으로 듣기", value=True)
        run_voice = st.button("실행", type="primary", use_container_width=True)

        # 최근 처리 결과: 인식된 질문(STT 정확성 확인), 이메일 발송 결과
        notice = st.session_state.notice
        if notice:
            if notice["source"] == "voice":
                st.markdown("#### 🎙️ 인식된 질문")
                st.info(notice["question"])
            if notice["email"]:
                if notice["email"]["status"] == "success":
                    st.success(f"📧 {notice['email']['to']}(으)로 이메일을 보냈습니다.")
                else:
                    st.error(f"이메일 발송 실패: {notice['email'].get('error')}")

    # ── 오른쪽: 에이블과 실시간 대화 (음성 질문도 함께 표시) ──
    with col_right:
        st.subheader(f"💬 '{st.session_state.active_doc}' 실시간 대화")
        current_history = st.session_state.messages_dict.setdefault(st.session_state.active_doc, [])
        with st.container(height=480, border=True):
            if not current_history:
                st.caption(f"'{st.session_state.active_doc}'에 대한 첫 대화를 시작해 보세요.")
            for message in current_history:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # 최근 답변 음성 (새 답변일 때만 자동 재생)
        if st.session_state.last_audio:
            st.audio(st.session_state.last_audio, format="audio/mp3",
                     autoplay=st.session_state.play_audio)
            st.session_state.play_audio = False

        with st.form("chat_form", clear_on_submit=True):
            user_input = st.text_area("에이블에게 메시지 보내기",
                                      placeholder="어떤 업무를 도와드릴까요?")
            submitted = st.form_submit_button("💬 에이블에게 보내기",
                                              type="primary", use_container_width=True)

        if st.button("🗑️ 대화 기록 지우기", use_container_width=True):
            st.session_state.messages_dict[st.session_state.active_doc] = []
            st.session_state.notice = None
            st.session_state.last_audio = None
            st.rerun()

    # ── 질문 처리: 음성(실행 버튼) 또는 텍스트(보내기 버튼) ──
    question, source = None, None
    if run_voice:
        if audio_value is None:
            col_left.warning("먼저 마이크 버튼을 눌러 질문을 녹음해 주세요.")
        else:
            try:
                with col_left, st.spinner("음성을 인식하고 있습니다.."):
                    # STT: 녹음은 파일이 아니라 메모리 데이터 → (파일명, 바이트)로 전달
                    transcript = client.audio.transcriptions.create(
                        model="whisper-1",
                        file=("recording.wav", audio_value.getvalue())
                    )
                question, source = transcript.text, "voice"
            except Exception as e:
                col_left.error(f"음성 인식 오류: {e}")
    elif submitted and user_input.strip():
        question, source = user_input.strip(), "text"

    if question:
        try:
            with col_right, st.spinner("답변을 준비하고 있습니다.."):
                answer, email = ask_able(question, current_meeting)

                # 대화 히스토리 저장 (질문·답변 한 쌍씩, 최근 MAX_HISTORY개만 유지)
                current_history.append({"role": "user", "content": question})
                current_history.append({"role": "assistant", "content": answer})
                st.session_state.messages_dict[st.session_state.active_doc] = current_history[-MAX_HISTORY:]
                st.session_state.notice = {"source": source, "question": question, "email": email}

                # LLM 최종 응답을 음성으로 즉시 재생 (미션 5·6 TTS)
                if voice_on:
                    speech = client.audio.speech.create(
                        model="tts-1",
                        voice="nova",
                        input=to_speech_text(answer),
                        instructions="비서가 담당자에게 얘기하는 정중한 말투로 빠뜨리지 말고 또박또박 읽어주세요."
                    )
                    st.session_state.last_audio = speech.content
                    st.session_state.play_audio = True
            st.rerun()
        except Exception as e:
            col_right.error(f"오류: {e}")


# ══════════════════════════════════════════════════
# 메뉴 2: 업무 우선순위 & 목 분석 (아이젠하워 매트릭스)
# ══════════════════════════════════════════════════
elif page == PAGES[1]:
    st.subheader("📊 업무 우선순위 & 목표 분석")
    st.caption("선택된 회의록에서 김 대리 업무를 긴급도 × 중요도로 분류하고, 신제품 마케팅 지출 내역 및 주요 목표 진행 상황 대시보드를 시각화합니다.")

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

    priority_template = """
다음 회의록을 분석하세요.
1. 김 대리에게 배정된 업무를 모두 추출하세요 (기한, 긴급도, 중요도, 상세 근거).
2. 참석자별 배정 업무 수를 세세요.

[회의록]
{meeting}
"""
    def classify(urgent, important):
        if urgent == "높음" and important == "높음": return "즉시 착수"
        elif important == "높음": return "순차 진행"
        else: return "위임 검토"

    if st.button("우선순위 분석"): # 수정 : 12번 검증 셀 문자열과 일치하도록 type="primary"를 제거해도 되는지?
        if not current_meeting:
            st.warning("먼저 회의록 파일을 업로드해 주세요.")
        else:
            try:
                with st.spinner("업무를 분석하고 있습니다.."):
                    response = client.responses.create(
                        model="gpt-4o-mini",
                        input=[{"role": "user",
                                "content": priority_template.format(meeting=current_meeting)}],
                        text={"format": priority_schema},   # 정해진 JSON 형식으로만 답변
                        temperature=0
                    )
                    data = json.loads(response.output_text)   # JSON 문자열 → 딕셔너리
                    st.session_state[f"priority_data_{st.session_state.active_doc}"] = data # 주석 추가가ㅏㅏ
            except Exception as e:
                st.error(f"분석 오류: {e}")

    # ── ① 업무 집중도 (R&R 밸런스) ──────────
    pdata = st.session_state.get(f"priority_data_{st.session_state.active_doc}")
    if pdata:
        try:
            df_people = pd.DataFrame(pdata["참석자별업무"])
            kim = df_people[df_people["이름"].str.contains("김")]["업무수"].sum() if not df_people.empty else 0
            total = df_people["업무수"].sum() if not df_people.empty else 1

            st.markdown("#### ⚖️ 업무 집중도")
            col1, col2 = st.columns(2)
            col1.metric("김 대리 배정 업무", f"{kim}건")
            col2.metric("전체 배정 업무 중 비중",
                        f"{kim / total * 100:.0f}%" if total else "-") # if total else 삭제할까 말까...?

            # 김 대리 업무가 3건 이상이거나 절반 이상이면 경보
            if kim >= 3 or (total and kim / total >= 0.5):
                st.error("🚨 김 대리에게 업무가 집중되어 있습니다. 우선순위와 위임을 검토하세요.")
            else:
                st.success("업무 배분이 적정한 수준입니다.")

            st.bar_chart(df_people.set_index("이름")["업무수"])

            # ── ② 우선순위 매트릭스 ──────────────────
            df_task = pd.DataFrame(pdata["업무목록"])
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

            # ── ③ 상세 분석 표 ───────────────────────
            st.markdown("#### 📋 상세 분석 표")
            st.dataframe(df_task[["분류", "업무명", "기한", "긴급도", "중요도", "근거"]],
                         column_config={
                             "근거": st.column_config.TextColumn("분석 근거 및 배경", width="large"),
                             "업무명": st.column_config.TextColumn("배정 업무명", width="medium")
                         },
                         use_container_width=True,
                         hide_index=True
            )
            st.caption("※ 중요도는 회의 내용을 바탕으로 한 에이블의 판단입니다.")

        except Exception as e:
            st.error(f"오류가 발생했습니다: {e}")


    # ── ④ 시각화 대시보드: 지출 내역 & 목표 진행 상황] ──
    st.divider()
    st.markdown("### 📈 마케팅 지출 내역 및 목표 진행 상황 대시보드")

    col_dash1, col_dash2 = st.columns([1, 1], gap="large")

    with col_dash1:
        st.markdown("#### 💰 마케팅 예산 지출 내역 (총 5,000만 원)")
        st.caption("회의록에 명시된 5천만 원 예산 배분 구성")

        # 지출 데이터프레임
        df_expense = pd.DataFrame({
            "항목": ["SNS 광고 (인스타/유튜브)", "인플루언서 협업", "숏폼/카드뉴스 제작", "출시 프로모션 이벤트", "예비비"],
            "금액(백만원)": [25.0, 10.0, 8.0, 5.0, 2.0]
        })

        # Streamlit 내장 도넛/파이 대용 시각화 및 지표
        c_tot1, c_tot2 = st.columns([1, 2])
        c_tot1.metric("총 집행 예산", "₩50M", delta="전체 100%")
        c_tot2.dataframe(df_expense, hide_index=True, use_container_width=True)
        st.bar_chart(df_expense.set_index("항목")["금액(백만원)"])

    with col_dash2:
        st.markdown("#### 🎯 김 대리 목표 진행 상황 (목표 달성률 %)")
        st.caption("배정된 3대 핵심 과업 및 출시 일정 기준 진행 현황")

        df_goals = pd.DataFrame({
            "핵심 목표 및 마일스톤": [
                "경쟁사 A사 온라인 마케팅 사례 분석",
                "신제품 기능 강조 숏폼/후기 초안 작성",
                "중급 인플루언서 3인 후보군 섭외 조사",
                "출시 2주 전 티저 광고 개시 준비",
                "출시 기념 할인 & SNS 공유 프로모션"
            ],
            "진행률(%)": [85, 60, 45, 30, 20]
        })

        # 수평형 바 차트 형태로 진행률 시각화
        st.dataframe(
            df_goals,
            column_config={
                "진행률(%)": st.column_config.ProgressColumn(
                    "진행 상황 (%)",
                    format="%d%%",
                    min_value=0,
                    max_value=100
                ),
                "핵심 목표 및 마일스톤": st.column_config.TextColumn("추진 과제", width="large")
            },
            hide_index=True,
            use_container_width=True
        )

# ── 사이드바 아래쪽: 회의록 상태 · 저장된 대화 ──
with st.sidebar:
    status = "✅" if st.session_state.meeting_text else "⬜"
    st.subheader(f"📋 히스토리 관리 {status}")   # ✅: 회의록 준비됨 / ⬜: 회의록 없음
    # 저장된 대화 수 (st.metric보다 한 단계 작은 글자)
    st.markdown(
        f"<p style='font-size:0.875rem; margin:0;'>저장된 대화</p>"
        f"<p style='font-size:1.75rem; margin:0;'>{len(st.session_state.messages) // 2}건</p>",
        unsafe_allow_html=True
    )

# ══════════════════════════════════════════════════════════════
# 메뉴 3: 마감 캘린더 (앱 안 달력 + .ics 다운로드)
# 회의록 / 문서별 일정이 누적 병합되고, 다른 회의록을 추가해도 일정 중복을 방지합니다.
#  - 이 셀은 app.py 끝에 "이어 붙이기"(-a) 합니다.
#  - ⚠️ app.py 셀(%%writefile app.py)을 실행한 직후, 이 셀을 한 번만 실행하세요.
# ══════════════════════════════════════════════════════════════
from datetime import date, datetime, timedelta
from zoneinfo import ZoneInfo
import calendar as pycal

# 업무 고유 ID 생성 및 유사 업무 비교
import uuid
import hashlib
from difflib import SequenceMatcher

# 오늘 날짜는 한국 시간 기준 (Streamlit Cloud 서버는 UTC라 date.today()를 쓰면 오전 9시 전엔 하루 전 날짜가 됨)
def kst_today():
    return datetime.now(ZoneInfo("Asia/Seoul")).date()

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
회의록에서 김대리업무와 주요일정을 추출하세요.

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

# ============================================================
# 업무 고유 ID 및 변경 이력 관리
# ============================================================
# 기능:
# 1. 업무마다 고유 ID 부여
# 2. 기존 업무와 신규 업무의 유사도 비교
# 3. 업무가 수정되면 기존 ID 유지
# 4. 변경 전후 내용을 이력으로 보관
#
# 주의:
# 유사도가 높다고 자동으로 같은 업무라고 확정하지 않음
# 최종 판단은 사용자가 수행
# ============================================================

def create_task_id():
    """업무별 고유 ID 생성"""
    return str(uuid.uuid4())


def compare_task_names(old_name, new_name):
    """두 업무명의 문자열 유사도 계산"""

    old_name = re.sub(r"\s+", "", old_name)
    new_name = re.sub(r"\s+", "", new_name)

    return SequenceMatcher(
        None,
        old_name,
        new_name
    ).ratio()


def update_task_history(task, new_due, meeting_date, source):
    """기존 업무의 마감일을 수정하고 변경 이력을 저장"""

    # 기존 일정과 변경된 일정 비교
    old_due = task["due"]

    # 변경 이력 저장 공간이 없다면 생성
    task.setdefault("history", [])

    # 업무 변경 사항 기록
    task["history"].append({
        "meeting_date": meeting_date,
        "old_due": old_due,
        "new_due": new_due,
        "source": source,
        "updated_at": kst_now_str()
    })

    # 기존 업무의 마감일 변경
    task["due"] = new_due


def find_similar_tasks(new_name, schedule, threshold=0.65):
    """기존 일정 중 유사한 업무 후보를 검색"""

    candidates = []

    for task in schedule:

        similarity = compare_task_names(
            task["name"],
            new_name
        )

        if similarity >= threshold:

            candidates.append({
                "task": task,
                "similarity": similarity
            })

    # 유사도가 높은 순서로 정렬
    return sorted(
        candidates,
        key=lambda x: x["similarity"],
        reverse=True
    )


def get_source_hash(content, meeting_date):
    """동일 회의록을 반복 분석했는지 확인하는 식별값"""

    raw = f"{meeting_date.isoformat()}\n{content}"

    return hashlib.sha256(
        raw.encode("utf-8")
    ).hexdigest()

# ============================================================
# 회의록 날짜 자동 인식
# ============================================================
# 지원 형식:
# 2026-11-04
# 2026.11.04
# 2026/11/04
# 2026년 11월 4일
#
# 주의:
# 회의 날짜 또는 회의 일시로 표시된 항목만 우선 탐색
# 날짜를 확인하지 못하면 None 반환
# ============================================================

def extract_meeting_date(text):

    # 회의 날짜가 표시된 행을 우선 검색
    patterns = [
        r"회의\s*일시\s*[:：]\s*(\d{4})[-./년]\s*(\d{1,2})[-./월]\s*(\d{1,2})",
        r"회의\s*날짜\s*[:：]\s*(\d{4})[-./년]\s*(\d{1,2})[-./월]\s*(\d{1,2})",
        r"회의일\s*[:：]\s*(\d{4})[-./년]\s*(\d{1,2})[-./월]\s*(\d{1,2})",
    ]

    for pattern in patterns:

        match = re.search(pattern, text)

        if match:

            try:

                year, month, day = map(
                    int,
                    match.groups()
                )

                return date(year, month, day)

            except ValueError:

                continue

    # 회의일을 확인하지 못했다면 임의로 지정하지 않음
    return None

def create_schedule_task(name, kind, expr, due, source, meeting_date):
    """기존 일정 데이터 구조를 유지하며 신규 업무 생성"""

    return {
        "task_id": create_task_id(),
        "name": name,
        "kind": kind,
        "expr": expr,
        "due": due,
        "use": due is not None, # 기한이 없는 업무는 임의 날짜로 달력에 표시하지 않음
        "done": False,
        "source": source,
        "meeting_date": meeting_date,
        "history": []
    }


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

# ── ④ 업무 상태 · 남은 기간 표시 (달력 색상과 정렬용) # 동글뱅이 4~5번. 코드 수정 안해도되는지 검토ㄱ) ──
def task_status(due, done, today):
    """(아이콘, 상태, 정렬 순서) 반환: 완료 → 기한 경과 → 마감 임박 → 예정"""
    if done:
        return "✅", "완료", 3
    if due < today:
        return "🔴", "기한 경과", 0
    if (due - today).days <= 3:
        return "🟠", "마감 임박", 1
    return "🔵", "예정", 2

# ============================================================
# 회의록 날짜 자동 인식
# ============================================================
# 지원 형식:
# 2026-11-04
# 2026.11.04
# 2026/11/04
# 2026년 11월 4일
#
# 주의:
# 회의 날짜 또는 회의 일시로 표시된 항목만 우선 탐색
# 날짜를 확인하지 못하면 None 반환
# ============================================================

def extract_meeting_date(text):

    # 회의 날짜가 표시된 행을 우선 검색
    patterns = [
        r"회의\s*일시\s*[:：]\s*(\d{4})[-./년]\s*(\d{1,2})[-./월]\s*(\d{1,2})",
        r"회의\s*날짜\s*[:：]\s*(\d{4})[-./년]\s*(\d{1,2})[-./월]\s*(\d{1,2})",
        r"회의일\s*[:：]\s*(\d{4})[-./년]\s*(\d{1,2})[-./월]\s*(\d{1,2})",
    ]

    for pattern in patterns:

        match = re.search(pattern, text)

        if match:

            try:

                year, month, day = map(
                    int,
                    match.groups()
                )

                return date(year, month, day)

            except ValueError:

                continue

    # 회의일을 확인하지 못했다면 임의로 지정하지 않음
    return None

def deadline_label(due, today):
    """D-3, 오늘 마감, 2일 경과처럼 남은 기간을 글로 표시"""
    days_left = (due - today).days
    if days_left < 0:
        return f"{-days_left}일 경과"
    return "오늘 마감" if days_left == 0 else f"D-{days_left}"

def sort_tasks(items, today):
    """미완료·급한 업무가 먼저 오도록 정렬"""
    return sorted(items, key=lambda it: (task_status(it["due"], it["done"], today)[2], it["name"]))

# ── 달력 버튼 동작 (on_click 콜백: 화면을 다시 그리기 전에 실행되어 한 번에 반영) ──
def move_month(offset):
    view = st.session_state.cal_view
    year, month0 = divmod(view.year * 12 + view.month - 1 + offset, 12)   # 연도가 바뀌어도 정확
    st.session_state.cal_view = date(year, month0 + 1, 1)

def go_today():
    st.session_state.cal_view = kst_today().replace(day=1)
    st.session_state.cal_selected = kst_today()

def select_day(day):
    st.session_state.cal_selected = day

# ── ⑤ 세션 상태 초기화 (메뉴를 옮겨도 일정·달력 위치 유지) ── (각각 주석 달기?!!!!)
if "messages_dict" not in st.session_state:
    st.session_state.messages_dict = {}       # {문서명: [메시지들]} 형태로 문서별 대화 분리
if "uploaded_docs" not in st.session_state:
    st.session_state.uploaded_docs = {}       # {문서명: 본문내용}
if "archived_meetings" not in st.session_state:
    st.session_state.archived_meetings = []   # [{"시간": ..., "제목": ..., "내용": ...}] 삭제되어도 계속 유지
if "active_doc" not in st.session_state:
    st.session_state.active_doc = ""
if "notice" not in st.session_state:
    st.session_state.notice = None
if "last_audio" not in st.session_state:
    st.session_state.last_audio = None
    st.session_state.play_audio = False
if "schedule" not in st.session_state:
    st.session_state.schedule = []
if "sched_ver" not in st.session_state:
    st.session_state.sched_ver = 0
if "cal_base" not in st.session_state:
    st.session_state.cal_base = kst_today()
if "cal_view" not in st.session_state:
    st.session_state.cal_view = kst_today().replace(day=1)
if "cal_selected" not in st.session_state:
    st.session_state.cal_selected = kst_today()
if "followups" not in st.session_state:
    st.session_state.followups = []
if "fu_ver" not in st.session_state:
    st.session_state.fu_ver = 0
if "briefing" not in st.session_state:
    st.session_state.briefing = None

# ── ⑥ 화면 ───────────────────────────────────────
if page == PAGES[2]:
    st.subheader("📅 마감 캘린더")
    st.caption("회의록을 바꿔가며 일정을 추출해도, 기존 일정을 날리지 않고 중복 없이 누적하여 캘린더에 병합합니다.")

    # ① 일정 추출
    # 회의록 날짜 자동 인식 + 수동 수정 기능

    detected_date = extract_meeting_date(current_meeting)

    # 현재 문서별로 날짜 입력 상태를 구분
    document_key = hashlib.sha256(
        current_meeting.encode("utf-8")
    ).hexdigest()[:12]

    date_key = f"meeting_date_{document_key}"

    # 새 문서를 처음 선택한 경우에만 날짜 초기화
    if date_key not in st.session_state:

        if detected_date:
            st.session_state[date_key] = detected_date
        else:
            st.session_state[date_key] = kst_today()

    # 회의 날짜 자동 인식 여부 표시
    if detected_date:

        st.success(
            f"✅ 회의 날짜 자동 인식: {detected_date}"
        )

    else:

        st.warning(
            "⚠️ 회의록에서 회의 날짜를 확인하지 못했습니다. "
            "아래에서 회의 날짜를 직접 선택해 주세요."
        )

    c1, c2 = st.columns(
        [1, 1],
        vertical_alignment="bottom"
    )

    # 자동 인식한 날짜도 사용자가 수정 가능
    base_day = c1.date_input(
        "회의 날짜 (기한 계산 기준일)",
        key=date_key
    )

    # 기존 코드와 호환되도록 상태 유지
    st.session_state.cal_base = base_day

    if c2.button("🔍 회의록에서 일정 추출", type="primary", use_container_width=True):
        if not current_meeting:
            st.warning("먼저 사이드바에서 회의록 파일을 업로드해 주세요.")
        else:
            try:
                with st.spinner("신규 일정 추출 및 기존 캘린더와 병합 중.."):
                    response = client.responses.create(
                        model="gpt-4o-mini",
                        input=[{"role": "user",
                                "content": schedule_template.format(meeting=current_meeting)}],
                        text={"format": schedule_schema},
                        temperature=0
                    )
                    data = json.loads(response.output_text)

                # ====================================================
                # 기존 일정과 신규 추출 업무 비교
                # ====================================================
                # 자동으로 병합하지 않고 검토 대기 목록 생성
                # 동일 회의록을 반복 분석하면 중복 생성 방지
                # ====================================================

                source_hash = get_source_hash(
                    current_meeting,
                    base_day
                )

                if "processed_meetings" not in st.session_state:
                    st.session_state.processed_meetings = set()

                # 동일한 회의록의 중복 처리 방지
                if source_hash in st.session_state.processed_meetings:

                    st.info(
                        "이미 분석하여 등록한 회의록입니다. "
                        "기존 업무와 변경 이력을 확인해 주세요."
                    )

                else:

                    pending = []

                    for kind, items in [
                        ("김 대리 업무", data["김대리업무"]),
                        ("주요 일정", data["주요일정"])
                    ]:

                        for item in items:

                            # 구분 이름만 있는 빈 항목은 건너뛰기
                            if item["일정명"] in (
                                "김 대리 업무",
                                "주요 일정"
                            ):
                                continue

                            name = item["일정명"]

                            # 기존 기한 처리 규칙 유지
                            expr = (
                                item["기한표현"]
                                if item["기한표현"] not in (
                                    None, "", "null"
                                )
                                else None
                            )

                            days = item["기한일수"]

                            if expr and re.search(
                                r"\d+\s*(주|일)\s*전",
                                expr
                            ):
                                days = None

                            due = calc_due(
                                base_day,
                                expr,
                                days
                            )

                            # 기존 업무와 유사한 후보 검색
                            candidates = find_similar_tasks(
                                name,
                                st.session_state.schedule
                            )

                            # 신규 추출 결과를 검토 대기 목록에 저장
                            pending.append({
                                "name": name,
                                "kind": kind,
                                "expr": expr,
                                "due": due,
                                "source": st.session_state.active_doc,
                                "meeting_date": base_day,
                                "candidates": [
                                    c["task"]["task_id"]
                                    for c in candidates
                                ]
                            })

                    st.session_state.pending_schedule = pending
                    st.session_state.pending_source_hash = source_hash

                    st.success(
                        f"✅ {len(pending)}건의 업무를 추출했습니다. "
                        "아래에서 신규 등록 또는 기존 업무 수정 여부를 확인해 주세요."
                    )
                    # 수정 : 아래와 # 신규 업무~사용자 확인 전까지 범위 삭제 해야 하는건지...?
            except Exception as e:
                st.error(f"오류: {e}")

    # ============================================================
    # 신규 업무 / 기존 업무 수정 여부 사용자 확인
    # ============================================================
    # 유사한 업무가 있으면 기존 업무 수정을 우선 제안
    # 사용자가 최종 확인해야만 일정에 반영
    # ============================================================

    pending = st.session_state.get(
        "pending_schedule",
        []
    )

    if pending:

        st.markdown("### 🔎 신규 업무 및 변경사항 검토")

        for idx, item in enumerate(pending):

            with st.container(border=True):

                st.markdown(
                    f"**📌 {item['name']}**"
                )

                st.caption(
                    f"회의일: {item['meeting_date']} | "
                    f"마감일: {item['due'] or '미정'}"
                )

                # 기존 일정 중 유사 업무 후보 확인
                candidates = [
                    task
                    for task in st.session_state.schedule
                    if task["task_id"] in item["candidates"]
                ]

                # 유사 업무가 없으면 신규 업무로 등록
                if not candidates:

                    st.info(
                        "기존 일정에서 유사한 업무를 찾지 못했습니다."
                    )

                    action = "신규 업무로 등록"

                    selected_task = None

                else:

                    st.warning(
                        "⚠️ 기존 업무와 유사한 일정이 발견되었습니다."
                    )

                    # 사용자에게 신규/수정 여부 선택 요청
                    options = ["신규 업무로 등록"]

                    options += [
                        (
                            f"기존 업무 수정: {task['name']} "
                            f"(마감 {task['due']})"
                        )
                        for task in candidates
                    ]

                    selected_option = st.radio(
                        "이 업무를 어떻게 처리하시겠습니까?",
                        options,
                        key=f"task_action_{idx}"
                    )

                    action = selected_option

                    selected_task = None

                    # 기존 업무 수정 선택 시 대상 업무 확인
                    if selected_option != "신규 업무로 등록":

                        selected_index = options.index(
                            selected_option
                        ) - 1

                        selected_task = candidates[selected_index]

                # 사용자가 확인한 경우에만 반영
                if st.button(
                    "✅ 확인 및 적용",
                    key=f"confirm_task_{idx}"
                ):

                    if selected_task is None:

                        new_task = create_schedule_task(
                            item["name"],
                            item["kind"],
                            item["expr"],
                            item["due"],
                            item["source"],
                            item["meeting_date"]
                        )

                        st.session_state.schedule.append(
                            new_task
                        )

                        st.success("신규 업무가 등록되었습니다.")

                    else:

                        update_task_history(
                            selected_task,
                            item["due"],
                            item["meeting_date"],
                            item["source"]
                        )

                        st.success(
                            "기존 업무가 수정되고 변경 이력이 저장되었습니다."
                        )

                    # 처리한 항목은 검토 대기 목록에서 제거
                    st.session_state.pending_schedule.pop(idx)

                    # 모든 항목 검토가 완료된 경우
                    if not st.session_state.pending_schedule:

                        st.session_state.processed_meetings.add(
                            st.session_state.pending_source_hash
                        )

                    st.session_state.sched_ver += 1

                    st.rerun()

                    # 수정 : 들여쓰기가 잘 되어있는지 검토
    schedule = st.session_state.schedule
    ver = st.session_state.sched_ver


    if not schedule:
        st.info("회의록을 업로드한 뒤 '회의록에서 일정 추출'을 눌러 주세요.")
    else:
        # ② 일정 확인 및 수정 (고친 값은 일정 목록에 바로 저장 → 메뉴를 옮겨도 유지)
        with st.expander("✏️ 일정 확인 및 수정 (날짜 정하기 · 완료 표시)"):
            st.caption("기한이 없거나 애매한 일정은 날짜를 정한 뒤 '달력 표시'를 체크하세요.")
            for i, it in enumerate(schedule):
                a, b, c, d = st.columns([1, 3, 2, 1], vertical_alignment="center")
                it["use"] = a.checkbox("달력 표시", value=it["use"], key=f"use_{ver}_{i}")
                b.markdown(f"**{it['name']}** ({it['kind']})  \n<small>출처: {it.get('source', '알 수 없음')}</small>", unsafe_allow_html=True)
                # 기한 미정 업무는 회의일 또는 오늘을 입력창의 초기값으로 제안
                # 제안값은 확정 마감일이 아니며 사용자가 직접 확인해야 함
                initial_due = it["due"] or it.get("meeting_date") or kst_today()

                it["due"] = c.date_input(
                    "마감일",
                    value=initial_due,
                    key=f"due_{ver}_{i}"
                )
                it["done"] = d.checkbox("완료", value=it["done"], key=f"done_{ver}_{i}")

        today = kst_today()
        active = [it for it in schedule if it["use"]]

        # 날짜별 마감 업무 / 마일스톤 모으기
        due_map, ms_map, rows = {}, {}, []
        for it in active:
            due_map.setdefault(it["due"], []).append(it)
            for r in make_milestones(it["name"], it["due"], base_day):
                rows.append(r)
                if r["단계"] != "D-day":
                    ms_map.setdefault(r["날짜"], []).append(r)

        # ③ 달력 위쪽: 달 이동
        st.divider()
        view = st.session_state.cal_view
        n1, n2, n3, n4 = st.columns([1, 3, 1, 1], vertical_alignment="center")
        if n1.button("◀ 이전 달", use_container_width=True):
            y, m = (view.year, view.month - 1) if view.month > 1 else (view.year - 1, 12)
            st.session_state.cal_view = date(y, m, 1)
            st.rerun()
        n2.markdown(f"### {view.year}년 {view.month}월")
        if n3.button("다음 달 ▶", use_container_width=True):
            y, m = (view.year, view.month + 1) if view.month < 12 else (view.year + 1, 1)
            st.session_state.cal_view = date(y, m, 1)
            st.rerun()
        if n4.button("오늘", use_container_width=True):
            st.session_state.cal_view = today.replace(day=1)
            st.session_state.cal_selected = today
            st.rerun()
        st.caption("🔴 기한 경과 · 🟠 마감 임박(3일 이내) · 🔵 예정 · ✅ 완료 · ⚪ 기한 미정 · ⏰ 마일스톤(D-14, D-7, D-1)")

        # ④ 달력: 요일 머리글 + 날짜 칸 (월요일 시작)
        head = st.columns(7)
        for col, w in zip(head, ["월", "화", "수", "목", "금", "토", "일"]):
            col.markdown(f"**{w}**")

        for week in pycal.Calendar(firstweekday=0).monthdayscalendar(view.year, view.month):
            cols = st.columns(7)
            for col, d in zip(cols, week):
                if d == 0:           # 이번 달이 아닌 빈 칸
                    continue
                day = date(view.year, view.month, d)
                with col.container(border=True):
                    label = f"{d}일 · 오늘" if day == today else f"{d}일"
                    selected = day == st.session_state.cal_selected
                    if st.button(label, key=f"cal_{day}", use_container_width=True, type="primary" if selected else "secondary"):
                        st.session_state.cal_selected = day
                        st.rerun()
                    for it in due_map.get(day, [])[:2]:
                        s_icon = "✅" if it["done"] else "🔴"
                        st.caption(f"{s_icon} {it['name'][:9]}")

                    # 칸이 너무 길어지지 않게 마감 업무는 2건까지, 제목은 14자까지만 #(수정 : 삭제할지 말지ㅠ)
                    items = sort_tasks(due_map.get(day, []), today)
                    if items:
                        st.caption(f"마감 업무 {len(items)}건")
                        for it in items[:2]:
                            icon = task_status(it["due"], it["done"], today)[0]
                            name = it["name"] if len(it["name"]) <= 14 else it["name"][:14] + "…"
                            st.caption(f"{icon} {name}")
                        if len(items) > 2:
                            st.caption(f"외 {len(items) - 2}건")
                    ms = ms_map.get(day, [])
                    if ms:
                        st.caption(f"⏰ 알림 {len(ms)}건 ({', '.join(r['단계'] for r in ms)})")

        # ⑤ 선택한 날짜의 마감 업무
        sel = st.session_state.cal_selected
        st.markdown(f"#### 📌 {sel.month}월 {sel.day}일 마감 업무")
        items = sort_tasks(due_map.get(sel, []), today) ### items = due_map.get(sel, []) (이걸로 수정 안해도 될지)
        if not items:
            st.info("선택한 날짜에 마감하는 업무가 없습니다.")
        for it in items:
            icon, status, _ = task_status(it["due"], it["done"], today)
            with st.container(border=True):
                st.markdown(f"**{it['name']}**")
                st.caption(f"{icon} {status} · {it['kind']} · {deadline_label(it['due'], today)}")
                st.caption(f"기한 근거(회의 표현): {it['expr'] or '회의록에 기한 표현 없음 · 직접 지정'}")
        for r in ms_map.get(sel, []):
            st.caption(f"⏰ {r['단계']} 알림: {r['일정명']} (마감 {r['날짜'] + timedelta(days=int(r['단계'][2:]))})")
        if items:
            st.caption("완료 여부는 위의 '일정 확인 및 수정'에서 변경할 수 있습니다.")

        # ⑥ 기한 미정 업무
        undecided = [it for it in schedule if not it["use"]]
        with st.expander(f"⚪ 기한 미정 업무 {len(undecided)}건"):
            if not undecided:
                st.write("기한 미정 업무가 없습니다.")
            for it in undecided:
                st.markdown(f"**{'✅ 완료' if it['done'] else '⬜ 미완료'} · {it['name']}**")
                st.caption(f"{it['kind']} · 기한 근거: {it['expr'] or '회의록에 기록 없음'} · "
                           "'일정 확인 및 수정'에서 날짜를 정하고 '달력 표시'를 체크하면 달력에 나타납니다.")

        # ⑦ 캘린더 파일 다운로드 (Google·Outlook·휴대폰 캘린더에서 열면 등록됨)
        st.divider()
        if rows:
            rows.sort(key=lambda r: r["날짜"])
            st.download_button("📥 캘린더 파일(.ics) 다운로드", data=make_ics(rows),
                               file_name="able_schedule.ics", mime="text/calendar",
                               type="primary", use_container_width=True)
            st.caption(f"마일스톤 {len(rows)}개(D-14, D-7, D-1, D-day)가 담깁니다. "
                       "파일을 열면 Google·Outlook·휴대폰 캘린더에 바로 등록됩니다.")
    # ============================================================
    # 외부 캘린더 연동 (마감 캘린더 메뉴 전용)
    # ============================================================
    # 기능:
    # 1. 등록된 업무 중 캘린더에 추가할 업무 선택
    # 2. Google Calendar 일정 등록 화면 열기
    # 3. Outlook Calendar 일정 등록 화면 열기
    # 4. 업무명 및 마감일 자동 입력
    # 5. 사용자 확인 후 외부 캘린더에 저장
    #
    # 주의:
    # 기존 업무 데이터와 .ics 다운로드 기능은 변경하지 않음
    # ============================================================

    from urllib.parse import urlencode

    st.divider()
    st.subheader("🔗 외부 캘린더 연동")

    st.caption(
        "업무를 선택하면 해당 업무명과 마감일이 입력된 "
        "Google·Outlook 캘린더 화면으로 이동합니다."
    )

    # 1. 마감일이 확정된 업무만 선택 목록에 표시
    calendar_tasks = [
        task
        for task in st.session_state.schedule
        if task.get("due") is not None
        and task.get("use", False)
    ]

    if calendar_tasks:

        # 2. 캘린더에 등록할 업무 선택
        selected_index = st.selectbox(
            "📌 캘린더에 등록할 업무",
            range(len(calendar_tasks)),
            format_func=lambda i: (
                f"{calendar_tasks[i]['name']} "
                f"(마감: {calendar_tasks[i]['due']})"
            ),
            key="external_calendar_task"
        )

        selected_task = calendar_tasks[selected_index]

        # 3. 선택한 업무 정보 가져오기
        task_name = selected_task["name"]
        task_due = selected_task["due"]

        # 날짜를 YYYY-MM-DD 형식으로 변환
        if isinstance(task_due, str):
            task_due = date.fromisoformat(task_due[:10])

        # 외부 캘린더에는 종일 일정으로 등록
        start_date = task_due
        end_date = task_due + timedelta(days=1)

        # 4. 일정 상세 설명
        task_description = (
            f"업무명: {task_name}\n"
            f"업무 구분: {selected_task.get('kind', '미정')}\n"
            f"회의록: {selected_task.get('source', '미정')}\n"
            f"회의일: {selected_task.get('meeting_date', '미정')}"
        )

        # ====================================================
        # Google Calendar 등록 링크
        # ====================================================
        google_params = {
            "action": "TEMPLATE",
            "text": task_name,
            "dates": (
                f"{start_date:%Y%m%d}/"
                f"{end_date:%Y%m%d}"
            ),
            "details": task_description
        }

        google_url = (
            "https://calendar.google.com/calendar/render?"
            + urlencode(google_params)
        )

        # ====================================================
        # Outlook Calendar 등록 링크
        # ====================================================
        outlook_params = {
            "path": "/calendar/action/compose",
            "rru": "addevent",
            "subject": task_name,
            "startdt": start_date.isoformat(),
            "enddt": end_date.isoformat(),
            "allday": "true",
            "body": task_description
        }

        outlook_url = (
            "https://outlook.live.com/calendar/0/"
            "deeplink/compose?"
            + urlencode(outlook_params)
        )

        # 5. 외부 캘린더 연결 버튼
        col_google, col_outlook = st.columns(2)

        with col_google:
            st.link_button(
                "🔵 Google Calendar에 등록",
                google_url,
                use_container_width=True
            )

        with col_outlook:
            st.link_button(
                "🟣 Outlook Calendar에 등록",
                outlook_url,
                use_container_width=True
            )

        st.info(
            "📅 외부 캘린더에서 일정 내용을 확인하거나 "
            "수정한 후 저장해 주세요."
        )

    else:
        st.info(
            "마감일이 확정된 업무가 없습니다. "
            "업무 일정을 먼저 등록해 주세요."
        )

# ══════════════════════════════════════════════════════════════
# 메뉴 4: 팔로업 메일 초안 자동 생성 & 원클릭 승인
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
- 김 대리 업무 수행에 실제로 필요한 자료와 정보를 담당자별로 모두 추출하세요.
- 동일한 담당자에게 여러 자료를 요청해야 한다면 하나의 메일로 통합하세요.
- 서로 다른 담당자에게 필요한 자료가 있다면 담당자별로 각각 메일 초안을 작성하세요.
- 근거에는 회의록에서 그 참석자가 한 발언을 그대로 적으세요.
- 회의록에 근거가 없는 요청은 만들지 마세요.
- 선행 자료 요청이 필요하지 않은 참석자에게는 메일을 생성하지 마세요.

[메일 작성 규칙]
- 김 대리가 해당 참석자에게 보내는 메일입니다. 받는 사람 직함에 '님'을 붙여 정중하게 쓰세요.
- 무엇이, 왜(어떤 업무에 필요한지), 언제까지 필요한지를 포함하세요. 기한이 회의록에 없으면 '가능한 일정'을 여쭙는 표현으로 쓰세요.
- 마크다운 기호 없이 일반 텍스트로, 5~8문장 이내로 쓰세요.
- 마지막은 '김 대리 드림'으로 끝내세요.

[회의록]
{meeting}
"""

# ── ② 화면 ──
if "followups" not in st.session_state:
    st.session_state.followups = []      # 생성한 초안 목록 (메뉴를 옮겨도 유지)
    st.session_state.fu_ver = 0          # 새로 만들 때마다 +1 (입력칸 초기화용)

if page == PAGES[3]:
    st.subheader("📨 팔로업 메일 초안")
    st.caption("회의에서 동료에게 받아야 할 자료를 찾아 요청 메일 초안을 만들고, 승인하면 발송합니다. 발송 완료 후에도 내용을 수정하여 같은 담당자나 다른 수신자에게 자유롭게 추가 발송할 수 있습니다.")

    if st.button("초안 만들기", key="fu_make", type="primary"):
        if not current_meeting:
            st.warning("먼저 사이드바에서 회의록 파일을 업로드해 주세요.")
        else:
            try:
                with st.spinner("필요한 자료를 찾고 메일 초안을 쓰고 있습니다.."):
                    response = client.responses.create(
                        model="gpt-4o-mini",
                        input=[{"role": "user",
                                "content": followup_template.format(meeting=current_meeting)}],
                        text={"format": followup_schema},
                        temperature=0
                    )
                    data = json.loads(response.output_text)

                # 초안마다 받는 사람(시연용 기본값: 김 대리 주소)과 발송 여부 추가
                st.session_state.followups = [
                    {**d, "to": MY_EMAIL, "sent": None} for d in data["요청목록"]
                ]
                st.session_state.fu_ver += 1
                if not data["요청목록"]:
                    st.info("회의록에서 요청할 선행 자료를 찾지 못했습니다.")
            except Exception as e:
                st.error(f"오류가 발생했습니다: {e}")

    # 초안마다: 근거 확인 → 수정 → 승인 버튼 (고친 내용은 초안 목록에 바로 저장)
    ver = st.session_state.fu_ver
    for i, d in enumerate(st.session_state.followups):
        st.divider()
        st.markdown(f"**📌 {d['담당자']}님께 '{d['요청자료']}' 요청 메일 초안을 작성했습니다. 발송할까요?**")
        st.caption(f"관련 업무: {d['관련업무']}  \n회의 근거: \"{d['근거']}\"")

        d["to"] = st.text_input("받는 사람", value=d["to"], key=f"fu_to_{ver}_{i}",
                                help="시연용으로 김 대리 주소가 들어 있습니다. 실제 동료 주소로 바꿔 보내세요.")
        d["메일제목"] = st.text_input("제목", value=d["메일제목"], key=f"fu_sub_{ver}_{i}")
        d["메일본문"] = st.text_area("본문 (수정 가능)", value=d["메일본문"], height=200,
                                  key=f"fu_body_{ver}_{i}")

                  # ── 이메일 발송 버튼 ──
        col_s1, col_s2 = st.columns([1, 4])

        with col_s1:
            send_now = st.button(
                "🚀 이메일 전송",
                key=f"fu_send_{ver}_{i}",
                type="primary"
            )

        # 이메일 전송 버튼을 누른 경우에만 실제 발송
        if send_now:

            # 수신자 이메일 주소 확인
            if "@" not in d["to"]:
                st.warning("받는 사람 이메일 주소를 확인해 주세요.")

            else:
                with st.spinner("메일을 보내고 있습니다.."):

                    # 기존 미션 1의 이메일 발송 함수 재사용
                    result = send_email(
                        d["to"],
                        d["메일제목"],
                        d["메일본문"]
                    )

                # 실제 SMTP 발송 결과 확인
                if result["status"] == "success":

                    t_str = kst_now_str()

                    # 발송 이력 저장
                    d.setdefault("logs", []).append(
                        f"[{t_str}] 수신처: {d['to']} (발송 성공)"
                    )

                    # 발송 완료 여부 저장
                    d["sent"] = d["to"]

                    st.success(
                        f"✅ {d['to']} 주소로 이메일 전송이 완료되었습니다!"
                    )

                    st.info(
                        "📩 메일 서버에 정상적으로 발송 요청이 접수되었습니다. "
                        "실제 수신 여부는 수신자의 받은편지함에서 확인해 주세요."
                    )

                else:
                    st.error(
                        f"❌ 이메일 발송 실패: {result.get('error')}"
                    )

        # 이전 발송 이력이 있는 경우 표시
        if d.get("sent"):
            st.caption(f"📧 최근 발송 주소: {d['sent']}")

#수정 : 이부분은 삭제할지 말지 &^^^^^
        if d.get("logs"):
            with st.expander("이 메일의 발송 이력"):
                for log in d["logs"]:
                    st.caption(log)

# ══════════════════════════════════════════════════════════════
# 메뉴 5: 1분 출근길 오디오 브리핑
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

# ── 화면 ──
if "briefing" not in st.session_state:
    st.session_state.briefing = None      # {"script": 원고, "audio": 음성} 저장 (메뉴를 옮겨도 유지)

if page == PAGES[4]:
    st.subheader(f"🎧 1분 출근길 브리핑 ({st.session_state.active_doc})")
    st.caption("회의 결론과 먼저 챙길 일만 1분 안팎의 음성으로 들려 드립니다.")

    voice = st.selectbox("목소리", ["nova", "alloy", "shimmer", "onyx"], key="br_voice")

    if st.button("브리핑 생성", type="primary"): # 수정 : key="br_make" 삭제해도 되겠지...?
        if not current_meeting:
            st.warning("먼저 사이드바에서 회의록 파일을 업로드해 주세요.")
        else:
            try:
                with st.spinner("브리핑 원고를 쓰고 있습니다.."):
# ============================================================
# 출근길 브리핑 원고 생성
# ============================================================
# 작성 기준:
# 1. 공영방송 뉴스 원고에 준하는 정확한 한국어 문장
# 2. 표준어 규정 및 한글 맞춤법 준수
# 3. 숫자·날짜·범위 표현의 정확성 검토
# 4. 음성으로 읽기 자연스러운 문장 구성
# 5. 회의록에 없는 정보 생성 금지
# ============================================================

                    response = client.responses.create(
                        model="gpt-4o-mini",
# 수정 : 아래 들여쓰기 검증해야 함.
    input=[
        {
            "role": "system",
            "content": """
당신은 김 대리의 AI 비서 에이블이자
한국어 뉴스 원고 작성에 숙련된 전문 방송작가입니다.

[문장 작성 기준]

1. KBS 뉴스 원고처럼 정확하고 간결한 문체로 작성합니다.
2. 한글 맞춤법, 표준어 규정, 문장 호응을 준수합니다.
3. 주어와 서술어의 호응을 확인합니다.
4. 부자연스러운 번역투와 불필요한 피동 표현을 피합니다.
5. 문장을 짧고 명확하게 작성합니다.

[숫자 및 범위 표현]

- 25~39세 → 25세부터 39세까지
- 1~2인 가구 → 1인 또는 2인 가구
- 3천만 원 → 삼천만 원
- 11월 13일 → 11월 13일
- 15:00 → 오후 3시

숫자 범위를 하나의 숫자로 합쳐 읽지 않습니다.

[브리핑 구성]

1. 김 대리님께 짧은 인사
2. 회의의 주요 결정사항
3. 김 대리에게 배정된 업무
4. 마감일과 선행 준비사항
5. 짧은 마무리

[작성 원칙]

- 전체 분량은 약 300~350자로 작성합니다.
- 마크다운 기호와 목록 기호는 사용하지 않습니다.
- 실제 음성으로 읽기 자연스럽게 작성합니다.
- 회의록에 없는 정보는 만들지 않습니다.
- 확정사항과 제안·검토사항을 반드시 구분합니다.
- 업무 마감일을 임의로 변경하지 않습니다.
- 작성 후 맞춤법과 문장 호응을 한 번 더 점검합니다.
"""
        },

        {
            "role": "user",
            "content": f"[회의록]\n{current_meeting}"
        }
    ],

    temperature=0
)
                    script = response.output_text

                with st.spinner("음성으로 변환하고 있습니다.."):
                    speech = client.audio.speech.create(
                        model="tts-1",
                        voice=voice,
                        input=script,
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

# ══════════════════════════════════════════════════════════════
# 메뉴 6: 🍃 오늘의 할 일 (김 대리 업무 체크리스트)
#  - ⚠️ 출근길 브리핑 셀 다음에, 이 셀을 한 번만 실행하세요.
#  - 업무 목록은 '📅 마감 캘린더'에서 추출한 일정(st.session_state.schedule)을 그대로 사용
#  - 여기서 완료를 체크하면 캘린더의 완료 표시(✅)에도 바로 반영됨
# ══════════════════════════════════════════════════════════════

def today_group(item, today):
    """업무를 마감 상황별로 묶기"""
    if not item.get("use") or not item.get("due"):
        return "🗓️ 기한 확인 필요"
    if item["due"] < today:
        return "⚠️ 기한 경과"
    if item["due"] == today:
        return "🔥 오늘 마감"
    return "📋 앞으로 할 일"

def save_today_done(index, key):
    """체크박스를 바꾸면 일정 목록의 완료 여부를 저장"""
    st.session_state.schedule[index]["done"] = st.session_state[key]
    st.session_state.sched_ver += 1    # 캘린더의 완료 체크칸도 새 값으로 다시 그리기

if page == PAGES[5]:
    st.subheader("🍃 오늘의 할 일")
    today = kst_today()                 # 캘린더 셀의 한국 시간 기준 오늘
    st.caption(f"{today:%Y.%m.%d} · 김 대리님의 업무")

    # 김 대리 업무만 골라내기 (번호 i는 일정 목록에서의 위치 → 완료 저장에 사용)
    tasks = [(i, it) for i, it in enumerate(st.session_state.schedule)
             if it["kind"] == "김 대리 업무"]

    if not tasks:
        st.info("'📅 마감 캘린더' 메뉴에서 먼저 일정을 추출해 주세요.")
    else:
        pending = [(i, it) for i, it in tasks if not it["done"]]
        done = [(i, it) for i, it in tasks if it["done"]]

        c1, c2, c3 = st.columns(3)
        c1.metric("남은 업무", f"{len(pending)}건")
        c2.metric("오늘 마감", f"{sum(today_group(it, today) == '🔥 오늘 마감' for _, it in pending)}건")
        c3.metric("완료", f"{len(done)}건")

        ver = st.session_state.sched_ver
        for group in ["⚠️ 기한 경과", "🔥 오늘 마감", "📋 앞으로 할 일", "🗓️ 기한 확인 필요"]:
            items = [(i, it) for i, it in pending if today_group(it, today) == group]
            if not items:
                continue
            st.markdown(f"#### {group}")
            items.sort(key=lambda pair: pair[1]["due"] if pair[1]["use"] else date.max)

            for i, it in items:
                key = f"today_{ver}_{i}"
                with st.container(border=True):
                    st.checkbox(it["name"], value=False, key=key,
                                on_change=save_today_done, args=(i, key))
                    if it["use"]:
                        # deadline_label: 캘린더 셀의 'D-3 / 오늘 마감 / 2일 경과' 함수 재사용
                        st.caption(f"📅 마감일: {it['due']:%Y.%m.%d} · {deadline_label(it['due'], today)}")
                        if it["expr"]:
                            st.caption(f"회의록 표현: {it['expr']}")
                    elif it["expr"]:
                        st.caption(f"🗓️ 기한: {it['expr']} · 정확한 날짜 확인 필요 (캘린더에서 지정)")
                    else:
                        st.caption("🗓️ 기한 미정 · 캘린더의 '일정 확인 및 수정'에서 날짜를 정할 수 있어요")

        if not pending:
            st.success("모든 담당 업무를 완료했어요! 🎉")

        with st.expander(f"✅ 완료한 업무 {len(done)}건"):
            for i, it in done:
                key = f"today_done_{ver}_{i}"
                st.checkbox(it["name"], value=True, key=key,
                            on_change=save_today_done, args=(i, key))
            st.caption("체크를 해제하면 미완료 목록으로 돌아가요.")

# ══════════════════════════════════════════════════
# 메뉴 7: 🗂️ 회의 자료실 (날짜/시간별 보존 및 조회)
# ══════════════════════════════════════════════════
elif page == PAGES[6]:
    st.subheader("🗂️ 회의 자료실")
    st.caption(
        "업로드한 회의록을 조회하고, "
        "현재 작업 문서로 불러오거나 삭제할 수 있습니다."
    )

    archives = st.session_state.archived_meetings
    if not archives:
        st.info("아직 저장된 회의 자가 없습니다.")
    else:
        st.markdown(f"**총 보존 회의록:** `{len(archives)}건`")
        # ====================================================
        # 회의 자료실: 문서 조회 / 불러오기 / 2단계 삭제
        # ====================================================

        for idx, item in enumerate(reversed(archives), 1):

            # 기존 아카이브 데이터에 고유 ID가 없는 경우 생성
            if "document_id" not in item:
                item["document_id"] = str(uuid.uuid4())

            document_id = item["document_id"]

            with st.expander(
                f"🕒 [{item['time']}] {item['title']}",
                expanded=(idx == 1)
            ):

                # 1. 문서 정보 표시
                st.caption(
                    f"등록 시각: {item['time']}"
                )

                st.text_area(
                    "회의록 원문",
                    item["content"],
                    height=200,
                    key=f"arc_view_{document_id}"
                )

                # 2. 현재 작업 문서로 불러오기
                if st.button(
                    "📥 이 문서를 현재 작업 문서로 다시 불러오기",
                    key=f"restore_{document_id}"
                ):

                    st.session_state.uploaded_docs[
                        item["title"]
                    ] = item["content"]

                    st.session_state.active_doc = item["title"]

                    st.success(
                        f"'{item['title']}' 문서를 불러왔습니다!"
                    )

                    st.rerun()

                # 3. 문서 삭제 관리
                with st.expander("⚙️ 문서 관리"):

                    # STEP 1: 삭제 요청
                    if st.session_state.delete_target != document_id:

                        if st.button(
                            "🗑️ 문서 삭제",
                            key=f"delete_request_{document_id}"
                        ):

                            st.session_state.delete_target = document_id

                            st.rerun()

                    # STEP 2: 최종 확인
                    else:

                        st.warning(
                            "⚠️ 정말 이 문서를 삭제하시겠습니까?"
                        )

                        st.caption(
                            "삭제하면 회의 자료실에서 제거됩니다."
                        )

                        col_cancel, col_delete = st.columns(2)

                        # 삭제 취소
                        with col_cancel:

                            if st.button(
                                "취소",
                                key=f"cancel_{document_id}",
                                use_container_width=True
                            ):

                                st.session_state.delete_target = None

                                st.rerun()

                        # 최종 삭제
                        with col_delete:

                            if st.button(
                                "최종 삭제",
                                key=f"confirm_delete_{document_id}",
                                type="primary",
                                use_container_width=True
                            ):

                                # 아카이브에서 해당 문서 삭제
                                st.session_state.archived_meetings = [
                                    doc
                                    for doc in st.session_state.archived_meetings
                                    if doc.get("document_id") != document_id
                                ]

                                # 현재 작업 목록에서도 해당 문서 제거
                                st.session_state.uploaded_docs.pop(
                                    item["title"],
                                    None
                                )

                                # 삭제한 문서가 현재 작업 문서라면 변경
                                if st.session_state.active_doc == item["title"]:

                                    remaining = list(
                                        st.session_state.uploaded_docs.keys()
                                    )

                                    st.session_state.active_doc = (
                                        remaining[0]
                                        if remaining
                                        else ""
                                    )

                                # 삭제 확인 상태 초기화
                                st.session_state.delete_target = None

                                st.success("문서가 삭제되었습니다.")

                                st.rerun()
