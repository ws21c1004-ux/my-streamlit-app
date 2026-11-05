
import uuid
import pandas as pd
import streamlit as st
from datetime import datetime
from zoneinfo import ZoneInfo


def show_meeting_archive(classify_func=None, now_func=None):

    st.subheader("🗂️ 회의 자료실")

    # 세션 상태 초기화
    if "archived_meetings" not in st.session_state:
        st.session_state.archived_meetings = []

    if "uploaded_docs" not in st.session_state:
        st.session_state.uploaded_docs = {}

    if "active_doc" not in st.session_state:
        st.session_state.active_doc = ""

    if "delete_target" not in st.session_state:
        st.session_state.delete_target = None

    archives = st.session_state.archived_meetings

    # ==================================================
    # 1. 저장된 회의록 조회
    # ==================================================

    if archives:

        st.markdown(
            f"**총 보존 회의록:** `{len(archives)}건`"
        )

        for idx, item in enumerate(reversed(archives), 1):

            if "document_id" not in item:
                item["document_id"] = str(uuid.uuid4())

            document_id = item["document_id"]
            title = item.get("title", "제목 없음")
            content = item.get("content", "")
            saved_time = item.get("time", "시간 미정")

            with st.expander(
                f"🕒 [{saved_time}] {title}",
                expanded=(idx == 1)
            ):

                st.caption(f"등록 시각: {saved_time}")

                st.text_area(
                    "회의록 원문",
                    value=content,
                    height=200,
                    key=f"arc_view_{document_id}"
                )

                # 현재 작업 문서로 불러오기
                if st.button(
                    "📥 이 문서를 현재 작업 문서로 다시 불러오기",
                    key=f"restore_{document_id}"
                ):

                    st.session_state.uploaded_docs[title] = content
                    st.session_state.active_doc = title

                    st.success(
                        f"'{title}' 문서를 불러왔습니다!"
                    )

                    st.rerun()

                # 문서 삭제 관리
                with st.expander("⚙️ 문서 관리"):

                    if st.session_state.delete_target != document_id:

                        if st.button(
                            "🗑️ 문서 삭제",
                            key=f"delete_request_{document_id}"
                        ):

                            st.session_state.delete_target = document_id
                            st.rerun()

                    else:

                        st.warning(
                            "⚠️ 정말 이 문서를 삭제하시겠습니까?"
                        )

                        col_cancel, col_delete = st.columns(2)

                        with col_cancel:

                            if st.button(
                                "취소",
                                key=f"cancel_{document_id}",
                                use_container_width=True
                            ):

                                st.session_state.delete_target = None
                                st.rerun()

                        with col_delete:

                            if st.button(
                                "최종 삭제",
                                key=f"confirm_delete_{document_id}",
                                type="primary",
                                use_container_width=True
                            ):

                                st.session_state.archived_meetings = [
                                    doc for doc in archives
                                    if doc.get("document_id") != document_id
                                ]

                                st.session_state.uploaded_docs.pop(
                                    title, None
                                )

                                if st.session_state.active_doc == title:

                                    remaining = list(
                                        st.session_state.uploaded_docs.keys()
                                    )

                                    st.session_state.active_doc = (
                                        remaining[0] if remaining else ""
                                    )

                                st.session_state.delete_target = None
                                st.success("문서가 삭제되었습니다.")
                                st.rerun()

    # ==================================================
    # 2. 업무 보고서 다운로드
    # ==================================================

    st.divider()
    st.subheader("📥 업무 보고서 다운로드")

    st.caption(
        "현재 선택한 회의록과 업무 우선순위 분석 결과를 다운로드합니다."
    )

    doc = st.session_state.get("active_doc", "")
    docs = st.session_state.get("uploaded_docs", {})
    meeting = docs.get(doc, "")

    pdata = (
        st.session_state.get(f"priority_data_{doc}")
        if doc else None
    )

    tasks = (
        pdata.get("업무목록", [])
        if isinstance(pdata, dict)
        else []
    )

    task_df = pd.DataFrame(
        tasks if isinstance(tasks, list) else []
    )

    columns = [
        "업무명", "담당자", "기한",
        "긴급도", "중요도", "근거"
    ]

    for col in columns:
        if col not in task_df.columns:
            task_df[col] = ""

    task_df = task_df[columns].copy()

    # 기존 classify 함수가 있으면 사용
    if not task_df.empty:

        if classify_func is not None:
            task_df["우선순위"] = task_df.apply(
                lambda row: classify_func(
                    row["긴급도"],
                    row["중요도"]
                ),
                axis=1
            )
        else:
            task_df["우선순위"] = "미분류"

    else:
        task_df["우선순위"] = pd.Series(dtype=str)

    # 생성 시각
    if now_func is not None:
        now = now_func()
    else:
        now = datetime.now(
            ZoneInfo("Asia/Seoul")
        ).strftime("%Y-%m-%d %H:%M:%S")

    # Markdown 보고서
    lines = [
        "# 에이블 AI 업무 보고서",
        "",
        f"- 생성일: {now}",
        f"- 회의 문서: {doc or '선택 없음'}",
        "",
        "## 1. 회의록",
        meeting or "회의록이 없습니다.",
        "",
        "## 2. 업무 목록"
    ]

    if task_df.empty:
        lines.append(
            "분석된 업무가 없습니다. 업무 우선순위 분석 메뉴에서 먼저 분석하세요."
        )
    else:
        for _, row in task_df.iterrows():
            lines.extend([
                f"### {row['업무명']}",
                f"- 담당자: {row['담당자'] or '미기재'}",
                f"- 마감일: {row['기한'] or '기한 미정'}",
                f"- 우선순위: {row['우선순위']}",
                f"- 근거: {row['근거']}",
                ""
            ])

    report = "\n".join(lines)

    st.download_button(
        label="📄 Markdown 업무 보고서 다운로드",
        data=report.encode("utf-8-sig"),
        file_name="able_work_report.md",
        mime="text/markdown",
        use_container_width=True,
        key="meeting_archive_md_download"
    )

    st.download_button(
        label="📊 CSV 업무 목록 다운로드",
        data=task_df.to_csv(index=False).encode("utf-8-sig"),
        file_name="able_tasks.csv",
        mime="text/csv",
        use_container_width=True,
        key="meeting_archive_csv_download"
    )

    with st.expander("보고서 미리보기"):
        st.markdown(report)
