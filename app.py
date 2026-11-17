
import os
import re
import unicodedata
import streamlit as st

from langchain_pymupdf4llm import PyMuPDF4LLMLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_chroma import Chroma
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough, RunnableLambda
from langchain_core.output_parsers import StrOutputParser


# ==========================================
# 01번 노트북에서 만든 RAG 파이프라인 함수
# ==========================================

def rag_pipeline(filepath):

    # PDF 파일 읽기
    loader = PyMuPDF4LLMLoader(filepath)
    docs = loader.load()

    # 텍스트 분할
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        separators=["\n\n", "\n", ". ", " ", ""]
    )
    split_docs = text_splitter.split_documents(docs)

    # Embedding 모델
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-small"
    )

    # 벡터저장소 만들기
    vectorstore = Chroma(
        collection_name="aivle_docs",
        persist_directory="chroma_db",
        embedding_function=embeddings
    )

    # 기존 데이터 비우기
    vectorstore.reset_collection()

    # 문서 추가
    vectorstore.add_documents(split_docs)

    # 검색기 구성
    retriever = vectorstore.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k": 8,
            "fetch_k": 30,
            "lambda_mult": 0.65
        }
    )

    # 특정 미니프로젝트 차수 검색 보강
    def search_docs(question):

        semantic_docs = retriever.invoke(question)
        match = re.search(r"([1-7])\s*차", question)
        keyword_docs = []

        if match and (
            "미니프로젝트" in question
            or "미프" in question
        ):
            number = match.group(1)

            patterns = [
                f"{number}차미니프로젝트",
                f"{number}차미프"
            ]

            for doc in split_docs:
                content = re.sub(r"\s+", "", doc.page_content)

                if any(p in content for p in patterns):
                    keyword_docs.append(doc)

            keyword_docs.sort(
                key=lambda d: (
                    "무엇을했나요" in re.sub(
                        r"\s+", "", d.page_content
                    ),
                    "Q011" in d.page_content
                    if number == "1" else False
                ),
                reverse=True
            )

        combined_docs = keyword_docs[:5] + semantic_docs

        unique_docs = []
        seen = set()

        for doc in combined_docs:
            key = (
                doc.metadata.get("page"),
                doc.page_content
            )

            if key not in seen:
                unique_docs.append(doc)
                seen.add(key)

        return unique_docs[:10]

    # LLM 선언
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0
    )

    # 프롬프트 구성
    prompt = ChatPromptTemplate.from_template("""
당신은 KT AIVLE School 백서 기반 AI 비서입니다.

다음 문서 내용을 근거로 질문에 답변하세요.

[답변 규칙]
1. 문서에 있는 내용을 우선하여 답변하세요.
2. 미니프로젝트 차수를 묻는 경우 해당 차수의 내용을 확인하세요.
3. 프로젝트 주제, 수행 내용, 결과를 구체적으로 설명하세요.
4. 9기와 10기 사례를 혼동하지 마세요.
5. 문서에 없는 내용은 추측하지 마세요.
6. 근거가 부족하면 확인할 수 없다고 답변하세요.
7. 한국어로 명확하게 답변하세요.

[문서 내용]
{context}

[질문]
{question}

[답변]
""")

    # 문서 포맷터
    def format_docs(docs):
        formatted = []

        for doc in docs:
            page = doc.metadata.get("page")

            if page is not None:
                source = f"PDF {int(page) + 1}페이지"
            else:
                source = "페이지 정보 없음"

            formatted.append(
                f"[출처: {source}]\n{doc.page_content}"
            )

        return "\n\n".join(formatted)

    # RAG 체인 구성
    rag_chain = (
        {
            "context": RunnableLambda(search_docs) | format_docs,
            "question": RunnablePassthrough()
        }
        | prompt
        | llm
        | StrOutputParser()
    )

    return rag_chain


# ==========================================
# Streamlit 화면 설정
# ==========================================

st.set_page_config(
    page_title="AIVLE School AI 학습 도우미",
    page_icon="📘",
    layout="wide"
)

st.title("📘 KT AIVLE School AI 학습 도우미")
st.caption("AIVLE School 백서를 기반으로 질문에 답변하는 RAG 서비스")

# PDF 경로
path = "/content/drive/MyDrive/project04/"
target_name = "AIVLE School 백서.pdf"


# ==========================================
# PDF 파일 찾기
# ==========================================

def find_pdf():
    if not os.path.isdir(path):
        return None

    pdf_files = [
        f for f in os.listdir(path)
        if unicodedata.normalize("NFC", f).casefold()
        == unicodedata.normalize("NFC", target_name).casefold()
    ]

    if not pdf_files:
        return None

    return os.path.join(path, pdf_files[0])


# ==========================================
# RAG 체인 초기화
# ==========================================

@st.cache_resource(show_spinner="백서 데이터를 불러오는 중입니다...")
def load_rag(filepath, modified_time):
    return rag_pipeline(filepath)


filepath = find_pdf()

if filepath is None:
    st.error("백서 PDF 파일을 찾을 수 없습니다.")
    st.info("Google Drive의 project04 폴더와 파일명을 확인하세요.")
    st.stop()

if not os.environ.get("OPENAI_API_KEY"):
    st.error("OPENAI_API_KEY 환경변수가 설정되지 않았습니다.")
    st.stop()

try:
    rag_chain = load_rag(
        filepath,
        os.path.getmtime(filepath)
    )
    st.success("✅ AIVLE School 백서 연결 완료!")
except Exception as e:
    st.error(f"RAG 초기화 오류: {e}")
    st.stop()


# ==========================================
# 사용자 질문 입력
# ==========================================

st.subheader("💬 백서에 질문하기")

question = st.text_input(
    "궁금한 내용을 입력하세요.",
    placeholder="예: 1차 미니프로젝트에서는 무엇을 했나요?"
)

if st.button("🔎 질문하기", type="primary"):

    if not question.strip():
        st.warning("질문을 입력해 주세요.")

    else:
        with st.spinner("백서에서 답변을 검색하고 있습니다..."):
            try:
                result = rag_chain.invoke(question)

                st.markdown("### 🤖 AI 답변")
                st.write(result)

            except Exception as e:
                st.error(f"답변 생성 중 오류: {e}")



