import streamlit as st
from rag_engine import load_base, ask_rag

st.set_page_config(page_title="Heritage AI", layout="centered")


# =========================================================
# COLORFUL SIDEBAR
# =========================================================

with st.sidebar:

    st.markdown("""
    <style>

    [data-testid="stSidebar"] {
        background: linear-gradient(
            180deg,
            #fff7ed 0%,
            #fef3c7 35%,
            #fce7f3 70%,
            #ede9fe 100%
        );
    }

    [data-testid="stSidebar"] p {
        color: #374151;
    }

    .side-title {
        color: #7c2d12;
        text-align: center;
        font-size: 28px;
        font-weight: bold;
    }

    .side-subtitle {
        color: #9333ea;
        text-align: center;
        font-size: 13px;
    }

    .pipeline-title {
        color: #2563eb;
        font-size: 21px;
        font-weight: bold;
    }

    .info-title {
        color: #db2777;
        font-size: 21px;
        font-weight: bold;
    }

    .pipeline-item {
        padding: 8px;
        margin: 5px 0;
        border-radius: 10px;
        font-weight: bold;
        font-size: 14px;
    }

    .p1 { background-color: #dbeafe; color: #1d4ed8; }
    .p2 { background-color: #dcfce7; color: #15803d; }
    .p3 { background-color: #fef3c7; color: #b45309; }
    .p4 { background-color: #fce7f3; color: #be185d; }
    .p5 { background-color: #ede9fe; color: #7e22ce; }
    .p6 { background-color: #cffafe; color: #0e7490; }
    .p7 { background-color: #ffedd5; color: #c2410c; }
    .p8 { background-color: #dcfce7; color: #166534; }

    .info-box {
        padding: 10px;
        margin: 8px 0;
        border-radius: 10px;
    }

    .model-box {
        background-color: #dbeafe;
        border-left: 5px solid #2563eb;
    }

    .tech-box {
        background-color: #dcfce7;
        border-left: 5px solid #16a34a;
    }

    .member-box {
        background-color: #fce7f3;
        border-left: 5px solid #db2777;
    }

    .goal-box {
        background-color: #fef3c7;
        border-left: 5px solid #f59e0b;
    }

    .info-heading {
        font-weight: bold;
        font-size: 16px;
    }

    </style>
    """, unsafe_allow_html=True)

    st.markdown(
        '<div class="side-title">📜 Heritage AI</div>',
        unsafe_allow_html=True
    )

    st.markdown(
        '<div class="side-subtitle">✨ Explore History with AI ✨</div>',
        unsafe_allow_html=True
    )

    st.markdown("---")

    if st.button("🔄 Pipeline", use_container_width=True):
        st.session_state["sidebar_page"] = "pipeline"

    if st.button("ℹ️ Info", use_container_width=True):
        st.session_state["sidebar_page"] = "info"

    st.markdown("---")

    if st.session_state.get("sidebar_page") == "pipeline":

        st.markdown(
            '<div class="pipeline-title">🔄 RAG Pipeline</div>',
            unsafe_allow_html=True
        )

        st.caption("How our History AI works")

        st.markdown(
            '<div class="pipeline-item p1">📚 1. History Documents</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="pipeline-item p2">📄 2. Document Loading</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="pipeline-item p3">✂️ 3. Text Chunking</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="pipeline-item p4">🔢 4. Embeddings</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="pipeline-item p5">🗄️ 5. Vector Index</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="pipeline-item p6">🔍 6. Retrieval</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="pipeline-item p7">🧠 7. LLM</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            '<div class="pipeline-item p8">💬 8. Final Answer</div>',
            unsafe_allow_html=True
        )

    elif st.session_state.get("sidebar_page") == "info":

        st.markdown(
            '<div class="info-title">ℹ️ About Heritage AI</div>',
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-box model-box">
                <div class="info-heading">🤖 Our Model</div>
                <p>
                Heritage AI is a RAG-based history chatbot.
                It retrieves relevant historical information from
                our knowledge base and uses an LLM to generate
                contextual answers.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-box tech-box">
                <div class="info-heading">🛠️ Technologies Used</div>
                <p>
                🐍 Python<br>
                🎈 Streamlit<br>
                🔎 RAG<br>
                🔢 Embeddings<br>
                🗄️ Vector Index<br>
                🧠 Groq LLM
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-box member-box">
                <div class="info-heading">👥 Group Members</div>
                <p>
                👤 Lohita R Patil<br>
                👤 Member 2<br>
                👤 Member 3<br>
                👤 Member 4
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

        st.markdown(
            """
            <div class="info-box goal-box">
                <div class="info-heading">🎯 Our Goal</div>
                <p>
                Make history learning interactive, simple
                and accessible using Artificial Intelligence.
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )


# =========================================================
# MAIN CHATBOT
# =========================================================

st.title("Heritage AI")

st.caption(
    "Learn from the past, understand the present, and explore history "
    "through an interactive AI-powered chatbot."
)

st.caption("RAG pipeline with multiple files")
st.caption("Created by Group10")

my_key = st.secrets["GROQ_API_KEY"]


@st.cache_resource
def initiate_rag():
    return load_base("history")


index, embed_model, chunks = initiate_rag()


if "chat_hist" not in st.session_state:
    st.session_state.chat_hist = []


for qa in st.session_state.chat_hist:

    with st.chat_message(qa["role"]):
        st.write(qa["content"])


question = st.chat_input(
    "Ask anything about history and archaeology..."
)


if question:

    st.session_state.chat_hist.append(
        {"role": "user", "content": question}
    )

    with st.chat_message("user"):
        st.write(question)

    with st.chat_message("assistant"):

        with st.spinner("Asking RAG for answer"):

            answer = ask_rag(
                question,
                embed_model,
                chunks,
                index,
                api_key=my_key
            )

            st.write(answer)

            st.session_state.chat_hist.append(
                {"role": "assistant", "content": answer}
            )


if st.button("Clear chat"):

    st.session_state.chat_hist = []
    st.rerun()
