
import os
import re
from pathlib import Path

import streamlit as st
import chromadb

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI


# =====================================================
# PAGE CONFIGURATION
# =====================================================

st.set_page_config(
    page_title="Enterprise RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)

BASE_DIR = Path(__file__).parent

PDF_FILES = [
    BASE_DIR / "NLP_Units1-2_Combined.pdf",
    BASE_DIR / "NLP_Unit2_Combined.pdf"
]

CHROMA_PATH = BASE_DIR / "chroma_db"


# =====================================================
# PAGE TITLE
# =====================================================

st.title("🤖 Enterprise Knowledge RAG Chatbot")

st.caption(
    "Ask questions about the information in your knowledge base."
)


# =====================================================
# LOAD EMBEDDING MODEL
# =====================================================

@st.cache_resource
def load_embedding_model():
    return SentenceTransformer("all-MiniLM-L6-v2")


model = load_embedding_model()


# =====================================================
# LOAD CHROMADB
# =====================================================

@st.cache_resource
def load_chroma_collection():

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = client.get_or_create_collection(
        name="nlp_combined"
    )

    return collection


collection = load_chroma_collection()


# =====================================================
# CREATE KNOWLEDGE BASE IF EMPTY
# =====================================================

if collection.count() == 0:

    with st.spinner("Preparing knowledge base..."):

        all_documents = []
        all_metadatas = []

        for pdf_path in PDF_FILES:

            if not pdf_path.exists():
                continue

            reader = PdfReader(str(pdf_path))

            pdf_text = ""

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:
                    pdf_text += page_text + " "

            pdf_text = re.sub(
                r"\s+",
                " ",
                pdf_text
            ).strip()

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=150
            )

            chunks = splitter.split_text(pdf_text)

            for index, chunk in enumerate(chunks):

                all_documents.append(chunk)

                all_metadatas.append({
                    "source": pdf_path.name,
                    "chunk_index": index
                })

        if all_documents:

            embeddings = model.encode(
                all_documents
            ).tolist()

            ids = [
                f"chunk_{i}"
                for i in range(len(all_documents))
            ]

            collection.upsert(
                ids=ids,
                documents=all_documents,
                embeddings=embeddings,
                metadatas=all_metadatas
            )

        else:

            st.error("No PDF documents found.")


# =====================================================
# GEMINI MODEL
# =====================================================

api_key = st.secrets.get(
    "GOOGLE_API_KEY",
    os.environ.get("GOOGLE_API_KEY")
)

if not api_key:
    st.error("GOOGLE_API_KEY is not configured.")
    st.stop()


@st.cache_resource
def load_llm():

    return ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        temperature=0
    )


llm = load_llm()


# =====================================================
# ABBREVIATION MAP
# =====================================================

abbreviation_map = {

    "tf-idf": "term frequency inverse document frequency",
    "tfidf": "term frequency inverse document frequency",

    "vsm": "vector space model",
    "nlp": "natural language processing",

    "ml": "machine learning",
    "ai": "artificial intelligence",
    "dl": "deep learning",

    "wt": "what",
    "wht": "what",
    "hw": "how",

    "y": "why",
    "u": "you",
    "ur": "your",
    "r": "are",

    "bcz": "because",
    "bc": "because",

    "pls": "please",
    "plz": "please",

    "smpl": "simple",
    "abt": "about",
    "btw": "by the way",
    "tht": "that"
}


# =====================================================
# EXPAND ABBREVIATIONS
# =====================================================

def expand_abbreviations(question):

    expanded = question.strip()

    for abbreviation in sorted(
        abbreviation_map,
        key=len,
        reverse=True
    ):

        meaning = abbreviation_map[abbreviation]

        pattern = (
            r"\b"
            + re.escape(abbreviation)
            + r"\b"
        )

        expanded = re.sub(
            pattern,
            meaning,
            expanded,
            flags=re.IGNORECASE
        )

    return expanded


# =====================================================
# ABBREVIATION TESTING SECTION
# =====================================================

with st.expander("Test abbreviation recognition"):

    test_question = st.text_input(
        "Enter a question to test",
        placeholder="Example: wt is ml"
    )

    if test_question:

        st.write(
            "Original question:",
            test_question
        )

        st.write(
            "Expanded question:",
            expand_abbreviations(test_question)
        )


# =====================================================
# FOLLOW-UP QUESTION DETECTION
# =====================================================

def is_follow_up_question(question):

    question = question.lower().strip()

    follow_up_phrases = [

        "why this",
        "why that",
        "why it",
        "why is this",
        "why is it",
        "why we use",
        "why do we use",
        "why is this used",
        "why is it used",
        "why this is used",

        "explain this",
        "explain it",
        "explain more",
        "explain simply",

        "in simple",
        "in simple words",
        "in easy words",
        "more simply",

        "what does this mean",
        "what does it mean",

        "tell me more",
        "simplify this",
        "simplify it",

        "how this works",
        "how it works",
        "how does this work",
        "how does it work"
    ]

    return any(
        phrase in question
        for phrase in follow_up_phrases
    )


# =====================================================
# SESSION STATE
# =====================================================

if "messages" not in st.session_state:
    st.session_state.messages = []

if "conversation_history" not in st.session_state:
    st.session_state.conversation_history = []

if "last_retrieved_docs" not in st.session_state:
    st.session_state.last_retrieved_docs = []

if "last_retrieved_metas" not in st.session_state:
    st.session_state.last_retrieved_metas = []


# =====================================================
# RAG FUNCTION
# =====================================================

def ask_rag(question, k=3):

    expanded_question = expand_abbreviations(question)

    follow_up = is_follow_up_question(
        expanded_question
    )

    retrieval_distances = []

    # =================================================
    # RETRIEVE DOCUMENTS
    # =================================================

    if (
        follow_up
        and st.session_state.last_retrieved_docs
    ):

        relevant_docs = (
            st.session_state.last_retrieved_docs
        )

        relevant_metas = (
            st.session_state.last_retrieved_metas
        )

        retrieval_method = "Previous retrieved documents"

    else:

        retrieval_method = "Fresh document retrieval"

        if collection.count() == 0:

            return (
                "The knowledge base is empty. Please check the PDF files.",
                [],
                retrieval_method,
                []
            )

        query_embedding = model.encode(
            [expanded_question]
        ).tolist()

        results = collection.query(
            query_embeddings=query_embedding,
            n_results=min(k, collection.count()),
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        docs = results["documents"][0]
        metas = results["metadatas"][0]
        distances = results["distances"][0]

        relevant_docs = []
        relevant_metas = []

        # Keep the existing distance threshold.

        threshold = 1.1

        for doc, meta, distance in zip(
            docs,
            metas,
            distances
        ):

            retrieval_distances.append(
                round(float(distance), 4)
            )

            if distance <= threshold:

                relevant_docs.append(doc)
                relevant_metas.append(meta)

        # Save the retrieved documents for follow-ups.

        st.session_state.last_retrieved_docs = (
            relevant_docs
        )

        st.session_state.last_retrieved_metas = (
            relevant_metas
        )

    # =================================================
    # CHECK RETRIEVAL
    # =================================================

    if not relevant_docs:

        answer = (
            "I couldn't find this information "
            "in the provided documents."
        )

        st.session_state.conversation_history.append({
            "role": "user",
            "content": expanded_question
        })

        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": answer
        })

        return (
            answer,
            [],
            retrieval_method,
            retrieval_distances
        )

    # =================================================
    # PREPARE DOCUMENT CONTEXT
    # =================================================

    context = "\n\n".join(

        f"Source: {meta.get('source', 'Unknown')}\n{doc}"

        for doc, meta in zip(
            relevant_docs,
            relevant_metas
        )
    )

    # =================================================
    # PREPARE CONVERSATION HISTORY
    # =================================================

    history = "\n".join(

        f"{message['role']}: {message['content']}"

        for message in
        st.session_state.conversation_history[-6:]
    )

    # =================================================
    # GEMINI PROMPT
    # =================================================

    prompt = f"""
You are a friendly tutor helping a college student.

Previous conversation:
{history}

Original question:
{question}

Corrected question:
{expanded_question}

Document information:
{context}

Instructions:

1. Understand texting abbreviations using the corrected question.
2. Understand follow-up questions using previous conversation.
3. If the user asks "why this?", identify the previous topic.
4. Answer in simple student-friendly language.
5. Explain in your own words. Do not copy the PDF.
6. Answer the exact question.
7. Use ONLY the document information.
8. Do not invent facts or use outside information.
9. Give an example only when supported by the documents.
10. Keep the answer clear and concise.

If the required information is not in the document context,
reply exactly:

I couldn't find this information in the provided documents.

Answer:
"""

    # =================================================
    # GENERATE ANSWER
    # =================================================

    response = llm.invoke(prompt)

    if isinstance(response.content, str):

        answer = response.content

    elif isinstance(response.content, list):

        answer = "\n".join(
            item.get("text", "")
            for item in response.content
            if isinstance(item, dict)
            and item.get("type") == "text"
        )

    else:

        answer = str(response.content)

    # =================================================
    # SAVE CONVERSATION
    # =================================================

    st.session_state.conversation_history.append({
        "role": "user",
        "content": expanded_question
    })

    st.session_state.conversation_history.append({
        "role": "assistant",
        "content": answer
    })

    # =================================================
    # SOURCES
    # =================================================

    if "I couldn't find this information" in answer:

        sources = []

    else:

        sources = list(dict.fromkeys(
            meta.get("source", "Unknown")
            for meta in relevant_metas
        ))

    return (
        answer,
        sources,
        retrieval_method,
        retrieval_distances
    )


# =====================================================
# DISPLAY CHAT HISTORY
# =====================================================

for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(message["content"])

        if message.get("sources"):

            with st.expander("📚 Sources"):

                for source in message["sources"]:
                    st.write(source)


# =====================================================
# CHAT INPUT
# =====================================================

question = st.chat_input(
    "Ask a question..."
)


if question:

    # Display user question.

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)

    # Expand question.

    expanded_question = expand_abbreviations(question)

    follow_up = is_follow_up_question(
        expanded_question
    )

    # =================================================
    # GENERATE ANSWER
    # =================================================

    with st.chat_message("assistant"):

        with st.spinner("Thinking..."):

            try:

                answer, sources, method, distances = ask_rag(
                    question
                )

            except Exception as error:

                answer = (
                    "Something went wrong. Please try again."
                )

                sources = []
                method = "Error"
                distances = []

                st.error(str(error))

        st.markdown(answer)

        if sources:

            with st.expander("📚 Sources"):

                for source in sources:
                    st.write(source)

        # =================================================
        # DEBUG INFORMATION
        # =================================================

        with st.expander("Question processing and retrieval details"):

            st.write(
                "Original question:",
                repr(question)
            )

            st.write(
                "Expanded question:",
                repr(expanded_question)
            )

            st.write(
                "Follow-up detected:",
                follow_up
            )

            st.write(
                "Retrieval method:",
                method
            )

            st.write(
                "Retrieval distances:",
                distances
            )

            st.write(
                "Retrieved document count:",
                len(st.session_state.last_retrieved_docs)
            )

    # Save assistant message.

    st.session_state.messages.append({
        "role": "assistant",
        "content": answer,
        "sources": sources
    })


# =====================================================
# SIDEBAR
# =====================================================

with st.sidebar:

    st.header("Chat Controls")

    if st.button(
        "🗑️ New Chat",
        use_container_width=True
    ):

        st.session_state.messages = []

        st.session_state.conversation_history = []

        st.session_state.last_retrieved_docs = []

        st.session_state.last_retrieved_metas = []

        st.rerun()

    st.divider()

    st.subheader("Knowledge Base")

    st.write("NLP Units 1–2")

    st.write(
        f"Stored chunks: {collection.count()}"
    )

    st.divider()

    st.caption(
        "AI Retrieval-Augmented Generation Chatbot"
    )
