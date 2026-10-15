import os
import re
from pathlib import Path

import streamlit as st
import chromadb

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Enterprise RAG Chatbot",
    page_icon="🤖",
    layout="wide"
)


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = Path(__file__).parent

PDF_FILES = [
    BASE_DIR / "NLP_Units1-2_Combined.pdf",
    BASE_DIR / "NLP_Unit2_Combined.pdf"
]

CHROMA_PATH = BASE_DIR / "chroma_db"


# ============================================================
# TITLE
# ============================================================

st.title("🤖 Enterprise Knowledge RAG Chatbot")

st.caption(
    "Ask questions about the information available in the NLP knowledge base."
)


# ============================================================
# LOAD EMBEDDING MODEL
# ============================================================

@st.cache_resource
def load_embedding_model():

    return SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


model = load_embedding_model()


# ============================================================
# LOAD CHROMADB
# ============================================================

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


# ============================================================
# CREATE CHROMADB DATA IF EMPTY
# ============================================================

if collection.count() == 0:

    with st.spinner(
        "Preparing the knowledge base for the first time..."
    ):

        all_documents = []
        all_metadatas = []

        # ----------------------------------------------------
        # Read PDF files
        # ----------------------------------------------------

        for pdf_path in PDF_FILES:

            if not pdf_path.exists():
                continue

            reader = PdfReader(
                str(pdf_path)
            )

            pdf_text = ""

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:
                    pdf_text += page_text + " "

            # ------------------------------------------------
            # Clean whitespace
            # ------------------------------------------------

            pdf_text = re.sub(
                r"\s+",
                " ",
                pdf_text
            ).strip()

            # ------------------------------------------------
            # Split document into chunks
            # ------------------------------------------------

            text_splitter = RecursiveCharacterTextSplitter(

                chunk_size=1000,

                chunk_overlap=150,

                separators=[
                    ". ",
                    "? ",
                    "! ",
                    "; ",
                    ", ",
                    " ",
                    ""
                ]
            )

            chunks = text_splitter.split_text(
                pdf_text
            )

            # ------------------------------------------------
            # Store chunks and metadata
            # ------------------------------------------------

            for index, chunk in enumerate(chunks):

                all_documents.append(chunk)

                all_metadatas.append({

                    "source": pdf_path.name,

                    "chunk_index": index

                })

        # ----------------------------------------------------
        # Generate embeddings
        # ----------------------------------------------------

        if all_documents:

            embeddings = model.encode(
                all_documents
            ).tolist()

            ids = [
                f"chunk_{index}"
                for index in range(
                    len(all_documents)
                )
            ]

            collection.upsert(

                ids=ids,

                documents=all_documents,

                embeddings=embeddings,

                metadatas=all_metadatas

            )


# ============================================================
# GEMINI API
# ============================================================

api_key = os.environ.get(
    "GOOGLE_API_KEY"
)


if not api_key:

    st.error(
        "GOOGLE_API_KEY is not configured."
    )

    st.stop()


@st.cache_resource
def load_llm():

    return ChatGoogleGenerativeAI(

        model="gemini-3.5-flash-lite",

        google_api_key=api_key,

        temperature=0

    )


llm = load_llm()


# ============================================================
# ABBREVIATION EXPANSION
# ============================================================

abbreviation_map = {

    "vsm": "vector space model",

    "tf-idf": "term frequency inverse document frequency",

    "tfidf": "term frequency inverse document frequency",

    "nlp": "natural language processing"

}


def expand_abbreviations(question):

    expanded_question = question

    for abbreviation, meaning in abbreviation_map.items():

        pattern = (
            r"\b"
            + re.escape(abbreviation)
            + r"\b"
        )

        expanded_question = re.sub(

            pattern,

            f"{abbreviation} ({meaning})",

            expanded_question,

            flags=re.IGNORECASE

        )

    return expanded_question


# ============================================================
# SESSION STATE
# ============================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "conversation_history" not in st.session_state:

    st.session_state.conversation_history = []


if "last_retrieved_docs" not in st.session_state:

    st.session_state.last_retrieved_docs = []


if "last_retrieved_metas" not in st.session_state:

    st.session_state.last_retrieved_metas = []


# ============================================================
# FOLLOW-UP DETECTION
# ============================================================

def is_follow_up_question(question):

    question_lower = question.lower().strip()

    follow_up_phrases = [

        "in simple",

        "in simple?",

        "explain simply",

        "explain this",

        "explain it",

        "explain more",

        "more simply",

        "why is this used",

        "why is it used",

        "why is this useful",

        "why is it useful",

        "what does this mean",

        "what does it mean",

        "tell me more",

        "simplify this",

        "simplify it",

        "explain in simple",

        "explain in simple words",

        "in simple words"

    ]

    return any(

        phrase in question_lower

        for phrase in follow_up_phrases

    )


# ============================================================
# RAG FUNCTION
# ============================================================

def ask_rag(question, k=3):

    # --------------------------------------------------------
    # Check for follow-up question
    # --------------------------------------------------------

    follow_up = is_follow_up_question(
        question
    )


    # ========================================================
    # RETRIEVAL
    # ========================================================

    if (

        follow_up

        and st.session_state.last_retrieved_docs

    ):

        # ----------------------------------------------------
        # Follow-up question
        #
        # Reuse previous retrieved documents.
        # ----------------------------------------------------

        relevant_docs = (
            st.session_state.last_retrieved_docs
        )

        relevant_metas = (
            st.session_state.last_retrieved_metas
        )


    else:

        # ----------------------------------------------------
        # New question
        #
        # Perform a fresh ChromaDB search.
        # ----------------------------------------------------

        expanded_question = expand_abbreviations(
            question
        )


        # ----------------------------------------------------
        # Create question embedding
        # ----------------------------------------------------

        query_embedding = model.encode(

            [expanded_question]

        ).tolist()


        # ----------------------------------------------------
        # Search ChromaDB
        # ----------------------------------------------------

        results = collection.query(

            query_embeddings=query_embedding,

            n_results=k,

            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )


        docs = results["documents"][0]

        metas = results["metadatas"][0]

        distances = results["distances"][0]


        # ----------------------------------------------------
        # Similarity threshold
        # ----------------------------------------------------

        threshold = 1.1


        relevant_docs = []

        relevant_metas = []


        # ----------------------------------------------------
        # Filter relevant chunks
        # ----------------------------------------------------

        for doc, meta, distance in zip(

            docs,

            metas,

            distances

        ):

            if distance <= threshold:

                relevant_docs.append(doc)

                relevant_metas.append(meta)


        # ----------------------------------------------------
        # Save retrieved documents
        #
        # They can be reused for a follow-up question.
        # ----------------------------------------------------

        st.session_state.last_retrieved_docs = (
            relevant_docs
        )

        st.session_state.last_retrieved_metas = (
            relevant_metas
        )


    # ========================================================
    # NO RELEVANT INFORMATION
    # ========================================================

    if not relevant_docs:

        answer = (
            "I couldn't find this information "
            "in the provided documents."
        )


        # Save conversation

        st.session_state.conversation_history.append({

            "role": "user",

            "content": question

        })


        st.session_state.conversation_history.append({

            "role": "assistant",

            "content": answer

        })


        return answer, []


    # ========================================================
    # DOCUMENT CONTEXT
    # ========================================================

    context = "\n\n".join(

        f"Source: {meta.get('source', 'Unknown')}\n{doc}"

        for doc, meta in zip(

            relevant_docs,

            relevant_metas

        )
    )


    # ========================================================
    # CONVERSATION HISTORY
    # ========================================================

    history_text = "\n".join(

        f"{message['role']}: {message['content']}"

        for message in
        st.session_state.conversation_history[-6:]

    )


    # ========================================================
    # GEMINI PROMPT
    # ========================================================

    prompt = f"""
You are a friendly NLP tutor helping a college student.

Use ONLY the information in the provided document context.

Previous conversation:

{history_text}

Current student question:

{question}

Document context:

{context}

IMPORTANT:

- Understand the current question using the previous conversation.
- If the student says "it", "this", "that", "in simple",
  "explain more", or "why is it used", understand what
  they are referring to from the conversation.
- If this is a follow-up question, explain the previous topic
  in the way the student is asking for.
- Answer the current question directly.
- Do NOT copy sentences from the document.
- Do NOT simply repeat the document definition.
- Explain the information in your own simple words.
- Use easy language suitable for a college student.
- Start with a short, direct answer.
- Then explain it briefly.
- Use an example only when supported by the document.
- Do not add outside information.
- Do not say "Based on the provided documents".
- Do not mention that you are an AI model.
- Keep the answer concise and useful for studying.

If the required information is not available in the document
context, say exactly:

I couldn't find this information in the provided documents.

Answer:
"""


    # ========================================================
    # GENERATE ANSWER
    # ========================================================

    response = llm.invoke(
        prompt
    )


    content = response.content


    # ========================================================
    # EXTRACT RESPONSE TEXT
    # ========================================================

    if isinstance(content, str):

        answer = content


    elif isinstance(content, list):

        answer = "\n".join(

            item.get("text", "")

            for item in content

            if (

                isinstance(item, dict)

                and item.get("type") == "text"

            )

        )


    else:

        answer = str(content)


    # ========================================================
    # SAVE CONVERSATION
    # ========================================================

    st.session_state.conversation_history.append({

        "role": "user",

        "content": question

    })


    st.session_state.conversation_history.append({

        "role": "assistant",

        "content": answer

    })


    # ========================================================
    # SOURCES
    # ========================================================

    not_found_message = (
        "I couldn't find this information "
        "in the provided documents."
    )


    if not_found_message in answer:

        sources = []


    else:

        sources = []

        for meta in relevant_metas:

            source = meta.get(
                "source",
                "Unknown"
            )

            if source not in sources:

                sources.append(source)


    return answer, sources


# ============================================================
# DISPLAY EXISTING CHAT
# ============================================================

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )


        # Show sources for assistant messages

        if (

            message["role"] == "assistant"

            and message.get("sources")

        ):

            with st.expander(
                "📚 Sources"
            ):

                for source in message["sources"]:

                    st.write(
                        f"- {source}"
                    )


# ============================================================
# CHAT INPUT
# ============================================================

question = st.chat_input(
    "Ask a question about the NLP knowledge base..."
)


if question:

    # ========================================================
    # TEMPORARY DEBUG INFORMATION
    # ========================================================

    st.sidebar.write(
        "DEBUG - Last retrieved docs:",
        len(st.session_state.last_retrieved_docs)
    )

    st.sidebar.write(
        "DEBUG - Conversation history:",
        len(st.session_state.conversation_history)
    )

    st.sidebar.write(
        "DEBUG - Follow-up:",
        is_follow_up_question(question)
    )


    # --------------------------------------------------------
    # Display user question
    # --------------------------------------------------------

    st.session_state.messages.append({

        "role": "user",

        "content": question

    })


    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # --------------------------------------------------------
    # Generate response
    # --------------------------------------------------------

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Thinking..."
        ):

            answer, sources = ask_rag(
                question
            )


        st.markdown(
            answer
        )


        # ----------------------------------------------------
        # Display sources
        # ----------------------------------------------------

        if sources:

            with st.expander(
                "📚 Sources"
            ):

                for source in sources:

                    st.write(
                        f"- {source}"
                    )


    # --------------------------------------------------------
    # Save assistant response
    # --------------------------------------------------------

    st.session_state.messages.append({

        "role": "assistant",

        "content": answer,

        "sources": sources

    })


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.header(
        "Chat Controls"
    )


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


    st.subheader(
        "Knowledge Base"
    )


    st.write(
        "NLP Units 1–2"
    )


    st.write(
        f"Stored chunks: {collection.count()}"
    )


    st.divider()


    st.caption(
        "AI Retrieval-Augmented Generation "
        "(RAG) Chatbot"
    )
