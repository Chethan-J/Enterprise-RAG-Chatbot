import os
import re
from pathlib import Path

import streamlit as st
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI


# ---------------------------------------------------------
# STREAMLIT PAGE SETTINGS
# ---------------------------------------------------------

st.set_page_config(
    page_title="NLP Knowledge Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("NLP Knowledge Chatbot")
st.caption("AI-powered document question-answering system")


# ---------------------------------------------------------
# PROJECT PATHS
# ---------------------------------------------------------

BASE_DIR = Path(__file__).parent

PDF_FILES = [
    BASE_DIR / "NLP_Units1-2_Combined.pdf",
    BASE_DIR / "NLP_Unit2_Combined.pdf"
]

CHROMA_PATH = BASE_DIR / "chroma_db"


# ---------------------------------------------------------
# ABBREVIATION MAP
# ---------------------------------------------------------

abbreviation_map = {
    "vsm": "vector space model",
    "tf-idf": "term frequency inverse document frequency",
    "tfidf": "term frequency inverse document frequency",
    "nlp": "natural language processing"
}


def expand_abbreviations(question):

    expanded_question = question

    for abbreviation, meaning in abbreviation_map.items():

        pattern = r"\b" + re.escape(abbreviation) + r"\b"

        expanded_question = re.sub(
            pattern,
            f"{abbreviation} ({meaning})",
            expanded_question,
            flags=re.IGNORECASE
        )

    return expanded_question


# ---------------------------------------------------------
# LOAD MODEL, CHROMADB AND GEMINI
# ---------------------------------------------------------

@st.cache_resource
def load_resources():

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = client.get_or_create_collection(
        name="nlp_combined"
    )

    # -----------------------------------------------------
    # CREATE KNOWLEDGE BASE IF EMPTY
    # -----------------------------------------------------

    if collection.count() == 0:

        all_chunks = []
        all_metadatas = []

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

        for pdf_file in PDF_FILES:

            reader = PdfReader(
                str(pdf_file)
            )

            text = ""

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:

                    text += " " + page_text

            text = re.sub(
                r"\s+",
                " ",
                text
            ).strip()

            chunks = text_splitter.split_text(
                text
            )

            for i, chunk in enumerate(chunks):

                all_chunks.append(chunk)

                all_metadatas.append({
                    "source": pdf_file.name,
                    "chunk_index": i
                })

        embeddings = model.encode(
            all_chunks,
            show_progress_bar=False
        )

        ids = [
            f"chunk_{i}"
            for i in range(len(all_chunks))
        ]

        collection.upsert(
            ids=ids,
            documents=all_chunks,
            embeddings=embeddings.tolist(),
            metadatas=all_metadatas
        )

    # -----------------------------------------------------
    # GEMINI API KEY
    # -----------------------------------------------------

    api_key = os.environ.get(
        "GOOGLE_API_KEY"
    )

    if not api_key:

        raise ValueError(
            "GOOGLE_API_KEY is not configured."
        )

    # -----------------------------------------------------
    # GEMINI MODEL
    # -----------------------------------------------------

    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        temperature=0
    )

    return model, collection, llm


# ---------------------------------------------------------
# LOAD RESOURCES
# ---------------------------------------------------------

model, collection, llm = load_resources()


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


if "conversation_history" not in st.session_state:

    st.session_state.conversation_history = []


# ---------------------------------------------------------
# CLEAR CHAT
# ---------------------------------------------------------

if st.sidebar.button("Clear Chat"):

    st.session_state.messages = []

    st.session_state.conversation_history = []

    st.rerun()


# ---------------------------------------------------------
# RAG FUNCTION
# ---------------------------------------------------------

def ask_rag(question, k=3):

    conversation_history = (
        st.session_state.conversation_history
    )

    # -----------------------------------------------------
    # STEP 1: GET PREVIOUS CONVERSATION
    # -----------------------------------------------------

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in conversation_history[-6:]
    )


    # -----------------------------------------------------
    # STEP 2: FIND PREVIOUS USER QUESTION
    # -----------------------------------------------------

    previous_user_question = ""

    for message in reversed(
        conversation_history
    ):

        if message["role"] == "user":

            previous_user_question = message["content"]

            break


    # -----------------------------------------------------
    # STEP 3: DETECT FOLLOW-UP QUESTION
    # -----------------------------------------------------

    question_lower = question.lower().strip()

    follow_up_words = [
        "it",
        "this",
        "that",
        "these",
        "those",
        "them",
        "they",
        "why",
        "how",
        "explain",
        "simple",
        "simply",
        "more",
        "useful",
        "use",
        "importance"
    ]

    question_words = question_lower.split()

    is_short_question = (
        len(question_words) <= 8
    )

    contains_follow_up_word = any(
        word in question_words
        for word in follow_up_words
    )

    is_follow_up = (
        bool(previous_user_question)
        and (
            is_short_question
            or contains_follow_up_word
        )
    )


    # -----------------------------------------------------
    # STEP 4: CREATE RETRIEVAL QUESTION
    # -----------------------------------------------------

    if is_follow_up:

        standalone_question = (
            previous_user_question
            + " "
            + question
        )

    else:

        standalone_question = question


    # -----------------------------------------------------
    # STEP 5: EXPAND ABBREVIATIONS
    # -----------------------------------------------------

    expanded_question = expand_abbreviations(
        standalone_question
    )


    # -----------------------------------------------------
    # STEP 6: CREATE QUESTION EMBEDDING
    # -----------------------------------------------------

    query_embedding = model.encode(
        [expanded_question]
    ).tolist()


    # -----------------------------------------------------
    # STEP 7: SEARCH CHROMADB
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # STEP 8: FILTER RELEVANT DOCUMENTS
    # -----------------------------------------------------

    threshold = 1.1

    relevant_docs = []

    relevant_metas = []

    for doc, meta, distance in zip(
        docs,
        metas,
        distances
    ):

        if distance <= threshold:

            relevant_docs.append(
                doc
            )

            relevant_metas.append(
                meta
            )


    # -----------------------------------------------------
    # STEP 9: NO RELEVANT INFORMATION
    # -----------------------------------------------------

    if not relevant_docs:

        answer = (
            "I couldn't find this information "
            "in the provided documents."
        )

        st.session_state.conversation_history.append({
            "role": "user",
            "content": question
        })

        st.session_state.conversation_history.append({
            "role": "assistant",
            "content": answer
        })

        return answer, []


    # -----------------------------------------------------
    # STEP 10: PREPARE DOCUMENT CONTEXT
    # -----------------------------------------------------

    context = "\n\n".join(
        f"Source: {meta.get('source', 'Unknown')}\n{doc}"
        for doc, meta in zip(
            relevant_docs,
            relevant_metas
        )
    )


    # -----------------------------------------------------
    # STEP 11: CREATE FINAL ANSWER PROMPT
    # -----------------------------------------------------

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

- Understand follow-up questions using the previous conversation.
- If the student says "it", "this", "that", "why",
  "explain more", "explain simply", or similar,
  understand what topic they are referring to.
- Answer the current question directly.
- Do NOT start the answer with "Based on the provided documents".
- Do NOT copy sentences directly from the documents.
- Explain the information in your own simple words.
- Break difficult technical ideas into easy language.
- Start with a short direct answer.
- Then explain it clearly.
- Use a simple example when the example is supported by the context.
- Do not add outside information.
- Keep the answer concise and useful for a college student.
- If the required information is not available in the context,
  say exactly:

"I couldn't find this information in the provided documents."

- Do not mention that you are an AI model.

Answer:
"""


    # -----------------------------------------------------
    # STEP 12: GENERATE ANSWER
    # -----------------------------------------------------

    response = llm.invoke(
        prompt
    )

    content = response.content


    # -----------------------------------------------------
    # STEP 13: EXTRACT RESPONSE TEXT
    # -----------------------------------------------------

    if isinstance(
        content,
        str
    ):

        answer = content

    elif isinstance(
        content,
        list
    ):

        answer = "\n".join(
            item.get(
                "text",
                ""
            )
            for item in content
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
            )
        )

    else:

        answer = str(
            content
        )


    # -----------------------------------------------------
    # STEP 14: SAVE CONVERSATION
    # -----------------------------------------------------

    st.session_state.conversation_history.append({
        "role": "user",
        "content": question
    })

    st.session_state.conversation_history.append({
        "role": "assistant",
        "content": answer
    })


    # -----------------------------------------------------
    # STEP 15: PREPARE SOURCES
    # -----------------------------------------------------

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

                sources.append(
                    source
                )


    # -----------------------------------------------------
    # STEP 16: RETURN RESULT
    # -----------------------------------------------------

    return answer, sources


# ---------------------------------------------------------
# DISPLAY PREVIOUS MESSAGES
# ---------------------------------------------------------

for message in st.session_state.messages:

    with st.chat_message(
        message["role"]
    ):

        st.markdown(
            message["content"]
        )

        if message.get("sources"):

            with st.expander(
                "View sources"
            ):

                for source in message["sources"]:

                    st.write(
                        "📄",
                        source
                    )


# ---------------------------------------------------------
# CHAT INPUT
# ---------------------------------------------------------

question = st.chat_input(
    "Ask a question about NLP..."
)


# ---------------------------------------------------------
# PROCESS QUESTION
# ---------------------------------------------------------

if question:

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

    with st.chat_message(
        "assistant"
    ):

        with st.spinner(
            "Searching documents..."
        ):

            try:

                answer, sources = ask_rag(
                    question
                )

                st.markdown(
                    answer
                )

                if sources:

                    with st.expander(
                        "View sources"
                    ):

                        for source in sources:

                            st.write(
                                "📄",
                                source
                            )

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": answer,

                    "sources": sources

                })

            except Exception as e:

                st.error(
                    f"Error: {e}"
                )
