import os
import re
from pathlib import Path

import streamlit as st
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI


# =========================================================
# STREAMLIT SETTINGS
# =========================================================

st.set_page_config(
    page_title="NLP Knowledge Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("NLP Knowledge Chatbot")
st.caption("AI-powered document question-answering system")


# =========================================================
# PROJECT PATHS
# =========================================================

BASE_DIR = Path(__file__).parent

PDF_FILES = [
    BASE_DIR / "NLP_Units1-2_Combined.pdf",
    BASE_DIR / "NLP_Unit2_Combined.pdf"
]

CHROMA_PATH = BASE_DIR / "chroma_db"


# =========================================================
# ABBREVIATION MAP
# =========================================================

abbreviation_map = {
    "vsm": "vector space model",
    "tf-idf": "term frequency inverse document frequency",
    "tfidf": "term frequency inverse document frequency",
    "nlp": "natural language processing"
}


# =========================================================
# COMMON TYPO MAP
# =========================================================

typo_map = {
    "td-idf": "tf-idf",
    "tfidf": "tf-idf",
    "vsm": "vsm",
    "nlp": "nlp"
}


def correct_common_terms(question):

    corrected_question = question

    for wrong, correct in typo_map.items():

        if wrong == correct:
            continue

        pattern = r"\b" + re.escape(wrong) + r"\b"

        corrected_question = re.sub(
            pattern,
            correct,
            corrected_question,
            flags=re.IGNORECASE
        )

    return corrected_question


# =========================================================
# ABBREVIATION EXPANSION
# =========================================================

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


# =========================================================
# LOAD RESOURCES
# =========================================================

@st.cache_resource
def load_resources():

    # -----------------------------------------------------
    # LOAD EMBEDDING MODEL
    # -----------------------------------------------------

    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )


    # -----------------------------------------------------
    # CONNECT TO CHROMADB
    # -----------------------------------------------------

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


            # Clean extracted text
            text = re.sub(
                r"\s+",
                " ",
                text
            ).strip()


            # Split document into chunks
            chunks = text_splitter.split_text(
                text
            )


            for i, chunk in enumerate(chunks):

                all_chunks.append(
                    chunk
                )

                all_metadatas.append({
                    "source": pdf_file.name,
                    "chunk_index": i
                })


        # -------------------------------------------------
        # CREATE EMBEDDINGS
        # -------------------------------------------------

        embeddings = model.encode(
            all_chunks,
            show_progress_bar=False
        )


        # -------------------------------------------------
        # CREATE IDS
        # -------------------------------------------------

        ids = [
            f"chunk_{i}"
            for i in range(
                len(all_chunks)
            )
        ]


        # -------------------------------------------------
        # STORE IN CHROMADB
        # -------------------------------------------------

        collection.upsert(
            ids=ids,
            documents=all_chunks,
            embeddings=embeddings.tolist(),
            metadatas=all_metadatas
        )


    # =====================================================
    # GEMINI API
    # =====================================================

    api_key = os.environ.get(
        "GOOGLE_API_KEY"
    )


    if not api_key:

        raise ValueError(
            "GOOGLE_API_KEY is not configured."
        )


    llm = ChatGoogleGenerativeAI(
        model="gemini-3.5-flash-lite",
        google_api_key=api_key,
        temperature=0
    )


    return model, collection, llm


# =========================================================
# LOAD RESOURCES
# =========================================================

model, collection, llm = load_resources()


# =========================================================
# SESSION STATE
# =========================================================

if "messages" not in st.session_state:

    st.session_state.messages = []


if "conversation_history" not in st.session_state:

    st.session_state.conversation_history = []


# =========================================================
# CLEAR CHAT
# =========================================================

if st.sidebar.button("Clear Chat"):

    st.session_state.messages = []

    st.session_state.conversation_history = []

    st.rerun()


# =========================================================
# RAG FUNCTION
# =========================================================

def ask_rag(question, k=3):

    conversation_history = (
        st.session_state.conversation_history
    )


    # =====================================================
    # STEP 1: CORRECT COMMON TERMS
    # =====================================================

    corrected_question = correct_common_terms(
        question
    )


    # =====================================================
    # STEP 2: GET RECENT CONVERSATION
    # =====================================================

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in conversation_history[-6:]
    )


    # =====================================================
    # STEP 3: FIND PREVIOUS USER QUESTION
    # =====================================================

    previous_user_question = ""

    for message in reversed(
        conversation_history
    ):

        if message["role"] == "user":

            previous_user_question = (
                message["content"]
            )

            break


    # =====================================================
    # STEP 4: DETECT FOLLOW-UP QUESTION
    # =====================================================

    question_lower = corrected_question.lower().strip()

    question_words = question_lower.split()


    follow_up_phrases = [
        "in simple",
        "in simple terms",
        "explain it",
        "explain this",
        "explain that",
        "explain simply",
        "explain more",
        "more simply",
        "why is it useful",
        "why is this useful",
        "why is it used",
        "why is this used",
        "what is its use",
        "what is its importance",
        "how is it useful",
        "how is this useful"
    ]


    contains_follow_up_phrase = any(
        phrase in question_lower
        for phrase in follow_up_phrases
    )


    short_follow_up = (
        len(question_words) <= 8
    )


    pronoun_follow_up = any(
        word in question_words
        for word in [
            "it",
            "this",
            "that",
            "they",
            "them",
            "these",
            "those"
        ]
    )


    is_follow_up = (
        bool(previous_user_question)
        and (
            contains_follow_up_phrase
            or short_follow_up
            or pronoun_follow_up
        )
    )


    # =====================================================
    # STEP 5: CREATE RETRIEVAL QUESTION
    # =====================================================

    if is_follow_up:

        retrieval_question = (
            previous_user_question
            + " "
            + corrected_question
        )

    else:

        retrieval_question = corrected_question


    # =====================================================
    # STEP 6: EXPAND ABBREVIATIONS
    # =====================================================

    expanded_retrieval_question = (
        expand_abbreviations(
            retrieval_question
        )
    )


    # =====================================================
    # STEP 7: CREATE EMBEDDING
    # =====================================================

    query_embedding = model.encode(
        [
            expanded_retrieval_question
        ]
    ).tolist()


    # =====================================================
    # STEP 8: SEARCH CHROMADB
    # =====================================================

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


    # =====================================================
    # STEP 9: FILTER RELEVANT RESULTS
    # =====================================================

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


    # =====================================================
    # STEP 10: NO RELEVANT INFORMATION
    # =====================================================

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


    # =====================================================
    # STEP 11: PREPARE DOCUMENT CONTEXT
    # =====================================================

    context_parts = []


    for doc, meta in zip(
        relevant_docs,
        relevant_metas
    ):

        source = meta.get(
            "source",
            "Unknown"
        )


        context_parts.append(
            f"Source: {source}\n{doc}"
        )


    context = "\n\n".join(
        context_parts
    )


    # =====================================================
    # STEP 12: FINAL ANSWER PROMPT
    # =====================================================

    prompt = f"""
You are a friendly NLP tutor helping a college student.

Your task is to answer the student's question using ONLY
the information available in the document context.

Previous conversation:

{history_text}

Current student question:

{corrected_question}

Document context:

{context}


IMPORTANT ANSWERING RULES:

1. Answer the student's current question directly.

2. If the student asks for a simple explanation, explain the
   concept in very easy language suitable for a college student.

3. Do NOT copy sentences from the documents.

4. Do NOT start the answer with:
   "Based on the provided documents"

5. Do NOT use phrases such as:
   "According to the document"
   unless absolutely necessary.

6. Explain the concept in your own words.

7. If the concept has important technical terms, explain those
   terms briefly.

8. Start with one short and clear definition.

9. Then give a short simple explanation.

10. If appropriate, give a small example using ONLY information
    supported by the document context.

11. If the student asks "why is it used", explain the purpose
    or application using ONLY information from the context.

12. If the student asks "in simple", "explain simply",
    "explain more", or similar, simplify the explanation rather
    than repeating the document wording.

13. If the question is a follow-up question, understand what
    "it", "this", "that", "they", or "them" refers to from the
    previous conversation.

14. Do not add information that is not supported by the
    document context.

15. Keep the answer concise. Normally use 2 to 5 short
    paragraphs or bullet points.

16. Do not mention that you are an AI model.

17. If the required information is not available in the
    document context, answer EXACTLY:

I couldn't find this information in the provided documents.


Now answer the student's question.
"""


    # =====================================================
    # STEP 13: GENERATE ANSWER
    # =====================================================

    response = llm.invoke(
        prompt
    )


    content = response.content


    # =====================================================
    # STEP 14: EXTRACT RESPONSE
    # =====================================================

    if isinstance(
        content,
        str
    ):

        answer = content.strip()


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
        ).strip()


    else:

        answer = str(
            content
        ).strip()


    # =====================================================
    # STEP 15: SAVE CONVERSATION
    # =====================================================

    st.session_state.conversation_history.append({
        "role": "user",
        "content": question
    })


    st.session_state.conversation_history.append({
        "role": "assistant",
        "content": answer
    })


    # =====================================================
    # STEP 16: PREPARE SOURCES
    # =====================================================

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


    # =====================================================
    # STEP 17: RETURN ANSWER
    # =====================================================

    return answer, sources


# =========================================================
# DISPLAY PREVIOUS CHAT
# =========================================================

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


# =========================================================
# CHAT INPUT
# =========================================================

question = st.chat_input(
    "Ask a question about NLP..."
)


# =========================================================
# PROCESS QUESTION
# =========================================================

if question:

    # -----------------------------------------------------
    # SAVE USER MESSAGE
    # -----------------------------------------------------

    st.session_state.messages.append({
        "role": "user",
        "content": question
    })


    # -----------------------------------------------------
    # DISPLAY USER MESSAGE
    # -----------------------------------------------------

    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # -----------------------------------------------------
    # GENERATE ASSISTANT RESPONSE
    # -----------------------------------------------------

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


                # -------------------------------------------------
                # DISPLAY ANSWER
                # -------------------------------------------------

                st.markdown(
                    answer
                )


                # -------------------------------------------------
                # DISPLAY SOURCES
                # -------------------------------------------------

                if sources:

                    with st.expander(
                        "View sources"
                    ):

                        for source in sources:

                            st.write(
                                "📄",
                                source
                            )


                # -------------------------------------------------
                # SAVE ASSISTANT MESSAGE
                # -------------------------------------------------

                st.session_state.messages.append({

                    "role": "assistant",

                    "content": answer,

                    "sources": sources

                })


            except Exception as e:

                st.error(
                    f"Error: {e}"
                )
