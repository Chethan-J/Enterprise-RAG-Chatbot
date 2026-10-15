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

    # Load embedding model
    model = SentenceTransformer(
        "all-MiniLM-L6-v2"
    )

    # Connect to ChromaDB
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

                all_chunks.append(chunk)

                all_metadatas.append({
                    "source": pdf_file.name,
                    "chunk_index": i
                })

        # Generate embeddings
        embeddings = model.encode(
            all_chunks,
            show_progress_bar=False
        )

        ids = [
            f"chunk_{i}"
            for i in range(len(all_chunks))
        ]

        # Store in ChromaDB
        collection.upsert(
            ids=ids,
            documents=all_chunks,
            embeddings=embeddings.tolist(),
            metadatas=all_metadatas
        )

    # -----------------------------------------------------
    # LOAD GEMINI API KEY
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


# Load resources
model, collection, llm = load_resources()


# ---------------------------------------------------------
# SESSION STATE
# ---------------------------------------------------------

if "messages" not in st.session_state:

    st.session_state.messages = []


if "conversation_history" not in st.session_state:

    st.session_state.conversation_history = []


# ---------------------------------------------------------
# CLEAR CHAT BUTTON
# ---------------------------------------------------------

if st.sidebar.button("Clear Chat"):

    st.session_state.messages = []

    st.session_state.conversation_history = []

    st.rerun()


# ---------------------------------------------------------
# RAG FUNCTION
# ---------------------------------------------------------

def ask_rag(question, k=3):

    # -----------------------------------------------------
    # STEP 1: GET PREVIOUS CONVERSATION
    # -----------------------------------------------------

    conversation_history = (
        st.session_state.conversation_history
    )

    history_text = "\n".join(
        f"{message['role']}: {message['content']}"
        for message in conversation_history[-6:]
    )


    # -----------------------------------------------------
    # STEP 2: REWRITE FOLLOW-UP QUESTION
    # -----------------------------------------------------

    if conversation_history:

        rewrite_prompt = f"""
You are helping an NLP student.

Previous conversation:

{history_text}

Current student question:

{question}

Rewrite the current question as a complete standalone question
that can be used to search an NLP knowledge base.

If the current question already makes sense by itself,
return it unchanged.

If the question uses words such as:
"it", "this", "that", "they", "them",
"explain more", "explain it", "why is it useful",
understand what the student is referring to
from the previous conversation.

Do not answer the question.

Only return the rewritten question.
"""

        rewrite_response = llm.invoke(
            rewrite_prompt
        )

        rewrite_content = rewrite_response.content

        if isinstance(
            rewrite_content,
            str
        ):

            standalone_question = rewrite_content

        elif isinstance(
            rewrite_content,
            list
        ):

            standalone_question = "\n".join(
                item.get(
                    "text",
                    ""
                )
                for item in rewrite_content
                if (
                    isinstance(item, dict)
                    and item.get("type") == "text"
                )
            )

        else:

            standalone_question = str(
                rewrite_content
            )

    else:

        standalone_question = question


    # -----------------------------------------------------
    # STEP 3: EXPAND ABBREVIATIONS
    # -----------------------------------------------------

    expanded_question = expand_abbreviations(
        standalone_question
    )


    # -----------------------------------------------------
    # STEP 4: CREATE QUESTION EMBEDDING
    # -----------------------------------------------------

    query_embedding = model.encode(
        [expanded_question]
    ).tolist()


    # -----------------------------------------------------
    # STEP 5: SEARCH CHROMADB
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
    # STEP 6: FILTER RELEVANT DOCUMENTS
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
    # STEP 7: NO RELEVANT INFORMATION
    # -----------------------------------------------------

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


    # -----------------------------------------------------
    # STEP 8: PREPARE DOCUMENT CONTEXT
    # -----------------------------------------------------

    context = "\n\n".join(
        f"Source: {meta.get('source', 'Unknown')}\n{doc}"
        for doc, meta in zip(
            relevant_docs,
            relevant_metas
        )
    )


    # -----------------------------------------------------
    # STEP 9: CREATE FINAL ANSWER PROMPT
    # -----------------------------------------------------

    prompt = f"""
You are a friendly NLP tutor helping a college student.

Use ONLY the information in the provided document context.

Previous conversation:

{history_text}

Current student question:

{question}

Standalone version of the question:

{standalone_question}

Document context:

{context}

IMPORTANT:

- Understand follow-up questions using the previous conversation.
- Words such as "it", "this", "that", or "why"
  may refer to something discussed previously.
- Answer the student's current question directly.
- Do NOT copy sentences from the document.
- Explain the information in your own simple words.
- Break difficult technical ideas into easy language.
- Start with a short, direct answer.
- Then explain it simply.
- Use an example or analogy only when supported by the context.
- Do not add outside information.
- Keep the answer concise and useful for a student.
- If the required information is not available in the documents,
  say:

"I couldn't find this information in the provided documents."

- Do not mention that you are an AI model.

Answer:
"""


    # -----------------------------------------------------
    # STEP 10: GENERATE ANSWER
    # -----------------------------------------------------

    response = llm.invoke(
        prompt
    )

    content = response.content


    # -----------------------------------------------------
    # STEP 11: EXTRACT RESPONSE TEXT
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
    # STEP 12: SAVE CONVERSATION
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
    # STEP 13: PREPARE SOURCES
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
    # STEP 14: RETURN ANSWER AND SOURCES
    # -----------------------------------------------------

    return answer, sources


# ---------------------------------------------------------
# DISPLAY PREVIOUS CHAT MESSAGES
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
# PROCESS NEW QUESTION
# ---------------------------------------------------------

if question:

    # Save user message
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })


    # Display user message
    with st.chat_message(
        "user"
    ):

        st.markdown(
            question
        )


    # Generate assistant response
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


                # Display sources
                if sources:

                    with st.expander(
                        "View sources"
                    ):

                        for source in sources:

                            st.write(
                                "📄",
                                source
                            )


                # Save assistant message
                st.session_state.messages.append({

                    "role": "assistant",

                    "content": answer,

                    "sources": sources

                })


            except Exception as e:

                st.error(
                    f"Error: {e}"
                )
