import os
from pathlib import Path

import streamlit as st
import chromadb
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI


# --------------------------------------------------
# 1. Page Settings
# --------------------------------------------------

st.set_page_config(
    page_title="NLP Knowledge Chatbot",
    page_icon="🤖",
    layout="centered"
)

st.title("NLP Knowledge Chatbot")
st.caption("AI-powered document question-answering system")


# --------------------------------------------------
# 2. Project Paths
# --------------------------------------------------

BASE_DIR = Path(__file__).parent

PDF_FILES = [
    BASE_DIR / "NLP_Units1-2_Combined.pdf",
    BASE_DIR / "NLP_Unit2_Combined.pdf"
]

CHROMA_PATH = BASE_DIR / "chroma_db"


# --------------------------------------------------
# 3. Load Resources
# --------------------------------------------------

@st.cache_resource
def load_resources():

    # Load embedding model
    model = SentenceTransformer("all-MiniLM-L6-v2")

    # Create local ChromaDB
    client = chromadb.PersistentClient(
        path=str(CHROMA_PATH)
    )

    collection = client.get_or_create_collection(
        name="nlp_combined"
    )

    # --------------------------------------------------
    # Create database if it is empty
    # --------------------------------------------------

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

            reader = PdfReader(str(pdf_file))

            text = ""

            for page in reader.pages:

                page_text = page.extract_text()

                if page_text:
                    text += " " + page_text

            # Clean whitespace
            import re

            text = re.sub(
                r"\s+",
                " ",
                text
            ).strip()

            # Create chunks
            chunks = text_splitter.split_text(text)

            for i, chunk in enumerate(chunks):

                all_chunks.append(chunk)

                all_metadatas.append({
                    "source": pdf_file.name,
                    "chunk_index": i
                })

        # Create embeddings
        embeddings = model.encode(
            all_chunks,
            show_progress_bar=False
        )

        # Store in ChromaDB
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

    # --------------------------------------------------
    # Gemini API
    # --------------------------------------------------

    api_key = os.environ.get("GOOGLE_API_KEY")

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


# Load everything
model, collection, llm = load_resources()


# --------------------------------------------------
# 4. Chat History
# --------------------------------------------------

if "messages" not in st.session_state:
    st.session_state.messages = []


for message in st.session_state.messages:

    with st.chat_message(message["role"]):

        st.markdown(
            message["content"]
        )

        if message.get("sources"):

            with st.expander("View sources"):

                for source in message["sources"]:
                    st.write("📄", source)


# --------------------------------------------------
# 5. User Question
# --------------------------------------------------

question = st.chat_input(
    "Ask a question about NLP..."
)


if question:

    # Add user question
    st.session_state.messages.append({
        "role": "user",
        "content": question
    })

    with st.chat_message("user"):
        st.markdown(question)


    # --------------------------------------------------
    # 6. Generate Answer
    # --------------------------------------------------

    with st.chat_message("assistant"):

        with st.spinner(
            "Searching documents..."
        ):

            try:

                # Create question embedding
                question_embedding = model.encode(
                    question
                )


                # Search ChromaDB
                results = collection.query(
                    query_embeddings=[
                        question_embedding.tolist()
                    ],
                    n_results=3,
                    include=[
                        "documents",
                        "metadatas",
                        "distances"
                    ]
                )


                docs = results["documents"][0]
                metas = results["metadatas"][0]
                distances = results["distances"][0]


                # Retrieval threshold
                threshold = 1.2


                relevant_docs = []
                relevant_metas = []


                for doc, meta, distance in zip(
                    docs,
                    metas,
                    distances
                ):

                    if distance <= threshold:

                        relevant_docs.append(doc)
                        relevant_metas.append(meta)


                # --------------------------------------------------
                # 7. No Relevant Information
                # --------------------------------------------------

                if not relevant_docs:

                    answer = (
                        "I couldn't find this information "
                        "in the provided documents."
                    )

                    sources = []


                # --------------------------------------------------
                # 8. Generate Answer Using Gemini
                # --------------------------------------------------

                else:

                    context = "\n\n".join(
                        relevant_docs
                    )


                    prompt = f"""
You are an assistant answering questions
from NLP course documents.

Use ONLY the information provided in the context.

Do not use general knowledge.

If the answer is not available in the context, say:

"I couldn't find this information in the provided documents."

Context:

{context}

Question:

{question}

Answer:
"""


                    response = llm.invoke(
                        prompt
                    )


                    content = response.content


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
                            item.get("text", "")
                            for item in content
                            if isinstance(
                                item,
                                dict
                            )
                            and item.get("type") == "text"
                        )


                    else:

                        answer = str(
                            content
                        )


                    # --------------------------------------------------
                    # 9. Get Sources
                    # --------------------------------------------------

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


                # --------------------------------------------------
                # 10. Display Answer
                # --------------------------------------------------

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


                # Save assistant response
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": sources
                })


            except Exception as e:

                st.error(
                    f"Error: {e}"
                )
