import streamlit as st
import yt_dlp
from sentence_transformers import SentenceTransformer

from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter

from openai import OpenAI
import requests

from langchain.docstore.document import Document

from dotenv import load_dotenv
load_dotenv()
import time
import os

st.set_page_config(page_title="Chat with YouTube Video", layout="wide")
st.title("🎥 Chat with YouTube Video")

client = OpenAI(
    base_url="https://openrouter.ai/api/v1",
    api_key = os.getenv("OPENAI_API_KEY")
)

video_url = st.text_input("Paste a YouTube video link here..")

if video_url:
    st.video(video_url)
    #st.info("Fetching transcript...")

    try:
        ydl_opts = {
            'skip_download': True,
            'writesubtitles': True,
            'writeautomaticsub': True,
            'subtitleslangs': ['en'],
            'quiet': True
        }
        with yt_dlp.YoutubeDL({'skip_download': True, 'writesubtitles': True, 'writeautomaticsub': True, 'quiet': True}) as ydl:
            info = ydl.extract_info(video_url, download=False)
            subs = info.get("subtitles") or info.get("automatic_captions")
            en_key = None
            if subs:
                for key in subs.keys():
                    if key.startswith("en"):
                        en_key = key
                        break
            if not en_key:
                st.error("No English captions available for this video.")
                st.stop()

            en_subs = subs[en_key]
            sub_url = None
            for sub in en_subs:
                if sub['ext'] == 'json3':
                    sub_url = sub['url']
                    break
            if not sub_url:
                sub_url = en_subs[0]['url']

            # Fetch subtitle JSON
            res = requests.get(sub_url).json()

            # Extract text
            transcript = []
            for event in res.get("events", []):
                for seg in event.get("segs", []):
                    transcript.append(seg.get("utf8", ""))

            text = " ".join(transcript)
            
            #st.success("✅ Transcript fetched successfully!")

            splitter = RecursiveCharacterTextSplitter(chunk_size=1000,chunk_overlap=100)
            chunks = splitter.split_text(text)

            #st.info("Creating embeddings...")
            model = SentenceTransformer("all-MiniLM-L6-v2")
            embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
            vectorstore = FAISS.from_texts(chunks, embeddings)

            #st.success("Transcript indexed!")

            if "chat_history" not in st.session_state:
                st.session_state.chat_history = []
            
            question = st.text_input("Ask a question about the video", key="question_input")

            if question:
                query_emb = embeddings.embed_query(question)
                docs_and_scores = vectorstore.similarity_search_by_vector(query_emb, k=3)
                context = "\n\n".join([doc.page_content for doc in docs_and_scores])

                prompt = f"Answer the question based on the following transcript and do not mention the transcript:\n\n{context}\n\nQuestion: {question}"

                with st.spinner("Thinking..."):
                    completion = client.chat.completions.create(
                        model = "meta-llama/llama-3-8b-instruct",
                        messages = [
                            {"role": "system", "content": "You are a helpful AI that answers questions about YouTube videos."},
                            {"role": "user", "content": prompt}
                        ]
                    )
                answer = completion.choices[0].message.content
                st.session_state.chat_history.append((question,answer))
            
            for q,a in st.session_state.chat_history:
                with st.chat_message("user"):
                    st.markdown(f"{q}")

                # Bot message
                with st.chat_message("assistant"):
                    st.markdown(f"{a}")
                
    except Exception as e:
        st.error(f"Error fetching transcript: {e}")


