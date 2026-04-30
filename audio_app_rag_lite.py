import streamlit as st
import openai
from gtts import gTTS
import os
import time
import json
import hashlib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

# RAG Configuration using lightweight TF-IDF instead of sentence-transformers
class RAGConfig:
    def __init__(self):
        self.vectorizer_file = "tfidf_vectorizer.pkl"
        self.knowledge_base_file = "knowledge_base.json"
        self.embeddings_file = "embeddings.npy"
        
class RAGEngine:
    def __init__(self, config: RAGConfig):
        self.config = config
        self.vectorizer = None
        self.knowledge_texts = []
        self.embeddings = None
        self.load_or_create_index()
        
    def load_or_create_index(self):
        """Load existing index or create new one"""
        if os.path.exists(self.config.knowledge_base_file) and os.path.exists(self.config.embeddings_file):
            with open(self.config.knowledge_base_file, 'r') as f:
                self.knowledge_texts = json.load(f)
            self.embeddings = np.load(self.config.embeddings_file)
            self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
            self.vectorizer.fit(self.knowledge_texts)
        else:
            self.create_sample_knowledge_base()
            
    def create_sample_knowledge_base(self):
        """Create a sample knowledge base for demonstration"""
        sample_knowledge = [
            "The weather today is sunny with a temperature of 72°F.",
            "Python is a high-level programming language created by Guido van Rossum.",
            "Machine learning is a subset of artificial intelligence that focuses on neural networks.",
            "Streamlit is an open-source Python framework for creating web applications.",
            "OpenAI's GPT models are large language models trained on vast amounts of text data.",
            "Whisper is an automatic speech recognition system developed by OpenAI.",
            "gTTS (Google Text-to-Speech) is a Python library for converting text to speech.",
            "TF-IDF stands for Term Frequency-Inverse Document Frequency.",
            "Vector embeddings represent text as numerical vectors for semantic search.",
            "RAG (Retrieval-Augmented Generation) combines retrieval systems with language models."
        ]
        
        # Create TF-IDF embeddings
        self.vectorizer = TfidfVectorizer(max_features=1000, stop_words='english')
        self.embeddings = self.vectorizer.fit_transform(sample_knowledge).toarray()
        
        # Store knowledge base
        self.knowledge_texts = sample_knowledge
        
        # Save to disk
        with open(self.config.knowledge_base_file, 'w') as f:
            json.dump(self.knowledge_texts, f)
        np.save(self.config.embeddings_file, self.embeddings)
            
    def add_knowledge(self, texts: list):
        """Add new knowledge to the vector database"""
        if not texts:
            return
            
        # Add to knowledge base
        self.knowledge_texts.extend(texts)
        
        # Recreate embeddings with all texts
        self.embeddings = self.vectorizer.fit_transform(self.knowledge_texts).toarray()
        
        # Save updated knowledge base and embeddings
        with open(self.config.knowledge_base_file, 'w') as f:
            json.dump(self.knowledge_texts, f)
        np.save(self.config.embeddings_file, self.embeddings)
            
    def retrieve_relevant_context(self, query: str, k: int = 3) -> list:
        """Retrieve most relevant context for the query"""
        if len(self.knowledge_texts) == 0:
            return []
            
        # Transform query using the same vectorizer
        query_embedding = self.vectorizer.transform([query]).toarray()
        
        # Calculate cosine similarity
        similarities = cosine_similarity(query_embedding, self.embeddings)[0]
        
        # Get top-k most similar indices
        top_indices = np.argsort(similarities)[-k:][::-1]
        
        # Return the top-k most relevant texts
        relevant_contexts = []
        for idx in top_indices:
            if similarities[idx] > 0.1:  # Threshold for relevance
                relevant_contexts.append(self.knowledge_texts[idx])
                
        return relevant_contexts

# 1. SETUP: Configuration & API Keys
openai.api_key = os.getenv("OPENAI_API_KEY")

st.set_page_config(page_title="RAG Audio Bot 2026 (Lite)", layout="wide")
st.title("🎙️ RAG-Powered Conversational Audio Bot (Lite)")
st.caption("Architecture: Chained Pipeline with RAG (Whisper + RAG + GPT-4o + gTTS)")

# Initialize RAG engine
rag_config = RAGConfig()
rag_engine = RAGEngine(rag_config)

# 2. SIDEBAR: TPM/PM Control Panel
with st.sidebar:
    st.header("⚙️ System Configuration")
    model_choice = st.selectbox("LLM Brain", ["gpt-4o", "gpt-4o-mini"])
    temperature = st.slider("Grounding (Temperature)", 0.0, 1.0, 0.3)
    
    # RAG Configuration
    st.divider()
    st.header("🔍 RAG Configuration")
    retrieval_k = st.slider("Context Chunks", 1, 5, 3)
    use_rag = st.checkbox("Enable RAG", value=True)
    
    # Knowledge Base Management
    st.divider()
    st.header("📚 Knowledge Base")
    with st.expander("Add Knowledge"):
        new_knowledge = st.text_area("Enter knowledge (one fact per line):")
        if st.button("Add to Knowledge Base"):
            if new_knowledge:
                facts = [fact.strip() for fact in new_knowledge.split('\n') if fact.strip()]
                rag_engine.add_knowledge(facts)
                st.success(f"Added {len(facts)} facts to knowledge base!")
    
    st.divider()
    st.header("📊 2026 Performance Metrics")
    st.info("Target TTFA: < 1.5s (RAG-enhanced Pipeline)")
    st.metric("Knowledge Base Size", f"{len(rag_engine.knowledge_texts)} facts")
    st.metric("Embedding Model", "TF-IDF (Lightweight)")

# 3. AUDIO INPUT: The "Ears" (ASR Stage)
audio_value = st.audio_input("Speak to the AI Assistant")

if audio_value:
    # A. ASR Stage: Whisper Transcription
    start_time = time.time()
    with st.status("👂 Listening and Transcribing...", expanded=True):
        # Save temporary file for transcription
        with open("temp_input.wav", "wb") as f:
            f.write(audio_value.read())
        
        # Call OpenAI Whisper API
        transcript_response = openai.audio.transcriptions.create(
            model="whisper-1", 
            file=open("temp_input.wav", "rb")
        )
        user_text = transcript_response.text
        st.write(f"**User said:** {user_text}")

    # B. RAG Stage: Context Retrieval
    retrieved_context = []
    if use_rag:
        with st.status("🔍 Retrieving Relevant Context...", expanded=True):
            retrieved_context = rag_engine.retrieve_relevant_context(user_text, k=retrieval_k)
            if retrieved_context:
                st.write("**Retrieved Context:**")
                for i, context in enumerate(retrieved_context, 1):
                    st.write(f"{i}. {context}")
            else:
                st.write("No relevant context found.")

    # C. LLM Stage: GPT-4o Reasoning with RAG
    with st.status("🧠 Reasoning with RAG...", expanded=True):
        # Build system prompt with retrieved context
        system_prompt = "You are a professional assistant. Be concise and helpful."
        if retrieved_context:
            context_text = "\n".join([f"- {ctx}" for ctx in retrieved_context])
            system_prompt += f"\n\nUse the following relevant context to inform your response:\n{context_text}\n\nIf the context doesn't help answer the question, respond based on your general knowledge."
        
        response = openai.chat.completions.create(
            model=model_choice,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_text}
            ],
            temperature=temperature
        )
        ai_text = response.choices[-1].message.content
        st.write(f"**AI Response:** {ai_text}")

    # D. TTS Stage: gTTS Vocalization
    with st.status("🗣️ Synthesizing Voice...", expanded=True):
        tts = gTTS(text=ai_text, lang='en')
        tts.save("ai_response.mp3")
        
        # Calculate TTFA (Time to First Audio)
        ttfa = round(time.time() - start_time, 2)
        st.audio("ai_response.mp3", autoplay=True)

    # 4. FINAL PERFORMANCE REPORT
    st.divider()
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("Measured TTFA", f"{ttfa}s")
    col2.metric("Architecture", "RAG-Chained")
    col3.metric("Context Chunks", len(retrieved_context))
    col4.metric("RAG Enabled", "Yes" if use_rag else "No")
    
    if ttfa > 1.5:
        st.warning("⚠️ High Latency detected. RAG processing adds overhead.")

# 5. KNOWLEDGE BASE VIEWER (Bottom Section)
st.divider()
st.header("📚 Current Knowledge Base")
if st.button("Refresh Knowledge Base"):
    rag_engine.load_or_create_index()
    st.rerun()

# Display knowledge base
if rag_engine.knowledge_texts:
    st.write(f"**Total Facts:** {len(rag_engine.knowledge_texts)}")
    with st.expander("View All Knowledge"):
        for i, fact in enumerate(rag_engine.knowledge_texts, 1):
            st.write(f"{i}. {fact}")
else:
    st.info("No knowledge base loaded. Add some facts using the sidebar.")
