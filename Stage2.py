"""
Stage 2: Synthetic Review Generation, Sentiment Analysis & RAG Integration
--------------------------------------------------------------------------
This script builds a Retrieval-Augmented Generation (RAG) pipeline that:
1. Uses predicted emotions from Stage 1 (predictions.csv)
2. Generates templated synthetic reviews
3. Embeds them using SentenceTransformer
4. Stores them in a FAISS vector index
5. Runs queries using LangChain RetrievalQA + HuggingFace summarizer
6. Performs sentiment analysis on retrieved results
7. Saves all intermediate and final outputs


"""

import os
import json
import random
import pandas as pd
import numpy as np
from tqdm import tqdm

from sentence_transformers import SentenceTransformer
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_core.documents import Document
#from langchain_community.retrievers import WikipediaRetriever
#from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate

from transformers import (
    pipeline,
    AutoTokenizer,
    AutoModelForSeq2SeqLM
)

from langchain_community.llms import HuggingFacePipeline
from transformers import pipeline as hf_pipeline





BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PRED_CSV = os.path.join(BASE_DIR, "outputs", "predictions.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs_stage2")
os.makedirs(OUTPUT_DIR, exist_ok=True)

assert os.path.exists(PRED_CSV), f"predictions.csv not found at {PRED_CSV}"



# Step 1. Load Predictions

print(f"Loading predictions from {PRED_CSV} ...")
pred_df = pd.read_csv(PRED_CSV)
print(pred_df.head())


# Step 2. Synthetic Review Generation

emotion_templates = {
    "angry": [
        "I felt frustrated after this experience. The service was poor and it made me upset.",
        "This upset me — too many mistakes and a rude interaction. Not recommended."
    ],
    "disgust": [
        "The place was dirty and off-putting. I would avoid it next time.",
        "An unpleasant experience — not acceptable and very disappointing."
    ],
    "fear": [
        "I felt uneasy and worried during this visit. It didn't feel safe.",
        "The experience made me anxious; I won't go back alone."
    ],
    "happy": [
        "I absolutely loved it! Friendly staff and an enjoyable time.",
        "Fantastic experience — exceeded my expectations and I'd come back."
    ],
    "neutral": [
        "It was okay — neither great nor terrible. A standard experience.",
        "Average service and experience; nothing noteworthy."
    ],
    "sad": [
        "I left feeling disappointed and down — the experience lacked warmth.",
        "It was a sad experience; it didn't meet my expectations."
    ],
    "surprise": [
        "I was pleasantly surprised by how good it was — unexpected delight!",
        "Unexpectedly great service — a nice surprise and worth recommending."
    ]
}

pred_df["Predicted_Emotion"] = pred_df["Predicted_Emotion"].str.lower().str.strip()

generated_rows = []
for _, row in pred_df.iterrows():
    emo = row["Predicted_Emotion"]
    if emo not in emotion_templates:
        emo = "neutral"
    for tpl in emotion_templates[emo]:
        generated_rows.append({
            "Image": row["Image"],
            "Predicted_Emotion": row["Predicted_Emotion"],
            "Review": tpl
        })

reviews_df = pd.DataFrame(generated_rows)
print(f"Generated {len(reviews_df)} synthetic reviews.")
gen_reviews_path = os.path.join(OUTPUT_DIR, "generated_reviews.csv")
reviews_df.to_csv(gen_reviews_path, index=False)
print(f"Saved generated reviews → {gen_reviews_path}")


# Step 3. Create Embeddings

EMBED_MODEL_NAME = "all-MiniLM-L6-v2"
print("Loading SentenceTransformer:", EMBED_MODEL_NAME)
sbert = SentenceTransformer(EMBED_MODEL_NAME)
texts = reviews_df["Review"].tolist()
embeddings = sbert.encode(texts, show_progress_bar=True, convert_to_numpy=True)


# Step 4. Build FAISS Vectorstore

documents = []
for _, row in reviews_df.iterrows():
    metadata = {"image": row["Image"], "emotion": row["Predicted_Emotion"]}
    documents.append(Document(page_content=row["Review"], metadata=metadata))

hf_emb = HuggingFaceEmbeddings(model_name=EMBED_MODEL_NAME)
faiss_store = FAISS.from_documents(documents, hf_emb)

faiss_path = os.path.join(OUTPUT_DIR, "faiss_reviews_index")
faiss_store.save_local(faiss_path)
print(f"Saved FAISS vectorstore → {faiss_path}")


# Step 5. Setup Summarization Model (Flan-T5)

SUM_MODEL = "google/flan-t5-small"
print("Loading summarization model:", SUM_MODEL)
tokenizer = AutoTokenizer.from_pretrained(SUM_MODEL)
model = AutoModelForSeq2SeqLM.from_pretrained(SUM_MODEL)

summarization_pipeline = pipeline(
    "text2text-generation",
    model=model,
    tokenizer=tokenizer,
    device=0 if __import__("torch").cuda.is_available() else -1,
    max_length=128,
    truncation=True
)
llm = HuggingFacePipeline(pipeline=summarization_pipeline)


# Step 6. Build LangChain RetrievalQA (RAG)

retriever = faiss_store.as_retriever(search_type="similarity", search_kwargs={"k": 4})

prompt_template = """
You are given a set of short user reviews about an experience, and metadata (image, emotion).
Summarize the main sentiment & top points from the retrieved reviews in 2–4 sentences.
Also include a short list of emotions present and counts.

Reviews:
{text}

Question:
{question}

Return the summary followed by a JSON object with keys:
- summary: <text>
- emotions_count: {{ emotion: count, ... }}
"""

PROMPT = PromptTemplate(
    template=prompt_template,
    input_variables=["text", "question"]
)

qa_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=retriever,   
    chain_type_kwargs={
        "prompt": PROMPT,                
        "document_variable_name": "text" 
    },
)


# Step 7. Sentiment Analysis

sentiment_model = hf_pipeline(
    "sentiment-analysis",
    model="distilbert-base-uncased-finetuned-sst-2-english",
    device=0 if __import__("torch").cuda.is_available() else -1
)


# Step 8. Define Query Function

def rag_query_and_sentiment(query_text, top_k=4):
    """Run RAG retrieval + summarization + sentiment analysis."""
    docs = retriever.get_relevant_documents(query_text)[:top_k]
    retrieved_texts = [d.page_content for d in docs]
    metadata = [d.metadata for d in docs]

    concatenated = "\n\n".join(retrieved_texts)
    
    
    qa_out = qa_chain.run({"query": query_text, "text": concatenated})

    prompt_filled = PROMPT.format(text=concatenated, question=query_text)
    summary_response = summarization_pipeline(prompt_filled, max_length=128, truncation=True)[0]["generated_text"]

    sentiments = sentiment_model(retrieved_texts)
    results = {
        "query": query_text,
        "summary": summary_response,
        "retrieved": [
            {"text": retrieved_texts[i], "metadata": metadata[i], "sentiment": sentiments[i]}
            for i in range(len(retrieved_texts))
        ]
    }
    return results

# Step 9. Example Queries

examples = [
    "Which reviews show positive experiences?",
    "Summarize customer feelings when they are surprised.",
    "Which images correspond to negative reviews and why?"
]

sample_results = []
for q in examples:
    print("\n" + "="*80)
    print("QUERY:", q)
    result = rag_query_and_sentiment(q)
    print("Summary:\n", result["summary"])
    print("\nTop Retrieved Docs and Sentiments:")
    for d in result["retrieved"]:
        print("-", d["metadata"], "| Sentiment:", d["sentiment"])
    sample_results.append({
        "query": q,
        "summary": result["summary"],
        "retrieved": json.dumps(result["retrieved"])
    })


# Step 10. Save Sample Outputs

rag_csv_path = os.path.join(OUTPUT_DIR, "rag_sample_outputs.csv")
pd.DataFrame(sample_results).to_csv(rag_csv_path, index=False)
print(f"Saved sample RAG outputs → {rag_csv_path}")


# Step 11. RAG Architecture Explanation

arch_expl = """
RAG architecture (this script):
1. Input: predicted emotions CSV -> generate templated reviews.
2. Embedding: SentenceTransformer (all-MiniLM-L6-v2) encodes review texts.
3. Vector DB: FAISS stores vectors + metadata (image, emotion).
4. Retriever: FAISS returns top-k similar reviews for a user query.
5. Generator/Summarizer: Local HuggingFace seq2seq model (flan-t5-small) summarizes retrieved reviews.
6. Sentiment: DistilBERT-based sentiment pipeline provides per-snippet sentiment.
7. LangChain ties retriever and LLM (HuggingFacePipeline) into a RetrievalQA chain.
"""
print(arch_expl)



print("\nAll files created in:", OUTPUT_DIR)

