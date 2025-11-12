# Facial Emotion Recognition & RAG-based Synthetic Review Pipeline 
 
 This project builds a facial emotion recognition and emotion-driven text review retrieval system using deep learning and RAG.

 
**Stages:**  
1️ Facial Expression Recognition (CNN + ResNet50)  
2️ Synthetic Review Generation, Embedding & RAG Integration  
3️ Design Thinking & Architecture Analysis  



## Project Overview
This project is divided into **Stage 1** and **Stage 2**:

### Stage 1 – Facial Expression Recognition
- Uses a **ResNet50** pretrained model to classify human facial expressions.
- Recognizes 7 emotions: `angry, disgust, fear, happy, neutral, sad, surprise`.
- Generates predictions for test images and saves them in a CSV.
- Outputs model, weights, confusion matrix, and sample predictions.

### Stage 2 – Synthetic Review Generation & RAG Pipeline
- Uses predicted emotions from Stage 1 to generate **templated synthetic reviews**.
- Embeds reviews using **SentenceTransformer** and stores them in a **FAISS** vector store.
- Implements **Retrieval-Augmented Generation (RAG)** with **LangChain** and summarization using **Flan-T5**.
- Performs sentiment analysis with **DistilBERT** and saves all outputs.

---

## Folder Structure
```text
project_root/
│
├─ train/                  # Training images (organized by emotion classes)
├─ test/                   # Test images (organized by emotion classes)
├─ outputs/                # Stage 1 outputs (models, predictions, plots)
├─ outputs_stage2/         # Stage 2 outputs (reviews, FAISS index, summaries)
├─ Stage1_facial_expression_recognition_resnet.py
├─ Stage2.py
├─ requirements.txt
└─ README.md


```
# Setup Instructions

### 1. Clone the Repository
```bash
git clone https://github.com/ya-sonia/Emotion_Detection.git
cd Facial_Recognition_Project
```

### 2. Create & Activate Conda Environment
```bash
conda create -n facial_exp python=3.10 -y
conda activate facial_exp
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Run Stage 1 – Facial Expression Recognition
```bash
python Stage1_facial_expression_recognition_resnet.py
```

### 5. Run Stage 2 – Synthetic Review Generation & RAG
```bash
python Stage2.py
```

# Dataset

Dataset used: [FER-2013 Kaggle Dataset](https://www.kaggle.com/datasets/msambare/fer2013)
Training images are organized by emotion classes in train/.
Test images are organized by emotion classes in test/.



# Results
