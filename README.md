# PSIML Applied AI Tour — Notebooks

This repository contains a set of hands-on notebooks used in the **Applied AI** part of **PSIML**.  
Each notebook is an independent “tour” through a key AI area:

1. Google Colab basics  
2. Vision  
3. NLP  
4. Voice / Audio  

All notebooks are designed to be run on **Google Colab**.

---

## 📂 Repository Structure

```text
psiml-applied-ai/
│
├── notebooks/
│   ├── Psiml_Tour_Collab.ipynb
│   ├── PSIML_Tour_Vision.ipynb
│   ├── PSIML_Tour_NLP.ipynb
│   ├── PSIML_Tour_Voice.ipynb
│
└── README.md
```

## 🔵 1. Google Colab Intro  
**File:** [`notebooks/Psiml_Tour_Collab.ipynb`](notebooks/Psiml_Tour_Collab.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pfe-rs/psiml-applied-ai/blob/master/notebooks/Psiml_Tour_Collab.ipynb)


This notebook gives a quick introduction to **Google Colab** and how to use it for machine learning and data analysis.

### You will learn:
- What Google Colab is and how it works  
- How to run cells, restart runtimes, and manage notebooks  
- How to use GPU/TPU resources  
- How to upload / download files and work with Google Drive  
- Basic workflow for running the other PSIML Applied AI notebooks  

This is the recommended first stop before exploring other notebooks.

---

## 👁️ 2. Vision — Find Anything in Images  
**File:** [`notebooks/PSIML_Tour_Vision.ipynb`](notebooks/PSIML_Tour_Vision.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pfe-rs/psiml-applied-ai/blob/master/notebooks/PSIML_Tour_Vision.ipynb)

This notebook focuses on modern **vision models** that can “find anything” in an image using natural language or flexible detection tools, going beyond classic object detection.

### You will learn:
- The difference between traditional object detection and foundation vision models  
- How to detect or segment objects using **text prompts**  
- How to highlight, crop, or mask regions of interest  
- How to build practical pipelines for:
  - Searching for objects  
  - Extracting and visualizing regions  
  - Using vision models as tools within larger systems  

---

## 📝 3. NLP — Understanding and Generating Text  
**File:** [`notebooks/PSIML_Tour_NLP.ipynb`](notebooks/PSIML_Tour_NLP.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pfe-rs/psiml-applied-ai/blob/master/notebooks/PSIML_Tour_NLP.ipynb)

This notebook introduces **language models** and practical NLP workflows.

### You will learn:
- How large language models generate and transform text  
- How tokenization works and why context length matters  
- Basic prompting strategies  
- How to use common NLP pipelines for:
  - Text generation  
  - Classification  
  - Sentiment analysis  
  - Summarization  
  - Named Entity Recognition (NER)  

You will see how to go from raw text to useful predictions in just a few lines of code.

---

## 🔊 4. Voice — Understanding and Generating Speech  
**File:** [`notebooks/PSIML_Tour_Voice.ipynb`](notebooks/PSIML_Tour_Voice.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pfe-rs/psiml-applied-ai/blob/master/notebooks/PSIML_Tour_Voice.ipynb)

This notebook covers **speech AI**, including both _speech-to-text_ and _text-to-speech_.

### You will learn:
- How automatic speech recognition (ASR) models work in practice  
- How to transcribe audio into text  
- How to generate synthetic speech  
- Basics of audio processing in Python (sampling, waveforms, spectrograms)  

It demonstrates complete workflows for turning **audio → text** and **text → audio**.

---

# ▶️ How to Run the Notebooks (Google Colab)

All notebooks are intended to be executed on **Google Colab**.

## **Option A — Open directly from GitHub**
1. Open any notebook on GitHub:  
   - `notebooks/Psiml_Tour_Collab.ipynb`  
   - `notebooks/PSIML_Tour_Vision.ipynb`  
   - `notebooks/PSIML_Tour_NLP.ipynb`  
   - `notebooks/PSIML_Tour_Voice.ipynb`
2. If the “**Open in Colab**” button is available, click it.  
3. If not, copy the GitHub URL and open it via:  
   **Colab → File → Open notebook → GitHub**

## **Option B — Download and upload to Colab**
1. Download the `.ipynb` file from GitHub  
2. Open https://colab.research.google.com  
3. Choose **Upload**, select the notebook, and run it  
4. Execute cells from top to bottom  
   (install commands like `pip install ...` should be run first)

---

# 💻 Running Outside Colab

If you want to run these notebooks on your own machine or server,  
please contact the PSIML team for guidance on the environment setup.

You can reach us via:
- **Discord**  
- **Instagram**  
- **Email**  
- **Direct message (DM)**  
or any other communication channel where PSIML provides support.

---

# ℹ️ About PSIML

**PSIML (Practical Seminar on Machine Learning)** is a hands-on educational initiative focused on modern AI methods, practical projects, and accessible machine learning resources.

This repository is part of the **Applied AI** materials used in PSIML workshops and sessions.