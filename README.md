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


This notebook provides a quick introduction to Google Colab—an online environment for running Python and Jupyter notebooks with many scientific and machine-learning libraries preinstalled. It demonstrates how to execute Python code and install additional packages directly within Colab.

This is the recommended first stop before exploring other notebooks.

---

## 👁️ 2. Vision — Find Anything in Images  
**File:** [`notebooks/PSIML_Tour_Vision.ipynb`](notebooks/PSIML_Tour_Vision.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pfe-rs/psiml-applied-ai/blob/master/notebooks/PSIML_Tour_Vision.ipynb)

This notebook demonstrates how modern vision models can locate, segment, and even modify objects in images using natural-language prompts. It combines three powerful tools: **Grounding DINO** for zero-shot object detection, **Segment Anything (SAM)** for generating high-quality masks, and **diffusers** pipelines for text-to-image generation and inpainting.

### You will learn:
- How zero-shot object detection works with **Grounding DINO**  
- How to turn detected boxes into segmentation masks using **SAM**  
- How to use **inpainting** models to replace or modify objects in the image  
- How to run complete, practical workflows for:
  - Finding objects using text prompts  
  - Visualizing detections and masks  
  - Editing images by removing or altering selected regions  


## 📝 3. NLP — Understanding and Generating Text  
**File:** [`notebooks/PSIML_Tour_NLP.ipynb`](notebooks/PSIML_Tour_NLP.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pfe-rs/psiml-applied-ai/blob/master/notebooks/PSIML_Tour_NLP.ipynb)

This notebook is a compact tour of modern language models: it starts with using LLM chat APIs (with system prompts, multi-turn conversations, and sampling parameters), then shows how to run a **small language model (SLM)** directly in Colab, and finally introduces a **vision-language model (VLM)** for image captioning.

### You will learn:
- How to call chat-style LLM APIs from code and structure system/user/assistant messages  
- How parameters like **temperature** and **top-p** affect model outputs  
- How to steer behavior with system prompts (e.g. for translation and “tricky” examples)  
- How to load and run a small open-source language model with `AutoModelForCausalLM` in Colab  
- How to use a vision-language model (`AutoModelForVision2Seq`) to generate natural language descriptions from images  

---

## 🔊 4. Voice — Understanding and Generating Speech  
**File:** [`notebooks/PSIML_Tour_Voice.ipynb`](notebooks/PSIML_Tour_Voice.ipynb)
[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/pfe-rs/psiml-applied-ai/blob/master/notebooks/PSIML_Tour_Voice.ipynb)

This notebook walks through a full speech-to-speech translation (S2ST) pipeline using a cascaded approach: **ASR → MT → TTS**. It uses a Whisper-style speech recognition model (`AutoModelForSpeechSeq2Seq` + `AutoProcessor`) to transcribe audio, a machine translation component (via OPUS-MT), and an XTTS-based `TTS` model to generate speech in the target language, with examples built on the Common Voice dataset.

### You will learn:
- What speech-to-speech translation is and why cascaded ASR → MT → TTS is a practical solution  
- How to load and run an ASR model with Hugging Face Transformers and pipelines  
- How to plug in a machine translation model (OPUS-MT) between ASR and TTS  
- How to use an XTTS text-to-speech model to synthesize translated speech  
- How to combine all components into a simple end-to-end speech-to-speech translation pipeline  


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


## ⚙️ Setting up the GPU (T4) in Google Colab

To ensure the notebooks run fast and smoothly, set Colab to use a **T4 GPU**:

1. Open the notebook in Colab  
2. Go to **Runtime → Change runtime type**  
3. Under **Hardware accelerator**, choose **GPU**  
4. Under **GPU type**, select **T4** (if available)  
5. Click **Save**  

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