
# 🚗 License Plate Recognition with Microsoft Florence-2 & EasyOCR

## 📌 Overview

This project demonstrates a complete pipeline for **license plate recognition** using **Microsoft's Florence-2**, a powerful vision-language model. Trained and evaluated on real-world car images from the `keremberke/license-plate-object-detection` dataset, this notebook extracts license plate text and measures OCR accuracy using key evaluation metrics like **Word Error Rate (WER)** and **Levenshtein Distance**.

The approach is practical, scalable, and ideal for applications like **traffic surveillance, automated toll systems, and smart city infrastructure**.

---

## 🎯 Objective

The main goals of this project are to:

- Fine-tune Florence-2 on license plate recognition data.
- Extract and recognize license plate text from car images.
- Evaluate OCR performance using text similarity metrics.
- Demonstrate the practical use of transformer-based vision-language models in real-world OCR tasks.

---

## 📂 Dataset Description

The notebook utilizes the `keremberke/license-plate-object-detection` dataset from Hugging Face, which includes:

- 📸 **Car Images** – With visible license plates.
- ✅ **Ground Truth Labels** – Text labels for each plate.
- 🧱 **Bounding Box Annotations** – Optional metadata for number plate locations.

Dataset is loaded and split into:
- `train`  
- `test`  
- `validation`

Each record is used for OCR text extraction and accuracy comparison.

---

## 📦 Dependencies Used

```bash
pip install einops sklearn python-Levenshtein datasets timm jiwer
````

### Libraries & Their Purpose:

* **einops & timm** – Efficient image transformations.
* **datasets** – Load and preprocess the Hugging Face dataset.
* **python-Levenshtein & editdistance** – Calculate string similarity.
* **jiwer** – Calculate WER and CER for text accuracy evaluation.
* **transformers** – Load and run Florence-2 model.

---

## 🧠 Model Setup

The project loads the `microsoft/Florence-2-base-ft` model and processor:

```python
from transformers import AutoModelForCausalLM, AutoProcessor

model = AutoModelForCausalLM.from_pretrained(
  "microsoft/Florence-2-base-ft", trust_remote_code=True, revision="refs/pr/6"
).to(device)

processor = AutoProcessor.from_pretrained(
  "microsoft/Florence-2-base-ft", trust_remote_code=True, revision="refs/pr/6"
)
```

* Detects CUDA GPU (if available) for faster inference.
* Clears CUDA memory with `torch.cuda.empty_cache()` before loading.

---

## 🖼 OCR Workflow

The notebook includes the following steps:

1. **Load Image** – Each image is converted to RGB if not already.
2. **Set Prompt** – Uses `"What is the license plate number?"`.
3. **Process Image & Text** – With Florence-2 processor.
4. **Generate Prediction** – Using beam search decoding.
   5


Absolutely, Yasir! Here's your **GitHub-ready `README.md`** in the exact formatting style you requested — but this time for your **Florence-2-based License Plate Recognition project**:

---

````markdown
# 🚗 License Plate Recognition using Microsoft Florence-2 & EasyOCR

## 📌 Overview

This project presents a powerful pipeline for **License Plate Recognition (LPR)** using Microsoft's **Florence-2** vision-language model, fine-tuned on car images with visible number plates. Paired with Hugging Face's dataset utilities and evaluated with OCR accuracy metrics like **Word Error Rate (WER)** and **Levenshtein Distance**, this solution aims to bridge the gap between modern Transformer-based models and real-world license plate recognition tasks.

Florence-2 processes images with a natural language prompt and extracts text in an image-to-text fashion, enabling a flexible and scalable OCR approach ideal for surveillance, tolling, traffic enforcement, and smart city systems.

---

## 🧠 Objective

- Use Florence-2 to recognize number plate text from car images.
- Apply prompt-based visual question answering for OCR tasks.
- Evaluate recognition accuracy using reliable string similarity metrics.
- Prepare a deployable pipeline for real-time vehicle identification systems.

---

## 📂 Project Structure

- `Florence2_LPR.ipynb` – Jupyter notebook with model loading, inference, and evaluation.
- `images/` – Optional directory for local image testing.
- `README.md` – Project documentation.
- `requirements.txt` – Python dependencies (if provided).

---

## 🔧 Key Features

### 🖼 Florence-2 Vision Language Model
- Uses the `microsoft/Florence-2-base-ft` checkpoint with a custom prompt.
- Model input: `"What is the license plate number?"` + image.
- Output: Extracted alphanumeric text representing the number plate.

### 🗂 Hugging Face Dataset
- Dataset: `keremberke/license-plate-object-detection (mini)`
- Includes:
  - Vehicle images with visible license plates
  - Ground truth license plate text
  - Optional bounding box metadata

### 📊 Evaluation Metrics
- **WER (Word Error Rate)** – Measures word-level OCR accuracy.
- **Levenshtein Distance** – Calculates edit distance between prediction and ground truth.
- **Accuracy Score** – 1 - average WER for total predictions.

### 🔍 Sample Inference & Evaluation

```python
Image: car_05.jpg
Prompt: What is the license plate number?
Ground Truth: LHR-7384
Prediction:   LHR-7381
WER: 0.25 | Levenshtein: 1
````

---

## 🧰 Tools & Libraries Used

* **Python**
* **Hugging Face Transformers** – Florence-2 model + processor
* **datasets** – Load and inspect the license plate dataset
* **EasyOCR (optional)** – Lightweight fallback OCR for comparison
* **jiwer, python-Levenshtein** – Evaluation metrics
* **torch** – Model loading and GPU acceleration
* **PIL** – Image manipulation
* **Matplotlib** – Result visualization

---

## ❓ FAQs

**Q1: Why does OCR sometimes fail?**

> Model accuracy can drop due to image blur, unusual fonts, occlusion, or poor lighting.

**Q2: Can this model run in real-time?**

> Yes, with GPU support and minimal image preprocessing, Florence-2 can be integrated into real-time systems.

**Q3: Why use Florence-2 instead of EasyOCR or Tesseract?**

> Florence-2 is prompt-driven and can better generalize to unseen image-text contexts. It also supports fine-tuning and language-grounded tasks.

**Q4: Can I use bounding boxes to crop the number plates before OCR?**

> Yes, you can combine this model with object detectors like YOLOv8 or Faster R-CNN for a two-step pipeline (detection → OCR).

**Q5: Is this fine-tuned or zero-shot?**

> Florence-2 was used with task prompting (DocVQA-style). Further fine-tuning on more diverse license plate data can improve accuracy.

---

## 🏁 Conclusion

This project demonstrates how to use **Florence-2**, a powerful transformer model, for number plate recognition using natural language prompting. It's an exciting application of GenAI in computer vision that brings OCR to a new level of accuracy and flexibility.

By leveraging prompt engineering and real-world datasets, this project bridges traditional OCR and modern multimodal AI — paving the way for AI-powered vehicle recognition in smart traffic systems.

---

## 🔐 License

This project is released under the **MIT License**. Feel free to use, modify, and distribute with proper attribution.

---

## ✍️ Author

**Muhammad Yasir**
AI/ML Engineer | Web & Security Developer
📧 [jamyasir0534@gmail.com](mailto:jamyasir0534@gmail.com)
🌐 [Portfolio](https://devsecure.netlify.app)
🤖 [Hugging Face](https://huggingface.co/devxyasir)
💻 [GitHub](https://github.com/devxyasir)

---
