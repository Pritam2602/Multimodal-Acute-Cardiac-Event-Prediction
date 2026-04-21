# 🫀 Multimodal Acute Myocardial Infarction (AMI) Prediction

## 📌 Overview
This project focuses on predicting **Acute Myocardial Infarction (AMI)** using a **multimodal deep learning approach** that combines:

- 📈 **Raw ECG signals**
- 🧾 **Clinical data (Electronic Health Records - EHR)**

By integrating both physiological and biochemical information, the system aims to improve the accuracy and reliability of heart attack prediction.

---

## 🚀 Key Features

- ✅ Multimodal learning (ECG + EHR)
- ✅ Temporal alignment of clinical data
- ✅ Missing value handling with indicators
- ✅ Early Fusion and Late Fusion model comparison
- ✅ AMI label generation using ICD codes
- ✅ PyTorch-based deep learning models

---

## 🧠 Methodology

1. **Data Collection**
   - MIMIC-IV dataset (clinical data)
   - MIMIC-IV-ECG / PhysioNet (ECG signals)

2. **Data Preprocessing**
   - Handle missing values using median imputation
   - Create missing indicator features
   - Encode categorical variables

3. **Temporal Feature Engineering**
   - Use only clinical data recorded **before ECG time**
   - Extract latest values (`_last`) for each feature

4. **ECG Processing**
   - Load signals using WFDB
   - Normalize and fix signal length

5. **Label Generation**
   - AMI labels derived using ICD diagnosis codes

6. **Model Development**
   - **Early Fusion:** Combine ECG + EHR at input level
   - **Late Fusion:** Process ECG and EHR separately, then combine

7. **Evaluation Metrics**
   - Accuracy
   - Precision
   - Recall (important for AMI)
   - F1-score
---

## 🏗️ Model Architecture

### 🔹 Early Fusion

ECG → Feature Vector  
EHR → Feature Vector  
→ Concatenation → Dense Layers → Output (AMI)

---

### 🔹 Late Fusion

ECG → CNN → Embedding  
EHR → Dense → Embedding  
→ Concatenation → Dense Layers → Output (AMI)---

## Training Workflow

- The training script now saves a resumable checkpoint after every epoch at early_fusion/artifacts/models/latest_checkpoint.pth.
- The best validation-F1 model is saved at early_fusion/artifacts/models/early_fusion_model.pth.
- After each epoch, the script asks whether it should continue to the next epoch.
- If you stop and rerun the command later, it resumes from the next unfinished epoch automatically.

