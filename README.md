
# Deep Learning-Based Pancreatic Tumor Detection

This project leverages deep learning and machine learning techniques to enhance the early detection and classification of pancreatic tumors using CT scan images and clinical data. By combining image processing with numerical analysis, the system aims to support clinicians with accurate, automated diagnostic insights.

## 🧠 Objective

- Utilize pre-trained deep learning models (AlexNet, VGG16, ResNet50, and EfficientNetB0) for pancreatic tumor classification from CT scans.
- Apply machine learning classifiers (Random Forest and Decision Tree) on clinical data for tumor diagnosis and stage prediction.
- Minimize false negatives and false positives to improve early diagnosis and treatment planning.

---

## 🏥 Problem Statement

Traditional diagnostic methods for pancreatic cancer, such as manual analysis of CT/MRI/EUS scans, are time-consuming and error-prone. They often struggle with early-stage tumor detection and exhibit high false positive and false negative rates. This project proposes an AI-driven approach to increase accuracy and reliability in diagnosis.

---

## 🔍 Proposed Solution

- **Image Classification**: Fine-tuning of pre-trained models on a dataset of CT scans to classify images as “Normal” or “Pancreatic Tumor.”
- **Clinical Data Analysis**: Using machine learning algorithms to process biomarkers (CA19-9, creatinine, LYVE1, REG1B, etc.) and patient data for diagnosis and staging.
- **Hybrid System**: Integration of image and numeric data analysis for comprehensive and reliable detection.

---

## 🗂️ Dataset

### 📸 CT Scan Images
- Total: 1,000 images (500 Normal, 500 Tumor)
- Format: JPG, 512x512 resolution
- Source: Kaggle

### 📊 Clinical Data
- Total Records: 1,770
- Features: 14 (age, sex, biomarkers, diagnosis, stage, etc.)
- Missing values handled through imputation and cleaning.

---

## ⚙️ Data Preprocessing

### Images
- Resize to model-specific dimensions (e.g., 227x227 for AlexNet)
- Convert to tensors using PyTorch
- Normalize pixel values

### Numerical Data
- Handle missing values via imputation
- One-hot encode categorical features
- Feature selection based on correlation

---

## 🤖 Models Used

### Image Classification
| Model          | Framework    | Accuracy |
|----------------|--------------|----------|
| AlexNet        | PyTorch      | ~93%     |
| VGG16          | Keras (TF)   | **97.5%** |
| ResNet50       | Keras (TF)   | ~95%     |
| EfficientNetB0 | Keras (TF)   | ~94%     |

> 🥇 **VGG16 outperformed all other models** in accuracy, precision, recall, and F1-score.

### Clinical Data Classification
| Model               | Accuracy |
|---------------------|----------|
| Decision Tree       | ~94%     |
| Random Forest       | **99%**  |

---

## 🧪 Evaluation Metrics

- **Accuracy**
- **Precision**
- **Recall (Sensitivity)**
- **F1-Score**
- Confusion Matrix

---

## 🏁 Results

- **Image Classification**: VGG16 achieved **97.5%** accuracy, demonstrating excellent reliability in detecting pancreatic tumors from CT scans.
- **Clinical Data Analysis**: Random Forest achieved **99%** accuracy, effectively analyzing patient biomarkers.

---

## 🧩 System Architecture

- Input: CT scan images + clinical records
- Preprocessing: Normalization, encoding, cleaning
- Processing: CNN (image) + Random Forest/Decision Tree (numeric)
- Output: Tumor classification + diagnostic insights


## 👩‍💻 Authors

- **Priyanka M** (RA2111003011351)
- **Srivathsan K** (RA2111003011360)

### 🎓 Institution:
SRM Institute of Science and Technology  
School of Computing, Department of Computing Technologies

---

## 🧑‍🏫 Guide

**Mrs. S. Poonkodi**  
Assistant Professor, Department of Computing Technologies

---

## 💡 Future Enhancements

- Integration of real-time medical imaging devices.
- Deploy the model in a web or mobile application for clinical usage.
- Expand dataset size for better generalization.
  

---

## 🗃️ Repository Structure

