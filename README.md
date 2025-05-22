# Land Cover Classification using Machine Learning and Deep Learning

This repository presents a comparative study of land cover classification using three different techniques: **Random Forest (SVM), Convolutional Neural Networks (CNN), and Transfer Learning with ResNet50**. The project utilizes the EuroSAT dataset consisting of satellite images and evaluates each model's performance in terms of accuracy, efficiency, and generalization.

---

## 📌 Project Overview

Accurate land cover classification plays a vital role in environmental monitoring, urban planning, and sustainable development. This study analyzes how traditional machine learning, deep learning, and transfer learning models perform on geospatial satellite data.

---

## 🗂️ Dataset

- **Name**: EuroSAT RGB Dataset  
- **Source**: [Kaggle - EuroSAT Dataset](https://www.kaggle.com/datasets/apollo2506/eurosat-dataset)
- **Details**:  
  - 27,000 RGB images  
  - 64×64 resolution  
  - 10 land cover classes  
  - Split: 70% training, 20% validation, 10% testing  
  - Format: PNG + CSV files with paths

---

## 🔍 Methods Used

### 🔹 Random Forest (SVM Approach)
- Flattened image features
- Feature scaling and label encoding
- Trained with 100 estimators and bootstrapping
- Achieved 61% accuracy with limited performance on complex classes

### 🔹 Convolutional Neural Network (CNN)
- Built from scratch with:
  - Conv2D + MaxPooling + Dropout + GlobalAveragePooling layers
- Trained over 83 epochs with early stopping and learning rate scheduling
- Achieved 78% accuracy
- Strong generalization but struggled slightly with built-up areas

### 🔹 Transfer Learning with ResNet50
- Used pretrained ResNet50 with 30 unfrozen layers
- Fine-tuned on EuroSAT dataset
- Achieved 96% accuracy and perfect AUC across all classes
- Best model in terms of performance and generalization

---

## 📊 Model Comparison Summary

| Model       | Accuracy | Training Time | Macro F1 | Strengths                               |
|-------------|----------|----------------|----------|------------------------------------------|
| SVM (RF)    | 61%      | ~20 mins       | 0.59     | Interpretable, fast                      |
| CNN         | 78%      | ~13 mins       | 0.77     | Balanced accuracy & efficiency           |
| ResNet50 TL | 96%      | ~76 mins       | 0.96     | Superior accuracy & generalization       |

---

## 📈 Results & Visualizations

- **ROC Curves**: Showed near-perfect AUC scores for CNN and ResNet50
- **Confusion Matrices**: Revealed class-wise prediction accuracy
- **Classification Reports**: Detailed F1-scores, precision, recall for each class
- **Loss & Accuracy Curves**: Monitored training vs validation behavior

---

## 🧠 Key Takeaways

- Deep learning models (especially ResNet50) significantly outperform traditional approaches.
- CNN models are a good trade-off between training time and accuracy.
- Transfer Learning provides the best results when computational resources allow.

---

## 📁 Folder Structure

```
├── dataset/               # Contains image and CSV data
├── models/                # Saved model weights and architectures
├── notebooks/             # Training & evaluation Jupyter notebooks
├── results/               # Plots and evaluation reports
├── README.md              # Project overview and instructions
└── requirements.txt       # Dependencies
```

---

## 🛠️ Requirements

- Python 3.8+
- TensorFlow / Keras
- scikit-learn
- Matplotlib / Seaborn
- Pandas / NumPy

Install dependencies:
```bash
pip install -r requirements.txt
```

---

## 👩‍💻 Author

**Swamini Sontakke**  
M.S. in Computer Science  
Project: Comparative Analysis of Land Cover Classification Techniques

---

## 📚 References

A full list of academic references is available in the report (see `/Report.pdf`).

---

## 📝 License

This project is for academic and research purposes only.
