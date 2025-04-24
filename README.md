# COVID-19 Detection Using Machine Learning: A Comparative Analysis

![COVID-19 Detection](https://via.placeholder.com/800x400?text=COVID-19+X-ray+Analysis)  
**A comparative study of machine learning algorithms for detecting COVID-19 from chest X-rays.**

---

## 📋 Table of Contents
- [Abstract](#abstract)
- [Introduction](#introduction)
- [Dataset](#dataset)
- [Algorithms Evaluated](#algorithms-evaluated)
- [Methodology](#methodology)
- [Results](#results)
- [Conclusion](#conclusion)
- [License](#license)

---

## 🌟 Abstract
The COVID-19 pandemic has accelerated the need for efficient diagnostic tools. This project evaluates the performance of **seven machine learning algorithms**—Support Vector Machines (SVM), Random Forests, Logistic Regression, Decision Trees, Artificial Neural Networks (ANN), ResNet, and DenseNet—for detecting COVID-19 using chest X-rays. The dataset includes **3,616 COVID-19-positive** and **10,192 normal X-rays**. Key findings reveal **Logistic Regression outperforms complex models with 99.1% accuracy**, highlighting its effectiveness for this task.

---

## 📚 Introduction
The study focuses on leveraging medical imaging (X-rays/CT scans) and machine learning to improve COVID-19 diagnostics. By comparing traditional and deep learning models, we aim to identify the most reliable approach for rapid and accurate detection, aiding healthcare systems globally.

---

## 📂 Dataset
- **Source**: [COVID-19 Radiography Database](https://www.kaggle.com/datasets/tawsifurrahman/covradiography-database) on Kaggle.
- **Details**:
  - **COVID-19 Cases**: 3,616 X-rays.
  - **Normal Cases**: 10,192 X-rays.
- **Preprocessing**:
  - Normalization and standardization.
  - Class balancing via **SMOTE** (oversampling) to address imbalance.

---

## 🤖 Algorithms Evaluated
1. **Support Vector Machines (SVM)**  
   - Linear/non-linear classification using optimal hyperplanes.
2. **Random Forests**  
   - Ensemble of decision trees for robust predictions.
3. **Logistic Regression**  
   - Binary classification using probabilistic modeling.
4. **Decision Trees**  
   - Hierarchical splits based on feature conditions.
5. **Artificial Neural Networks (ANN)**  
   - Multi-layer perceptrons for pattern recognition.
6. **ResNet**  
   - Deep residual networks addressing vanishing gradients.
7. **DenseNet**  
   - Densely connected convolutional blocks for feature reuse.

---

## 🔧 Methodology
1. **Data Preprocessing**  
   - Normalization, augmentation, and class balancing (SMOTE).
2. **Model Training**  
   - Train-test split, hyperparameter tuning, and cross-validation.
3. **Evaluation Metrics**  
   - Accuracy, precision, recall, and F1-score.

![Workflow](https://via.placeholder.com/600x300?text=Project+Flowchart+-+Data+Preprocessing+to+Model+Evaluation)

---

## 📊 Results
| Algorithm           | Accuracy |
|---------------------|----------|
| Logistic Regression | 99.1%    |
| SVM                 | 87%      |
| DenseNet            | 81%      |
| Random Forest       | 80%      |
| ResNet              | 75%      |
| Decision Tree       | 71%      |
| ANN                 | 66%      |

**Key Insight**: Logistic Regression achieved the highest accuracy (99.1%) despite its simplicity, outperforming deep learning models like ResNet and DenseNet.

---

## 🎯 Conclusion
- **Logistic Regression** emerged as the most effective model for this dataset, likely due to its interpretability, regularization, and compatibility with linear relationships.
- Deep learning models (ResNet, DenseNet) showed moderate performance, suggesting potential for improvement with larger datasets or advanced tuning.
- Model choice should align with **dataset characteristics** and clinical requirements. Future work may explore hybrid/ensemble approaches.


---

