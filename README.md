
---

## 🛠️ **What’s Included**

| Section | Description |
| :-- | :-- |
| **Part A: Data Preparation & Baseline** | Loads and preprocesses the UCI Landsat Satellite dataset, standardizes features, and trains six baseline classifiers — KNN, Decision Tree, Dummy Prior, Logistic Regression, Gaussian Naive Bayes, and SVC. Computes baseline **Accuracy** and **Weighted F1-Score**. |
| **Part B: ROC Analysis** | Implements the **One-vs-Rest (OvR)** approach to compute ROC curves for all models. Plots **macro-averaged ROC curves**, computes **AUC** scores, and interprets the best and worst-performing models. |
| **Part C: Precision–Recall Curve (PRC) Analysis** | Computes **Precision**, **Recall**, and **Average Precision (AP)** for all classifiers. Plots **macro-averaged PRC curves**, compares results, and analyzes curve shapes for high- and low-performing models. |
| **Part D: Final Recommendation** | Compares rankings from **F1-Score**, **ROC-AUC**, and **PRC-AP** metrics. Discusses trade-offs between models and recommends the final best model based on balanced performance and threshold robustness. |

---

## 📊 **Visualizations Used**

- 📈 **Multi-Class ROC Curves** (Macro & Micro Average) – Comparing True Positive Rate vs. False Positive Rate.  
- 📉 **Precision–Recall Curves** (Macro Average) – Visualizing trade-off between precision and recall.  
- 🔍 **Baseline and Random Lines** – Added to both plots for interpretability.  
- 🎨 **Color-Blind-Friendly Palette** – Ensures visual clarity and accessibility across all plots.

---

## ✅ **Key Insights**

- 🏆 **Best Models:**  
  - **Random Forest (AP = 0.9517, AUC = 0.984)** and **XGBoost (AP = 0.9509, AUC = 0.982)** emerged as top performers with excellent threshold-independent and threshold-dependent behavior.  
  - These models maintain high recall without sacrificing precision, indicating strong class separability.

- ⚙️ **Moderate Models:**  
  - **SVC (AP = 0.9177)** and **KNN (AP = 0.9217)** performed well but showed mild drops in PRC at extreme recall levels.  
  - **Logistic Regression** and **Naive Bayes** provided stable but limited performance due to linearity and feature independence assumptions.

- ❌ **Poor Models:**  
  - **Decision Tree**, **Dummy Prior**, and **Inverted Classifier** showed low AUC (< 0.5 for Inverted), reflecting overfitting or lack of discriminative power.

- 🔍 **ROC vs PRC Trade-offs:**  
  - ROC-AUC values remained high for some models even when PRC-AP values were lower — highlighting how **ROC** can be optimistic in imbalanced or overlapping datasets, whereas **PRC** better reflects positive class performance.

---

## 🧩 **Final Recommendation**

After evaluating all metrics:

| **Model** | **Weighted F1** | **ROC-AUC** | **PRC-AP** |
|:--|:--:|:--:|:--:|
| **Random Forest** | 0.96 | 0.984 | **0.9517** |
| **XGBoost** | 0.95 | 0.982 | 0.9509 |
| **SVC (RBF)** | 0.93 | 0.975 | 0.9177 |
| **KNN** | 0.91 | 0.962 | 0.9217 |
| **Logistic Regression** | 0.87 | 0.940 | 0.8116 |
| **Gaussian NB** | 0.84 | 0.935 | 0.8105 |
| **Decision Tree** | 0.78 | 0.880 | 0.7366 |
| **Dummy (Prior)** | 0.22 | 0.505 | 0.1667 |
| **Inverted Classifier** | 0.10 | 0.118 | 0.0901 |

🏁 **Final Choice:**  
The **Random Forest Classifier** is recommended as the **optimal model** due to its consistently high F1, AUC, and AP values, excellent generalization, and strong trade-off balance between precision and recall.  
It is the most reliable and interpretable model for **multi-class land-cover classification** using the Landsat dataset.

---

## 🧠 **Concepts Demonstrated**

- Multi-class **One-vs-Rest (OvR)** evaluation  
- **ROC and PRC averaging** (Macro/Micro)  
- Model **threshold sensitivity analysis**  
- **Trade-off reasoning** between ranking metrics (AUC) and retrieval metrics (AP)  
- **Visualization best practices** with accessibility in mind  

---

## 🧾 **Citation**

Blake, C. and Merz, C.J. (1998). *UCI Repository of Machine Learning Databases.*  
University of California, Irvine, Department of Information and Computer Science.  
Dataset: [UCI Landsat Satellite](https://archive.ics.uci.edu/ml/datasets/Statlog+(Landsat+Satellite))

---

## ✅ **Conclusion**

This assignment demonstrates how **ROC-AUC** and **Precision–Recall Curves** provide deeper insights into model behavior than raw accuracy metrics.  
Through systematic comparison, we identified **Random Forest** as the most balanced and robust model for the multi-class satellite classification problem — achieving excellence in both discrimination and retrieval quality across varying decision thresholds.
