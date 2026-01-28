# Intelligent Systems and Technologies - Practical Projects

This repository contains a series of practical projects completed as part of the **Intelligent Systems and Technologies** course. The projects cover a wide range of machine learning and data science topics, implemented in Python using popular libraries like Scikit-learn, Pandas, Matplotlib, and others.


## 📁 **Projects Overview**

### 1. **Projects 5–6: Development of KNN, SVM, and RF Classifiers**
- **Goal:** Build and compare K-Nearest Neighbors, Support Vector Machine, and Random Forest classifiers.
- **Dataset:** University dataset (UCI Repository).
- **Tasks:**
  - Preprocessing (handling missing values, encoding, scaling).
  - Hyperparameter tuning using GridSearchCV.
  - Model evaluation with accuracy, precision, recall, and F1-score.
  - Visualization using t-SNE, UMAP, TriMAP, and PaCMAP.
- **Results:** Comparison of model performance with visual insights.


### 2. **Projects 7–8: Working with Sampling Strategies for Imbalanced Classes**
- **Goal:** Apply oversampling techniques to handle class imbalance.
- **Methods:** SMOTE, Borderline-SMOTE1, Borderline-SMOTE2.
- **Tasks:**
  - Balance the dataset using different oversampling methods.
  - Re-train SVM, KNN, and RF classifiers on balanced data.
  - Compare performance before and after balancing.
- **Results:** Identification of the best balancing strategy for each classifier.


### 3. **Projects 9–10: Evolutionary Optimization Algorithms**
- **Goal:** Study and apply evolutionary algorithms for optimization.
- **Algorithms:** Genetic Algorithm, Differential Evolution.
- **Tasks:**
  - Optimize a multimodal test function (Ackley function).
  - Compare evolutionary algorithms with classical methods (BFGS).
  - Optimize hyperparameters of SVM, KNN, and RF using a Genetic Algorithm.
- **Results:** Performance comparison in terms of convergence speed, accuracy, and stability.


### 4. **Projects 11–12: Exploratory and Cluster Analysis**
- **Goal:** Perform exploratory data analysis and clustering on multiple datasets.
- **Datasets:**
  - University dataset (known labels)
  - Mall Customers dataset (unknown labels)
  - Mammoth dataset (large-scale clustering)
- **Clustering Methods:** K-Means, Hierarchical, Fuzzy C-Means, DBSCAN.
- **Tasks:**
  - Determine optimal cluster numbers using silhouette and elbow methods.
  - Evaluate clustering quality with silhouette score and Adjusted Rand Index.
  - Visualize clusters using t-SNE, UMAP, TriMAP, and PaCMAP.
- **Results:** Identification of best clustering algorithms for each dataset with profiling of clusters.


## 🛠️ **Technologies Used**
- **Python 3**
- **Libraries:** Pandas, NumPy, Scikit-learn, Matplotlib, Seaborn, Imbalanced-learn, SciPy, UMAP, Trimap, Pacmap, Geneticalgorithm
- **Environment:** Jupyter Notebook
