# Intelligent Systems and Technologies Projects

## 📘 Overview  
This repository contains a series of practical projects completed as part of the **"Intelligent Systems and Technologies"** course. Each project explores different machine learning, data mining, and optimization techniques using Python and popular data science libraries.

## 📁 Projects  

### **1. Project 1: Association Rule Mining**  
**Objectives:**  
- Implement and compare association rule mining algorithms:  
  - Apriori  
  - Efficient Apriori  
  - FPGrowth  
- Generate rules with minimum confidence thresholds of 60% and 80%  
- Compute support, confidence, and significance metrics  
- Visualize results and compare algorithm performance  

**Datasets:**  
- Test dataset: 20 transactions of stationery purchases  
- Real dataset: BreadBasket_DMS.csv  

**Libraries used:**  
- `apriori-python`, `efficient-apriori`, `fpgrowth-py`  
- `pandas`, `numpy`, `matplotlib`  
- `pyarmviz` (for visualization)  


### **2. Project 2: Non-linear Dimensionality Reduction**  
**Objectives:**  
- Apply non-linear dimensionality reduction algorithms to visualize high-dimensional data in 2D:  
  - t-SNE  
  - UMAP  
  - TriMAP  
  - PaCMAP  
- Compare different scaling methods:  
  - MinMaxScaler  
  - StandardScaler  
  - RobustScaler  
- Test at least 6 parameter combinations per algorithm  
- Apply methods to both tabular data and 3D object embedding  

**Datasets:**  
- University dataset (UCI)  
- 3D mammoth point cloud (`mammoth.csv`)  

**Libraries used:**  
- `scikit-learn`  
- `umap-learn`, `trimap`, `pacmap`  
- `matplotlib`, `seaborn`  


### **3. Project 3: Development of KNN, SVM, and RF Classifiers**  
**Goal:** Build and compare K-Nearest Neighbors, Support Vector Machine, and Random Forest classifiers.  
**Dataset:** University dataset (UCI Repository).  
**Tasks:**  
- Preprocessing (handling missing values, encoding, scaling)  
- Hyperparameter tuning using GridSearchCV  
- Model evaluation with accuracy, precision, recall, and F1-score  
- Visualization using t-SNE, UMAP, TriMAP, and PaCMAP  
**Results:** Comparison of model performance with visual insights.


### **4. Project 4: Working with Sampling Strategies for Imbalanced Classes**  
**Goal:** Apply oversampling techniques to handle class imbalance.  
**Methods:** SMOTE, Borderline-SMOTE1, Borderline-SMOTE2.  
**Tasks:**  
- Balance the dataset using different oversampling methods  
- Re-train SVM, KNN, and RF classifiers on balanced data  
- Compare performance before and after balancing  
**Results:** Identification of the best balancing strategy for each classifier.


### **5. Project 5: Evolutionary Optimization Algorithms**  
**Goal:** Study and apply evolutionary algorithms for optimization.  
**Algorithms:** Genetic Algorithm, Differential Evolution.  
**Tasks:**  
- Optimize a multimodal test function (Ackley function)  
- Compare evolutionary algorithms with classical methods (BFGS)  
- Optimize hyperparameters of SVM, KNN, and RF using a Genetic Algorithm  
**Results:** Performance comparison in terms of convergence speed, accuracy, and stability.


### **6. Project 6: Exploratory and Cluster Analysis**  
**Goal:** Perform exploratory data analysis and clustering on multiple datasets.  
**Datasets:**  
- University dataset (known labels)  
- Mall Customers dataset (unknown labels)  
- Mammoth dataset (large-scale clustering)  
**Clustering Methods:** K-Means, Hierarchical, Fuzzy C-Means, DBSCAN.  
**Tasks:**  
- Determine optimal cluster numbers using silhouette and elbow methods  
- Evaluate clustering quality with silhouette score and Adjusted Rand Index  
- Visualize clusters using t-SNE, UMAP, TriMAP, and PaCMAP  
**Results:** Identification of best clustering algorithms for each dataset with profiling of clusters.


### **7. Project 7: Recurrent Neural Networks for Predictive Maintenance**  
**Objectives:**  
- Compare RNN, LSTM, and GRU architectures for predictive maintenance  
- Evaluate models using:  
  - Accuracy, Precision, Recall, F1-score, Loss  
- Analyze parameter counts in each layer  
- Visualize training/validation curves  
- Compare CPU vs GPU training times  

**Dataset:**  
- NASA Turbofan Engine Degradation Simulation  

**Libraries used:**  
- `tensorflow` (v1.15)  
- `keras`  
- `scikit-learn`  
- `pandas`, `numpy`, `matplotlib`  


## 🛠️ General Tools & Libraries  
- Python 3.x  
- Jupyter Notebook / Google Colab  
- pandas, numpy, matplotlib, seaborn  
- scikit-learn  
- Various specialized libraries per project (see above)  

## 📊 Key Findings  

1. **Association Rules:**  
   - FPGrowth was most efficient for large datasets  
   - Higher confidence thresholds reduce rule count but increase reliability  

2. **Dimensionality Reduction:**  
   - UMAP and t-SNE performed best for preserving local structure  
   - Scaling method significantly affects visualization quality  

3. **Classifier Comparison:**  
   - Random Forest and SVM outperformed KNN on the University dataset  
   - Oversampling improved recall for minority classes  

4. **Evolutionary Optimization:**  
   - Genetic Algorithm effectively optimized hyperparameters  
   - Differential Evolution showed faster convergence on test functions  

5. **Clustering:**  
   - DBSCAN performed well on irregular cluster shapes  
   - K-Means was most efficient for spherical, well-separated clusters  

6. **RNN Comparison:**  
   - LSTM and GRU outperformed simple RNN in capturing temporal dependencies  
   - GPU acceleration reduced training time by ~11x
