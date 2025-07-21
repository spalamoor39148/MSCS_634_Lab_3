# MSCS_634_Lab_3

# 📊 Clustering Analysis Using K-Means and K-Medoids Algorithms

## Purpose

The purpose of this lab is to explore **unsupervised machine learning techniques**, specifically clustering algorithms, using the **Wine dataset** from `sklearn.datasets`. The Wine dataset contains chemical attributes of wines from three different cultivars. The main objectives of the lab are:

- To apply and compare the performance of **K-Means** and **K-Medoids** clustering algorithms.
- To evaluate the clustering results using **Silhouette Score** and **Adjusted Rand Index (ARI)**.
- To understand the advantages and trade-offs between K-Means and K-Medoids in terms of accuracy, robustness, and interpretability.
- To visualize the resulting clusters using **Principal Component Analysis (PCA)**.

---

##  Key Insights & Observations

###  Algorithm Comparison
- **K-Means**:
  - Achieved **higher Silhouette Score** and **ARI**, indicating more compact and well-separated clusters.
  - Performs well when clusters are spherical and there are no significant outliers.
  - Cluster centroids are the mean of points, which may not correspond to actual data samples.

- **K-Medoids**:
  - Slightly lower performance metrics than K-Means but **more robust to noise and outliers**.
  - Uses actual data points (medoids) as cluster centers, making it **more interpretable**.
  - Slightly less compact cluster boundaries, as seen in PCA visualization.

###  Visual Findings
- PCA-based scatter plots revealed that both algorithms could capture the structure of the dataset reasonably well.
- K-Means showed **tighter, more circular clusters**, while K-Medoids had **more irregular shapes**, especially near class boundaries.

###  Metric Summary
| Metric               | K-Means | K-Medoids |
|----------------------|---------|-----------|
| Silhouette Score     | ~0.28   | ~0.27     |
| Adjusted Rand Index  | ~0.89   | ~0.86     |

---

##  Implementation Details

- **Standardization**: All features were normalized using **Z-score normalization** to bring them onto a common scale.
- **Evaluation Metrics**:
  - **Silhouette Score**: Evaluates how well samples are clustered with respect to their own cluster vs. other clusters.
  - **Adjusted Rand Index (ARI)**: Measures the similarity between the clustering result and the actual class labels.
- **Visualization**: PCA was used to reduce dimensionality to 2D for effective plotting and visual comparison of clusters and centroids/medoids.
- **Libraries Used**:
  - `scikit-learn` for K-Means, metrics, PCA, and preprocessing
  - `pyclustering` for K-Medoids (due to compatibility issues with `scikit-learn-extra`)

---

##  Challenges Faced & Decisions Made

- **Binary incompatibility with scikit-learn-extra**:
  - Attempting to use `KMedoids` from `scikit-learn-extra` caused a `numpy.dtype` size mismatch error.
  - This was resolved by switching to the **`pyclustering`** package, which offers a Python-native implementation of K-Medoids without C-extension dependencies.
  
- **Initial Medoid Selection**:
  - Initial medoids were selected from diverse data points (index 0, 50, 100) to prevent poor convergence or empty clusters.

- **Cluster Evaluation**:
  - Since clustering is unsupervised, external validation like ARI (comparing with true labels) was used **only for analysis**, not model training.

---

## Conclusions

- **K-Means** is generally more efficient and accurate for clean datasets with spherical clusters, but it is sensitive to outliers and may select centroids that do not correspond to real data points.
- **K-Medoids** is more robust and interpretable, particularly useful when you want actual examples as cluster representatives or when dealing with noisy data.
- Visualization and metric evaluation together provide valuable insights into cluster quality beyond just numerical scores.

---



