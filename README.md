# Unsupervised-ML.

# Netflix Movies and TV Shows Clustering Project

![Python](https://img.shields.io/badge/Python-3.x-blue.svg)  
![Machine Learning](https://img.shields.io/badge/Machine%20Learning-KMeans%2C%20Hierarchical-blue.svg)  
![Scikit-learn](https://img.shields.io/badge/Scikit--learn-0.24.2-orange.svg)  
![License](https://img.shields.io/badge/License-MIT-green.svg)  
![Contributions](https://img.shields.io/badge/Contributions-Welcome-brightgreen.svg)

---

## Table of Contents
- [Introduction](#introduction)
- [Problem Statement](#problem-statement)
- [Project Workflow](#project-workflow)
- [Models and Evaluation](#models-and-evaluation)
- [Dataset Overview](#dataset-overview)
- [Key Findings](#key-findings)
- [Technologies Used](#technologies-used)
- [Installation and Setup](#installation-and-setup)
- [Usage Instructions](#usage-instructions)
- [Future Work](#future-work)
- [Conclusion](#conclusion)
- [License](#license)
- [Contact Information](#contact-information)

---

## Introduction

The **Netflix Movies and TV Shows Clustering Project** is focused on analyzing and grouping Netflix content (movies and TV shows) into meaningful clusters using machine learning techniques. This project employs data preprocessing, exploratory data analysis (EDA), feature engineering, and clustering algorithms. The insights derived from the clusters could potentially help Netflix understand content similarity, target specific audiences, or recommend similar content.

---

## Problem Statement

With the massive amount of content available on Netflix, identifying similarities between movies and TV shows is challenging. The purpose of this project is to apply unsupervised learning techniques to group Netflix titles by analyzing their patterns and characteristics. Clustering will provide the foundation for better understanding user preferences, driving recommendations, and categorizing content efficiently.

---

## Project Workflow

This project follows the following structured workflow:

1. **Understanding and Exploring the Data**  
   → Data cleaning, handling missing values, and preparing data for further analysis.  

2. **Data Visualization and Storytelling**  
   → Utilizing charts and graphs to explore relationships between variables.  
   → Drawing insights to understand the data better.  

3. **Feature Engineering and Data Preprocessing**  
   → Transforming data with scaling, encoding, and PCA to make it suitable for machine learning.  

4. **Model Implementation**  
   - **Cluster Visualization**: Models were used to visualize how data is clustered and how distinct the clusters are.
   - **Performance Metrics**: Elbow Curve, Silhouette Score, and other evaluation techniques to determine optimal values.  

5. **Model Evaluation**  
   → Evaluate models' cluster quality using metrics like Silhouette coefficient and the Calinski-Harabasz Index.  

6. **Conclusions and Insights**  

---

## Models and Evaluation

Three machine learning models were implemented to achieve clustering and evaluate their performance:

### 1. **KMeans Clustering**  
   - Used the **Elbow Method** to determine the optimal number of clusters.  
   - Visualized clusters using scatter plots and centroids to evaluate alignment with data.  
   - Key metric: **Within-Cluster Sum of Squares (WCSS)**.  

### 2. **Silhouette Method**  
   - Analyzed the **Silhouette Score** for evaluating cluster quality.  
   - Demonstrated clarity of cluster boundaries.  

### 3. **Hierarchical Clustering**  
   - Implemented **Agglomerative Clustering** and visualized clusters using a **Dendrogram**.  
   - Evaluated clusters using Silhouette Score and **Davies-Bouldin Score**.  

#### **Evaluation Metrics Used**
   - **Silhouette Coefficient**: Measures the separation and clustering quality—higher scores indicate better separation.  
   - **Calinski-Harabasz Index**: Measures the ratio of between-cluster to within-cluster variance.  
   - **Distortion**: Average squared distance of cluster points from the center—the lower, the better.  

---

## Dataset Overview

The Netflix Movies and TV Shows dataset contains the following key attributes:  
- Title  
- Type (Movie/TV Show)  
- Genre  
- Release Year  
- Ratings  
- Duration  

#### Dataset Statistics:
- **Number of titles**: Includes movies and TV shows.  
- **Missing Data**: Handled appropriately during preprocessing.  

---

## Key Findings  

- **Optimal Clusters**:  
   - **KMeans:** Optimal number of clusters is **k=2**, as observed from the Elbow Curve and Silhouette Score.  
   - Clusters are non-overlapping and represent well-defined groups.  

- **Content Grouping**: Similar content (based on features like genre, duration, etc.) was grouped together.  

- **Hierarchical Analysis**: The Dendrogram provided a clear view of hierarchical grouping, complementing KMeans clustering.  

---

## Technologies Used

- **Python**: Main programming language for this project.
- **Libraries and Tools**:  
   - **Pandas**: For data manipulation and exploration.  
   - **Scikit-learn**: For clustering and evaluation.  
   - **Matplotlib/Seaborn**: For visualizations and storytelling.  
   - **Yellowbrick**: For evaluation metric visuals.  

---

## Installation and Setup  

1. Clone the repository:  
   ```bash
   git clone https://github.com/yourusername/netflix-clustering.git
   cd netflix-clustering

Usage Instructions
Load the Dataset: Ensure the cleaned Netflix dataset provided in the repository is available.
Run the Notebook: Open the notebook and execute each cell step-by-step:
Preprocess the data.
Implement clustering algorithms.
Evaluate and analyze their performance via various metrics.
View Insights:
Graphs and visualizations of clustering results (e.g., Elbow Curve, Silhouette Score).
Interpret the clustering patterns obtained for Netflix content.
Assess Results:
Explore the relationship between Netflix content clusters based on features like Genre, Duration, and Ratings.
Observe how clustering helps group similar movies or TV shows effectively.
Experiment:
Try modifying the clustering parameters (e.g., n_clusters) and see how they affect performance metrics.
Use your business understanding to validate cluster separations.
For additional exploration, you can implement advanced pipelines or integrate the cluster results into a content recommendation system.

Future Work
This project can be further extended with the following improvements:

Advanced Clustering Algorithms:
Experiment with algorithms like DBSCAN or Gaussian Mixture Models.
Content Recommendation System:
Use clustering results to develop a recommendation engine for similar content.
Explainable AI:
Integrate tools like SHAP or LIME to interpret feature importance behind clustering.
Bigger Dataset:
Expand the analysis to include other streaming platforms (like Amazon Prime, Hulu) for a broader perspective.
Conclusion
This project applies machine learning techniques to group Netflix titles into clusters based on their features, such as Genre, Type (TV Show/Movie), Ratings, and Duration. Models like KMeans, Silhouette Method, and Hierarchical Clustering were implemented to identify patterns and provide valuable insights into Netflix's content structure. These insights lay the groundwork for personalized recommendations, content categorization, and user segmentation for targeted marketing.

License
This project is licensed under the MIT License. Feel free to use, modify, and distribute the code as needed.

Contact Information
Author: Sanidhya Shekhar
Email: sanidhyashekhar1996@gmail.com
LinkedIn: linkedin.com/in/sanidhya-shekhar
GitHub: github.com/sanidhya1996
