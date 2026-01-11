# Finalterm-Machine-Learning
## Hands-On End-to-End Models: Classification, Regression, and Deep Learning
The goal of this project is to enhance practical knowledge in both **Machine Learning** and **Deep Learning** by developing end-to-end pipelines. This includes data cleaning, feature engineering, model selection, hyperparameter tuning, and performance evaluation to solve real-world data problems.

##  Repository Navigation
This repository is organized into three main project file:

| Project File | Task Type | Description |
| :--- | :--- | :--- |
| [Fraud Detection](./finalterm_transaction_data.ipynb) | **Classification** | Predicting the probability of fraudulent online transactions using Random Forest. |
| [Song Year Regression](./finalterm_regression.ipynb) | **Regression** | Predicting song release years based on audio timbre features using Ridge Regression. |
| [Fish Classification](./finalterm_fish_img.ipynb) | **Deep Learning** | Classifying fish species using custom CNN architectures and MobileNetV2 Transfer Learning. |

## Project Overviews & Model Descriptions

### A. Fraud Detection (Classification)
- **Problem:** Predicting if an online transaction is fraudulent (`isFraud`).
- **Pipeline:** * Merging Transaction and Identity datasets.
    - Handling high-null columns and median imputation.
    - Addressing class imbalance using `class_weight='balanced'`.
- **Model:** **Random Forest Classifier**.
- **Metric:** **ROC-AUC Score**. 
    - $ROC-AUC$ was used to evaluate the model's ability to distinguish between classes independent of the threshold.

### B. Song Release Year Prediction (Regression)
- **Problem:** Predicting a continuous value (Release Year) from 90 audio features.
- **Pipeline:** * Outlier detection and removal using the IQR method.
    - Feature scaling using `StandardScaler`.
    - Hyperparameter tuning via `GridSearchCV`.
- **Model:** **Ridge Regression** (Linear Regression with $L_2$ Regularization).
- **Metrics:** **RMSE** (Root Mean Squared Error) and **$R^2$ Score**.

### C. Fish Species Identification (Deep Learning)
- **Problem:** Classifying images of fish into multiple species categories.
- **Pipeline:** * Modern `tf.data` pipeline for efficient image loading.
    - In-model Data Augmentation (Flip, Rotation, Zoom).
- **Architectures:**
    1.  **Custom CNN:** A sequential architecture built from scratch.
    2.  **Transfer Learning:** Utilizing **MobileNetV2** pre-trained on ImageNet.
- **Metric:** **Categorical Accuracy** and **F1-Score**.
