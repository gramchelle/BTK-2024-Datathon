# DATATHON 2024
### Organized by BTK Academy, Google, and the Entrepreneurship Foundation
## Repository Overview
This repository contains all the necessary files and notebooks developed during my participation in Datathon 2024. The objective of the competition was to build a machine learning model that could predict specific outcomes with high accuracy. This repository includes the code, models, and final submission file used to generate my predictions.
In this project I worked with pretty much models to generate the best predictions. The most profitable ones were with stacking regressors. In this project i used catboost, extra trees regressor, xgboost and lightgbm to develop and boost the model.
### Contents:
* Notebook: The main Jupyter notebook where all data preprocessing, feature engineering, model training, and evaluation processes are documented.
* submission.csv: The file submitted for evaluation, containing the predicted values for the test set as per competition requirements.
## Datathon Main Task
In this competition, we worked with data from the Entrepreneurship Foundation, which has been collecting applications since 2014. The dataset (train.csv) includes a column called "Değerlendirme Puanı", representing the assessment score of each applicant. Additionally, the dataset contains anonymized information about applicants, such as their university, family background, residence details, and many other features. The goal was to analyze this information and accurately predict the "Değerlendirme Puanı" for unseen data.
## Project Description
In this competition, I explored a variety of machine learning models, focusing on improving predictive accuracy and model robustness. The core of my solution was built around ensemble techniques, particularly stacking regressors, which combines the predictions of several base models to enhance overall performance.
### Key Models Used:
* CatBoost: A gradient boosting algorithm optimized for categorical features.
* XGBoost: An efficient implementation of gradient-boosted decision trees.
* LightGBM: A powerful, fast, and memory-efficient model designed for large-scale data.
* Extra Trees Regressor: A tree-based model known for reducing overfitting and variance.
In addition to ensemble models, I experimented with regularization techniques like Ridge and Lasso regression to address overfitting and improve generalization.
### Data Preprocessing and Feature Engineering:
To extract the maximum potential from the dataset, I applied several preprocessing techniques, such as:
* Label encoding of categorical features.
* Handling missing values through imputation and mode filling.
* One-hot encoding for specific categorical variables where appropriate.
* Feature scaling using StandardScaler to normalize numerical data.
* Feature engineering by creating additional relevant features from the existing data to better capture relationships and improve model accuracy.
### Model Development and Optimization:
During model development, I implemented a robust pipeline that integrated cross-validation techniques, such as KFold Cross-Validation, to ensure the model's stability across different subsets of data. I also leveraged GridSearchCV to tune hyperparameters for models like LightGBM and XGBoost, which played a crucial role in optimizing performance.

One of the key strategies I employed was stacking regressors, a powerful ensemble learning technique. By stacking models like CatBoost, Extra Trees, and LightGBM, I could combine their strengths, resulting in improved prediction accuracy and lower RMSE scores. The ensemble model consistently outperformed individual models.
### Performance and Leaderboard Standing:
With approximately 575 individual participants (excluding teams), I achieved a final RMSE score of 6.37 on the private leaderboard. This placed me within the top 39% of participants including teams and %24.6 excluding teams, demonstrating the effectiveness of the models and techniques applied throughout the competition.
## Conclusion
This project allowed me to deeply explore advanced machine learning techniques, feature engineering, and ensemble methods. The final model was a result of iterative experimentation, rigorous cross-validation, and careful tuning. The experience not only enhanced my skills in model development but also in handling real-world datasets with a focus on predictive accuracy and model generalization.

For any questions or further information, please feel free to reach out.

You can also reach me from:
- [LinkedIn](https://www.linkedin.com/in/ozlemnurduman)
- [Kaggle Profile](https://www.kaggle.com/gramchelle)
