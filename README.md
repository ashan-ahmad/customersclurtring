# Customer Clustering using K-Nearest Neighbors (KNN)

This project demonstrates the use of the K-Nearest Neighbors (KNN) algorithm to classify customers into predefined categories based on their demographic and service usage data. The dataset used in this project is a telecommunications dataset that categorizes customers into four groups based on their service usage patterns.

## Project Overview

The goal of this project is to build a classifier that predicts the service category of a customer based on their demographic data. The project involves the following steps:

1. **Data Loading and Exploration**: Load the dataset and explore its structure and contents.
2. **Data Visualization and Analysis**: Perform exploratory data analysis (EDA) to understand the relationships between features and the target variable.
3. **Data Preprocessing**: Normalize the data and split it into training and testing sets.
4. **Model Training**: Train a KNN classifier with different values of `k` to find the optimal number of neighbors.
5. **Model Evaluation**: Evaluate the model's performance using accuracy and visualize the results.
6. **Hyperparameter Tuning**: Experiment with different values of `k` to identify the best-performing model.

## Dataset Description

The dataset contains demographic and service usage data of customers. The target variable, `custcat`, represents the service category and has four possible values:

1. Basic Service
2. E-Service
3. Plus Service
4. Total Service

The dataset is balanced across the four categories, making it suitable for classification tasks.

## Key Features

- **Correlation Analysis**: Identify the most relevant features for classification by analyzing their correlation with the target variable.
- **Data Normalization**: Standardize the input features to improve the performance of the KNN algorithm.
- **Train-Test Split**: Split the dataset into training and testing sets to evaluate the model's generalization ability.
- **Hyperparameter Tuning**: Experiment with different values of `k` to optimize the model's performance.

## Results

The project evaluates the model's accuracy for different values of `k` and visualizes the results. The optimal value of `k` is determined based on the highest accuracy achieved on the test set.

## How to Run

1. Clone the repository to your local machine.
2. Install the required Python libraries:
   ```bash
   pip install numpy pandas matplotlib seaborn scikit-learn
   ```
3. Open the Jupyter Notebook file `Customer_Clustring.ipynb`.
4. Run the cells sequentially to execute the project.

## Dependencies

- Python 3.x
- Jupyter Notebook
- Libraries: `numpy`, `pandas`, `matplotlib`, `seaborn`, `scikit-learn`

## Conclusion

This project demonstrates the application of the KNN algorithm for customer classification. By analyzing the dataset and tuning the hyperparameters, we can build an effective model to predict customer categories based on their demographic data.

Feel free to explore the notebook and experiment with the code to gain a deeper understanding of the KNN algorithm and its applications in classification tasks.
