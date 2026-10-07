# Iris Flower Classification — CodeAlpha Task 1

## Project Overview
This project is part of the CodeAlpha Data Science Internship.

The goal is to build a machine learning classification model that predicts the species of an iris flower from its sepal and petal measurements.

### Dataset
The supplied `Iris.csv` dataset contains 150 observations from three iris species:

- Iris-setosa
- Iris-versicolor
- Iris-virginica

The model uses these four measurements:
- Sepal Length
- Sepal Width
- Petal Length
- Petal Width

The `Id` column is only an identifier and is not used as a model feature.

## Approach
1. Loaded the dataset using Pandas.
2. Checked the dataset structure and class distribution.
3. Separated the features from the target variable.
4. Split the data into 80% training and 20% testing sets.
5. Standardized the numerical features.
6. Trained a Logistic Regression classification model using Scikit-learn.
7. Evaluated the model using accuracy, a classification report, and a confusion matrix.
8. Created a scatter plot to visualize the relationship between petal length and petal width.

## Result
Using `random_state=42` and a stratified 80/20 train-test split, the model achieved:

**Test Accuracy: 93.33%**

The confusion matrix shows that all 10 setosa flowers in the test set were classified correctly. One versicolor was classified as virginica and one virginica was classified as versicolor.

## Files
- `iris_classification.ipynb` — complete notebook with analysis and results
- `iris_classification.py` — Python script version
- `Iris.csv` — supplied dataset
- `petal_scatter.png` — feature visualization
- `confusion_matrix.png` — model evaluation plot
- `requirements.txt` — required Python libraries

## Libraries
- Pandas
- Matplotlib
- Scikit-learn

## Conclusion
The model was able to classify the three iris species with high accuracy using only four flower measurements. Petal measurements provide a clear separation between the species, while some overlap can occur between versicolor and virginica.
