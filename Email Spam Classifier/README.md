# Email Spam Classifier using Machine Learning 📧

## Overview

Welcome to the Email Spam Classifier project! This repository contains Python code that leverages machine learning techniques to classify emails as either spam or ham (non-spam). The model is trained on a dataset with approximately 97% accuracy, providing reliable predictions on unseen data.

## Components

1. **Data Preprocessing:**
   - The project begins with importing necessary libraries such as NumPy, Pandas, and scikit-learn.
   - The dataset (`mail_data.csv`) is loaded and preprocessed to handle missing values and categorize spam (0) and ham (1).

2. **Text Feature Extraction:**
   - Text data is converted into numerical features using the TF-IDF (Term Frequency-Inverse Document Frequency) vectorization technique.
   - The `TfidfVectorizer` is employed to transform the email messages into feature vectors.

3. **Model Training:**
   - The logistic regression algorithm is chosen for its effectiveness in binary classification tasks.
   - The dataset is split into training and testing sets using `train_test_split`.
   - The logistic regression model is trained on the features extracted from the training data.

4. **Evaluation:**
   - The accuracy of the model is evaluated on both the training and testing datasets using `accuracy_score`.
   - The achieved accuracy is ~97%, indicating the model's robust performance.

5. **Prediction:**
   - The trained model is then used to predict whether a given email (represented as input data features) is spam or ham.
   - Example mail provided in `input_your_mail` is classified and the result is printed.

## Usage

1. Ensure you have the required libraries installed (`numpy`, `pandas`, `scikit-learn`).
2. Run the provided Python script in your environment.
3. Observe the accuracy metrics and test the classifier on your own email samples.

Feel free to explore, modify, and contribute to this project!🎤🤖
Your feedback and suggestions are highly appreciated.🙏 
Happy coding! 🚀
