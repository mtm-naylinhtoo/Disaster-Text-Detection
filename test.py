from joblib import load
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# Load the dataset
data = pd.read_csv('dataset.csv')

# Preprocess the text data (assuming you've done it similarly during training)
# Replace missing values, remove special characters, lowercase, tokenize, etc.

# Transform text using CountVectorizer
cv = load('count.pkl')
X = cv.transform(data['text'].values)
# print("Logistic Regression Accuracy:", logreg_accuracy)
# print("Multinomial Naive Bayes Accuracy:", nb_accuracy)
# print("Support Vector Machine (SVM) Accuracy:", svm_accuracy)
# Load each trained model
models = ['multinomial.pkl', 'logistic_reg.pkl', 'svm.pkl']
for model_file in models:
    model = load(model_file)

    # Make predictions
    predictions = model.predict(X)

    # Calculate accuracy
    accuracy = accuracy_score(data['target'], predictions)
    precision = precision_score(data['target'], predictions)
    recall = recall_score(data['target'], predictions)
    f1 = f1_score(data['target'], predictions)

    print(f"\n{model_file}\n")
    print(f"Accuracy: {accuracy}")
    print(f"Precision: {precision}")
    print(f"Recall: {recall}")
    print(f"F1 Score: {f1}\n")
