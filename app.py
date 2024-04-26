from flask import Flask
from flask import render_template
from flask import request
from flask import jsonify
from flask import redirect
from flask import url_for
from nltk.stem import WordNetLemmatizer
import re
import nltk
from nltk.corpus import stopwords
import numpy as np
import string
# from sklearn.externals import joblib
from joblib import load, dump  # Add this line to import joblib
from sklearn.feature_extraction.text import CountVectorizer
from unidecode import unidecode
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
import contractions
import openai
import pandas as pd
import scipy.sparse

nltk.download('stopwords')
nltk.download('wordnet')
#nltk.download('WordNetLemmatizer')

app = Flask(__name__)
# Set your OpenAI API key
openai.api_key = 'enter your open api key here'

@app.route('/')
def index():
    return render_template("index.html" )

@app.route("/api", methods=["GET","POST"])
def api():
    if request.method == "POST":
        cv = load('count.pkl')
        model = load('logistic_reg.pkl')

        tweet = request.form["tweet"]
        text = tweet

        text = text.lower()
        text = text.split()
        p=[]
        for word in text:
            p.append(contractions.fix(word))
        t=[' '.join(p)]
        text=t[0]

        text = re.sub("[^a-zA-Z]", ' ', text)
        text=re.sub("https\S+"," ",text)
        text=re.sub("http\S+"," ",text)
        text=re.sub("\W"," ",text)
        text=re.sub("\d"," ",text)
        text=re.sub("\_","",text)
        text = unidecode(text)

        text = text.split()



        nonpunc=[]
        for c in text:
            if c not in string.punctuation:
                nonpunc.append(c)
        text = nonpunc


        """
        for word in text:
            if  word in set(stopwords.words('english')):
                text.append(PorterStemmer.stem(word))"""

        lemmatizer=WordNetLemmatizer()
        L=[lemmatizer.lemmatize(word) for word in text]

        text = [' '.join(text)]
        print("before text: ")
        print(text)
        X=cv.transform(text)

        def update_model(prediction, openai_response):
            print('before update')
            print(prediction[0])
            print(openai_response)
            if prediction[0] != openai_response:
                print("update")


                # Create DataFrame with new data
                new_data = pd.DataFrame({'text': text, 'target': [openai_response]})
                # Check if the CSV file exists
                try:
                    # If CSV exists, append new data
                    existing_data = pd.read_csv('train.csv')
                    updated_data = pd.concat([existing_data, new_data], ignore_index=True)
                    updated_data.to_csv('train.csv', index=False)
                except FileNotFoundError:
                    # If CSV doesn't exist, create it with new data
                    new_data.to_csv('test.csv', index=False)


                # Retrain the model using the collected data point
                # Load the dataset
                data = pd.read_csv('train.csv')

                # Extract input text (X_train) and labels (y_train) from the dataset
                X_train = data['text'].values
                y_train = data['target'].values
                # Load existing CountVectorizer and MultinomialNB model
                cv = load('count.pkl')


                # Transform input text using CountVectorizer
                X_new = cv.transform(text)

                # Update model with new data
                # X_combined = np.vstack([X_train, X_new])  # Combine old and new data
                # y_combined = np.append(y_train, openai_response)  # Combine old and new labels
                print("retraning")
                print(text)
                print(openai_response)
                # Transform existing training data using CountVectorizer
                X_train_transformed = cv.transform(X_train)
                print("x before and after")
                print(X_train_transformed)
                print(X_new)

                # Combine old and new data
                X_combined = scipy.sparse.vstack([X_train_transformed, X_new])
                print("x combined")
                print(X_combined)
                # Combine old and new labels
                y_combined = np.append(y_train, openai_response)
                print("y combined")
                print(y_combined)
                new_model = MultinomialNB()
                # new_model = LogisticRegression()
                # new_model = SVC()
                # Retrain the model using updated data
                new_model.fit(X_combined, y_combined)

                # Save the updated model
                dump(new_model, 'logistic_reg.pkl')


        # Make API request to OpenAI API
        # Construct the prompt

        prompt = f"Is this string related to a natural disaster? (Yes or No)\nString: {text}"
        response = openai.Completion.create(
            engine="gpt-3.5-turbo-instruct",
            prompt=prompt,
            max_tokens=30
        )

        openai_response_text = response.choices[0].text.strip()
        print("openai")
        print(response)
        openai_response = 1 if openai_response_text.lower() == "yes" else 0
        print("openai")
        print(openai_response)
        # openai_response = 1

        # Update model based on prediction and OpenAI API response

        prediction = model.predict(X)
        update_model(prediction, openai_response)

        if prediction == 1:
            msg = "Real disaster!!"
            return render_template("index.html", msg=msg, tweet=tweet)
        else:
            error = "Fake disaster!!"
            return render_template("index.html", error=error, tweet=tweet)
    else:
        return redirect(url_for("index"))



if __name__ == '__main__':
    app.run(debug=True)

#web: gunicorn app:app
