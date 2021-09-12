import nltk
import re
from flask import Flask, render_template, request
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
import joblib
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# nltk.download("stopwords")
# nltk.download("wordnet")
from sklearn.model_selection import train_test_split

app = Flask(__name__)
tfv = TfidfVectorizer(max_features=50000, lowercase=False, ngram_range=(1, 2))
real_news = pd.read_csv("True.csv")
fake_news = pd.read_csv("Fake.csv")
real_news["authenticity"] = 0
fake_news["authenticity"] = 1
dataset_true = real_news[["text", "authenticity"]]
dataset_fake = fake_news[["text", "authenticity"]]
dataset = pd.concat([dataset_true, dataset_fake])
dataset = dataset.sample(frac=1)
wnl = WordNetLemmatizer()
stopwords = stopwords.words("english")


def clean_news(news):
    news = news.lower()
    news = re.sub("[^a-zA-Z]", " ", news)
    single_words = news.split()
    lemmatized_news = [wnl.lemmatize(word) for word in single_words if not word in stopwords]
    final_news = " ".join(lemmatized_news)
    return final_news


dataset["text"] = dataset["text"].apply(lambda x: clean_news(x))
X = dataset.iloc[:5000, 0]
y = dataset.iloc[:5000, 1]
train_X, test_X, train_y, test_y = train_test_split(X, y, test_size=0.2, random_state=0)
# print(train_X)
vec_train_X = tfv.fit_transform(train_X).toarray()
# vec_test_X = tfv.transform(test_X).toarray()

model = joblib.load("multinomialNB.pkl")


def fake_news_pred(news):
    # vec_train_X = tfv.fit_transform(train_X).toarray()
    # vec_test_X = tfv.transform(test_X).toarray()
    # training_data = pd.DataFrame(vec_train_X, columns=tfv.get_feature_names())
    # test_data = pd.DataFrame(vec_test_X, columns=tfv.get_feature_names())
    input_news = [str(news)]
    vec_input_news = tfv.transform(input_news)
    prediction = model.predict(vec_input_news)
    return prediction


@app.route("/")
def home_page():
    return render_template("fakehome.html")


@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        message = request.form["message"]
        pred = fake_news_pred(message)
        print(pred)
        return render_template("fakehome.html", prediction=pred)
    else:
        return render_template("fakehome.html", prediction="Something went wrong")


if __name__ == "__main__":
    app.run(debug=True)
