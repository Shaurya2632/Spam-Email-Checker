import pandas as pd
from sklearn.naive_bayes import MultinomialNB
from sklearn.feature_extraction.text import TfidfVectorizer

df = pd.read_csv("../../data/datasets/emails.csv")
X = df.iloc[:, 0]
y = df.iloc[:, 1]

def preprocess(text):
	import re
	from nltk.corpus import stopwords
	from nltk.stem import WordNetLemmatizer

	stop_words = set(stopwords.words('english'))
	lemmatizer = WordNetLemmatizer()

	text = text.lower()  # lowercase
	text = re.sub(r'[^a-z0-9\s]', '', text)
	text = re.sub(r'\s+', ' ', text).strip()

	words = [lemmatizer.lemmatize(word) for word in text.split() if word not in stop_words]
	return ' '.join(words)

X = X.apply(preprocess)

vectorizer = TfidfVectorizer(ngram_range=(1, 2))
X = vectorizer.fit_transform(X)

model = MultinomialNB(alpha = 0.1)
model.fit(X, y)

toPredict = 'get free money'
toPredict_clean = preprocess(toPredict)
toPredict_vect = vectorizer.transform([toPredict_clean])

score = round(model.score(X, y) * 100, 2)
prediction = model.predict(toPredict_vect)[0]
