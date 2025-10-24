import dill

with open("src/model_files/spam_detector.pkl", 'rb') as f:
	data = dill.load(f)

model = data['model']
vectorizer = data['vectorizer']
norm = data['norm']

text = "open link for free money"

clean_text = norm(text)
vect_text = vectorizer.transform([clean_text])

prediction = model.predict(vect_text)[0]
print(f'Prediction: {prediction}')
