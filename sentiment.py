import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, LSTM, Dense, Dropout



nltk.download('stopwords')
nltk.download('wordnet')

data = pd.read_csv('IMDB Dataset.csv').sample(5000, random_state=42)

data['sentiment'] = data['sentiment'].map({'positive': 1, 'negative': 0})

def clean_text(text):
    text = re.sub(r'<[^>]+>|[^a-zA-Z]', ' ', text).lower().split()
    lemmatizer = WordNetLemmatizer()
    return ' '.join(lemmatizer.lemmatize(word) for word in text 
                  if word not in stopwords.words('english'))

data['cleaned_review'] = data['review'].apply(clean_text)


tokenizer = Tokenizer(num_words=5000, oov_token='<OOV>')
tokenizer.fit_on_texts(data['cleaned_review'])
sequences = tokenizer.texts_to_sequences(data['cleaned_review'])

padded_sequences = pad_sequences(sequences, maxlen=200, truncating='post')

X_train, X_test, y_train, y_test = train_test_split(padded_sequences, data['sentiment'], test_size=0.2, random_state=42)




model = Sequential([
    Embedding(5000, 64, input_length=200),  
    LSTM(64, dropout=0.2),  
    Dense(32, activation='relu'), 
    Dropout(0.3),
    Dense(1, activation='sigmoid')
])

model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

history = model.fit(X_train, y_train, epochs=5, batch_size=128, validation_data=(X_test, y_test))



loss, accuracy = model.evaluate(X_test, y_test)
print(f"Test Accuracy: {accuracy*100:.2f}%")

sample_text = "This movie was fantastic! I really loved it."
cleaned_sample = clean_text(sample_text)
sample_seq = tokenizer.texts_to_sequences([cleaned_sample])
padded_sample = pad_sequences(sample_seq, maxlen=200)
prediction = model.predict(padded_sample)
print(f"Prediction Score: {prediction[0][0]}")  



model.save('sentiment_analysis_model.h5')  