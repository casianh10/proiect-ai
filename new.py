import pandas as pd
import numpy as np
import re
import unicodedata
import csv
from sklearn.model_selection import train_test_split
from gensim.models import Word2Vec
from nltk.tokenize import sent_tokenize, word_tokenize
from nltk.corpus import stopwords
from sklearn.metrics import mean_squared_error
import nltk
from unidecode import unidecode

nltk.download('punkt')
nltk.download('stopwords')


# Funcție pentru eliminarea diacriticelor
def remove_diacritics(text):
    return unidecode(text)


# Funcție pentru preprocesarea textului
def preprocess_text(text):
    text = remove_diacritics(text)
    text = re.sub(r'\d+', '', text)
    text = text.lower()
    text = re.sub(r'\W', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text


# Funcție pentru citirea fișierului CSV și eliminarea diacriticelor
def read_csv_to_list_of_lists(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            row = [unidecode(cell) for cell in row]
            data.append(row)
    return data


# Citește fișierul CSV
file_path = 'ro_text_summarization_clean.csv'
data = read_csv_to_list_of_lists(file_path)

# Creează DataFrame
df = pd.DataFrame(data, columns=['Category', 'Title', 'Content', 'Summary', 'href', 'Source'])

# Preprocesează textele și rezumatele
df['processed_content'] = df['Content'].apply(preprocess_text)
df['processed_summary'] = df['Summary'].apply(preprocess_text)

# Împarte datele în seturi de antrenament și test
train_df, test_df = train_test_split(df, test_size=0.2, random_state=42)

# Tokenizează textele
train_sentences = [word_tokenize(text) for text in train_df['processed_content'].tolist()]
train_summaries = [word_tokenize(summary) for summary in train_df['processed_summary'].tolist()]

# Antrenează modelul Word2Vec
word2vec_model = Word2Vec(sentences=train_sentences, vector_size=100, window=5, min_count=1, workers=4)


# Funcție pentru a genera un rezumat folosind modelul Word2Vec
def summarize(text, model, num_sentences=3):
    sentences = sent_tokenize(text)
    sentences = [word_tokenize(remove_diacritics(sentence.lower())) for sentence in sentences]
    sentence_vectors = [np.mean([model.wv[word] for word in sentence if word in model.wv] or [np.zeros(100)], axis=0)
                        for sentence in sentences]

    sentence_scores = [np.linalg.norm(vec) for vec in sentence_vectors]
    ranked_sentences = [sentences[i] for i in np.argsort(sentence_scores)[-num_sentences:]]
    summary = ' '.join([' '.join(sentence) for sentence in ranked_sentences])
    return summary


# Generarea rezumatelor pentru setul de test
test_texts = test_df['Content'].tolist()
test_processed_texts = test_df['processed_content'].tolist()
predicted_summaries = [summarize(text, word2vec_model) for text in test_processed_texts]

# Calculează performanța modelului folosind eroarea pătratică medie (MSE)
test_summaries = test_df['processed_summary'].tolist()
mse = mean_squared_error(test_summaries, predicted_summaries)
print(f'Mean Squared Error: {mse}')

# Salvează predicțiile într-un fișier CSV
output_df = pd.DataFrame({
    'original_text': test_texts,
    'original_summary': test_summaries,
    'predicted_summary': predicted_summaries
})
output_df.to_csv('predicted_summaries.csv', index=False)
