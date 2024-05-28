import csv
import numpy as np
from nltk import word_tokenize
from gensim.models import Word2Vec
from rouge import Rouge
from unidecode import unidecode

# Reading the data and removing the diacritics
def read_csv_to_list_of_lists(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        csvreader = csv.reader(csvfile)
        for row in csvreader:
            row = [unidecode(cell) for cell in row]
            data.append(row)
    return data

# Pad texts to the same length
def pad_texts(texts, max_length):
    padded_texts = []
    for text in texts:
        if len(text) > max_length:
            padded_texts.append(text[:max_length])
        else:
            padded_texts.append(text + ' ' * (max_length - len(text)))
    return padded_texts

# Train or load Word2Vec model
def train_word2vec_model(sentences, vector_size=100, window=5, min_count=1, workers=4):
    model = Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    return model

# Transform content using Word2Vec
def transform_content_to_word2vec(content, model):
    words = word_tokenize(content.lower())
    words = [word for word in words if word in model.wv.key_to_index]
    if not words:
        return np.zeros((model.vector_size,))
    word_vectors = np.array([model.wv[word] for word in words])
    return np.mean(word_vectors, axis=0)

# Generate summary using Word2Vec vectors
def generate_summary(text, model, top_n=5):
    sentences = text.split('.')
    sentence_vectors = [transform_content_to_word2vec(sentence, model) for sentence in sentences]
    sentence_scores = [np.linalg.norm(vec) for vec in sentence_vectors]
    top_sentence_indices = np.argsort(sentence_scores)[-top_n:]
    summary = '. '.join([sentences[i] for i in sorted(top_sentence_indices)])
    return summary

# Evaluate summaries
def evaluate_summaries(generated_summaries, reference_summaries):
    rouge = Rouge()
    scores = rouge.get_scores(generated_summaries, reference_summaries, avg=True)
    return scores

# Main workflow
file_path = 'ro_text_summarization_clean.csv'
rows = read_csv_to_list_of_lists(file_path)

# Extract content column (assuming it's the 3rd column)
contents = [row[2] for row in rows]

# Determine the maximum length of texts
max_length = max(len(content) for content in contents)

# Pad texts to the maximum length
padded_contents = pad_texts(contents, max_length)

# Tokenize the padded contents for Word2Vec training
tokenized_contents = [word_tokenize(content.lower()) for content in padded_contents]

# Train Word2Vec model
model = train_word2vec_model(tokenized_contents)

# Process each padded content for Word2Vec and summary generation
summaries = []
for text in padded_contents:
    summary = generate_summary(text, model)
    summaries.append(summary)

# Extract reference summaries (assuming they're in the 4th column)
reference_summaries = [row[3] for row in rows]

# Evaluate summaries
evaluation_scores = evaluate_summaries(summaries, reference_summaries)
print("ROUGE scores:", evaluation_scores)
