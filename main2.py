import csv

import Rouge
from nltk import word_tokenize, sent_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from unidecode import unidecode


#reading the data and removing the diacritics
def read_csv_to_list_of_lists(file_path):
    # Initialize an empty list to hold the rows
    data = []

    # Open the CSV file
    with open(file_path, 'r', encoding='utf-8') as csvfile:
        # Create a CSV reader object
        csvreader = csv.reader(csvfile)
        # Iterate over each row in the CSV file
        for row in csvreader:
            row = [unidecode(cell) for cell in row] #remove diacritics
            # Append the row to the data list
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


# function that transforms a content, sentence by sentence in a tf-idf array
def transform_content_to_tfidf(content):
    # Lista de stop words personalizată în limba română
    custom_stop_words = {
        'am','si', 'sau', 'dar', 'de', 'pe', 'in', 'la', 'cu', 'al', 'un', 'o', 'mai', 'care', 'dupa','nu','poate','acesta','am','ne','se','le',
        'ca', 'din', 'pentru', 'este', 'acest', 'aceasta', 'acesti', 'aceste', 'acel', 'acele', 'fiecare','sa',
        'unei', 'unui', 'acele', 'ale', 'acestui', 'acelei', 'aceștia', 'aceleia', 'acestor', 'acestea', 'acestora'
        # Adaugă alte cuvinte considerate stop words
    }

    # Tokenizează conținutul în propoziții
    sentences = content.split('.')

    # Initializează lista pentru a stoca prelucrarea TF-IDF pentru fiecare propoziție
    tfidf_results = []

    # Pentru fiecare propoziție
    for sentence in sentences:
        # Tokenizează propoziția în cuvinte
        tokens = sentence.split()
        # Elimină stop words-urile și transformă cuvintele în litere mici
        words = [word.lower() for word in tokens if word.lower() not in custom_stop_words]
        # Reconstruiește propoziția folosind cuvintele prelucrate
        processed_sentence = ' '.join(words)
        # Adaugă propoziția prelucrată în lista rezultatelor TF-IDF
        tfidf_results.append(processed_sentence)

    # Initializează TfidfVectorizer
    vectorizer = TfidfVectorizer()

    # Transformă propozițiile prelucrate în matrice TF-IDF
    tfidf_matrix = vectorizer.fit_transform(tfidf_results)

    return tfidf_matrix, vectorizer


# PT FILIP SI CASETA CARE O SA FACA ECHIPA SI O SA REZOLVE ACEASTA DILEMA !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
def generate_summary(text, top_terms):
    #TO DO : implementare functie de generare a rezumatelor
    return None


# functie de evaluare a rezumatelor
def evaluate_summaries(generated_summaries, reference_summaries):
    rouge = Rouge()
    scores = rouge.get_scores(generated_summaries, reference_summaries, avg=True)
    return scores


expected_headers = ["Category", "Title", "Content", "Summary", "href", "Source"] #in caz ca aveti nevoie
file_path = 'ro_text_summarization_clean.csv'
rows = read_csv_to_list_of_lists(file_path)

# Transform the content to TF-IDF vectors
# exemplu utilizare - pentru o linie extrag content (col 2) il impart in cuvinte string ul
contents = [row[2] for row in rows]

# Modificare in workflow aduc textele la aceeasi lungime
# Determinam lungimea celui mai lung text
max_length = max(len(content) for content in contents)

# Aducem toate textele la aceeasi lungime
padded_contents = pad_texts(contents, max_length)


text = contents[0]
tfidf_matrix, vectorizer = transform_content_to_tfidf(text)
print("Matricea TF-IDF:\n", tfidf_matrix.toarray())
print("Numele caracteristicilor (cuvintelor):\n", vectorizer.get_feature_names_out())
# Obține vectorul TF-IDF pentru text
tfidf_vector = tfidf_matrix.toarray()
vocab = vectorizer.get_feature_names_out()
# Calculează suma valorilor TF-IDF pentru fiecare termen din text
term_scores = tfidf_vector.sum(axis=0)

# Sortează indexurile termenilor în funcție de scorurile lor TF-IDF
top_term_indices = term_scores.argsort()[::-1][:25]

# Extrage cele mai frecvente cuvinte din vocabular folosind indexurile
top_terms = [vocab[i] for i in top_term_indices]

# Afișează cele mai frecvente cuvinte
print("Cele mai frecvente 25 cuvinte din vocabular:")
print(top_terms)


#-----------------------------------------------------
#mai trebuie:
# asta am facut eu marian : sa facem textele de aceeasi lungime (eventual le umplem cu spatii libere sau ceva) cand le predam unui tool creat de noi sa faca texte pe baza cuvintelor cheie si a textului
#sa facem tool si sa calculam acuratetea in raport cu a 4 a coloana (unde avem rezumatele) - fie le comparam pe baza sentimentului fie altceva


# Main workflow
file_path = 'ro_text_summarization_clean.csv'
rows = read_csv_to_list_of_lists(file_path)

# Extract content column (assuming it's the 3rd column)
contents = [row[2] for row in rows]

# Step 1: Determine the maximum length of texts
max_length = max(len(content) for content in contents)

# Step 2: Pad texts to the maximum length
padded_contents = pad_texts(contents, max_length)

# Process each padded content for TF-IDF and summary generation
summaries = []
for text in padded_contents:
    # Step 3: Transform content to TF-IDF
    tfidf_matrix, vectorizer = transform_content_to_tfidf(text)
    tfidf_vector = tfidf_matrix.toarray()
    vocab = vectorizer.get_feature_names_out()
    term_scores = tfidf_vector.sum(axis=0)
    top_term_indices = term_scores.argsort()[::-1][:25]
    top_terms = [vocab[i] for i in top_term_indices]

    # Step 4: Generate summary
    summary = generate_summary(text, top_terms)
    summaries.append(summary)

# Extract reference summaries (assuming they're in the 4th column)
reference_summaries = [row[3] for row in rows]

# Step 5: Evaluate summaries
evaluation_scores = evaluate_summaries(summaries, reference_summaries)
print("ROUGE scores:", evaluation_scores)