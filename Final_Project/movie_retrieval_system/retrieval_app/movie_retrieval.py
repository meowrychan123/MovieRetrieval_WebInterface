import os
import math
import numpy as np
import nltk
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import os
import logging
import re

nltk.download('stopwords', quiet=True)
nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)

STOPWORDS = set(stopwords.words('english'))
LEMMATIZER = WordNetLemmatizer()



def load_movies(folder_path):
    data = {}
    doc_id_to_filename = {}
    doc_id = 0
    
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), 'r', encoding='utf-8') as file:
                content = file.read()
                
                # Parse the title, IMDb rating, and summary
                lines = content.split('\n')
                title = lines[0].replace('Title: ', '').strip()  # Get the title
                rating = lines[1].replace('IMDb Rating: ', '').strip()  # Get the rating
                summary = '\n'.join(lines[3:]).strip()  # Get the summary, skipping empty lines
                
                # Store in the data dictionary
                data[doc_id] = {'title': title, 'rating': rating, 'summary': summary}
                doc_id_to_filename[doc_id] = filename
                logging.info(f"Loaded file: {filename} with doc_id: {doc_id}")
                doc_id += 1
                
    return data, doc_id_to_filename

def tokenize(text):
    return text.lower().split()

def clean_text(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    tokens = tokenize(text)
    cleaned_tokens = [LEMMATIZER.lemmatize(word) for word in tokens if word not in STOPWORDS]
    return cleaned_tokens

def term_frequency(term, document):
    return document.count(term) / len(document)


def inverse_document_frequency(term, all_documents):
    num_docs_containing_term = sum(1 for doc in all_documents if term in doc)
    return math.log(len(all_documents) / (1 + num_docs_containing_term))


def compute_tfidf(document, all_documents, vocab):
    tfidf_vector = []
    for term in vocab:
        tf = term_frequency(term, document)
        idf = inverse_document_frequency(term, all_documents)
        tfidf_vector.append(tf * idf)
    return np.array(tfidf_vector)


def cosine_similarity(vec1, vec2):
    dot_product = np.dot(vec1, vec2)
    norm_vec1 = np.linalg.norm(vec1)
    norm_vec2 = np.linalg.norm(vec2)
    return dot_product / (norm_vec1 * norm_vec2) if norm_vec1 * norm_vec2 != 0 else 0

def process_queries(query, all_documents, doc_tfidf_vectors, vocab, top_k=5):
    tokenized_query = clean_text(query)
    query_vector = compute_tfidf(tokenized_query, all_documents, vocab)

    similarities = []
    for doc_id, doc_vector in enumerate(doc_tfidf_vectors):
        similarity = cosine_similarity(query_vector, doc_vector)
        similarities.append((doc_id, similarity))

    similarities.sort(key=lambda x: x[1], reverse=True)

    return similarities[:top_k]


def convert_doc_ids_to_filenames(doc_ids, doc_id_to_filename):
    return [doc_id_to_filename[doc_id] for doc_id in doc_ids]

