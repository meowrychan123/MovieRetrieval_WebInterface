from django.shortcuts import render
from .movie_retrieval import load_movies, process_queries, clean_text, compute_tfidf

# Load movies once when the server starts
folder_path = r"C:\Final_Project\movie_retrieval_system\Final_Dataset"

movies_data, doc_id_to_filename = load_movies(folder_path)

# Preprocess the documents
#tokenized_docs = [clean_text(doc) for doc in movies_data.values()]
tokenized_docs = [clean_text(movie['summary']) for movie in movies_data.values()]
vocab = sorted(set(word for doc in tokenized_docs for word in doc))
doc_tfidf_vectors = [compute_tfidf(doc, tokenized_docs, vocab) for doc in tokenized_docs]

def index(request):
    return render(request, 'index.html')

def results(request):
    query = request.GET.get('query', '')
    if query:
        similarities = process_queries(query, tokenized_docs, doc_tfidf_vectors, vocab)
        results = [
            {
                'filename': doc_id_to_filename[doc_id],
                'title': doc_id_to_filename[doc_id].replace('.txt', ''),
                'similarity': similarity
            } for doc_id, similarity in similarities
        ]
    else:
        results = []
    return render(request, 'results.html', {'results': results, 'query': query})

def movie_detail(request, filename):
    doc_id = list(doc_id_to_filename.keys())[list(doc_id_to_filename.values()).index(filename)]
    movie_content = movies_data[doc_id]
    return render(request, 'movie_detail.html', {
        'movie': {
            'title': movie_content['title'],
            'rating': movie_content['rating'],
            'summary': movie_content['summary']
        }
    })