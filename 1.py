from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

# Load model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Your data (database)
sentences = [
    "I love playing football",
    "Dogs are very loyal animals",
    "Python is a great programming language",
    "I enjoy watching movies",
    "Cats are independent pets"
]

# Convert to embeddings
embeddings = model.encode(sentences)

# User query
query = "Which animal is loyal?"
query_embedding = model.encode([query])

# Compute similarity
similarities = cosine_similarity(query_embedding, embeddings)

# Get best match
best_index = similarities.argmax()

print("Query:", query)
print("Best match:", sentences[best_index])
