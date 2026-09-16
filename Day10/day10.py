from sentence_transformers import SentenceTransformer
import numpy as np


# --------------------------------------------------
# 1. Create dataset
# --------------------------------------------------

sentences = [
    # Programming
    "Python is a popular programming language.",
    "Java is widely used for software development.",
    "Programming allows developers to create computer applications.",
    "Python can be used to build web applications.",
    "Data structures are important in computer programming.",
    "Learning algorithms helps improve programming skills.",

    # Artificial Intelligence
    "Artificial intelligence enables machines to perform intelligent tasks.",
    "Machine learning is a branch of artificial intelligence.",
    "Deep learning uses neural networks to learn from data.",
    "AI can be used for image recognition and classification.",
    "Natural language processing helps computers understand human language.",
    "Artificial intelligence is transforming many industries.",

    # Education
    "Students can improve their knowledge through regular study.",
    "Online education allows students to learn from anywhere.",
    "Universities provide students with opportunities for higher education.",
    "Reading books can improve learning and knowledge.",
    "Practical exercises help students understand difficult concepts.",
    "Good study habits can improve academic performance.",

    # Business
    "Businesses use data analysis to make better decisions.",
    "Marketing helps companies promote their products and services.",
    "Customer satisfaction is important for business success.",
    "Companies use technology to improve business operations.",
    "Entrepreneurs create new businesses to solve customer problems.",
    "Financial planning helps businesses manage their resources.",

    # Travel
    "Traveling allows people to explore new countries and cultures.",
    "Japan is a popular destination for international travelers.",
    "Tourists enjoy visiting historical places and landmarks.",
    "Travel planning helps people organize their holidays.",
    "Sri Lanka offers beautiful beaches and cultural attractions.",
    "Adventure tourism includes activities such as hiking and camping."
]


# --------------------------------------------------
# 2. Load embedding model
# --------------------------------------------------

print("Loading embedding model...")

model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Model loaded successfully.")


# --------------------------------------------------
# 3. Generate embeddings for all sentences
# --------------------------------------------------

print("\nGenerating sentence embeddings...")

embeddings = model.encode(
    sentences,
    convert_to_numpy=True
)

print("Embeddings generated successfully.")

print("Number of sentences:", len(sentences))
print("Embedding shape:", embeddings.shape)


# --------------------------------------------------
# 4. Generate query embedding
# --------------------------------------------------

def generate_query_embedding(query):

    query_embedding = model.encode(
        query,
        convert_to_numpy=True
    )

    return query_embedding


# --------------------------------------------------
# 5. Calculate cosine similarity
# --------------------------------------------------

def cosine_similarity(vector1, vector2):

    dot_product = np.dot(vector1, vector2)

    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    similarity = dot_product / (
        magnitude1 * magnitude2
    )

    return similarity


# --------------------------------------------------
# 6. Compare query with all sentences
# --------------------------------------------------

def find_similar_sentences(query_embedding):

    results = []

    for i, sentence_embedding in enumerate(embeddings):

        similarity = cosine_similarity(
            query_embedding,
            sentence_embedding
        )

        results.append(
            (sentences[i], similarity)
        )

    return results


# --------------------------------------------------
# 7. Rank results
# --------------------------------------------------

def rank_results(results):

    ranked_results = sorted(
        results,
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_results


# --------------------------------------------------
# 8. Display Top 5
# --------------------------------------------------

def display_top_5(ranked_results):

    print("\n======================================")
    print("       TOP 5 SEMANTIC MATCHES")
    print("======================================")

    for rank, (sentence, similarity) in enumerate(
        ranked_results[:5],
        start=1
    ):

        print(
            f"\n{rank}. {sentence}"
        )

        print(
            f"   Similarity: {similarity:.4f}"
        )


# --------------------------------------------------
# 9. Main program
# --------------------------------------------------

def main():

    print("======================================")
    print("       SEMANTIC SENTENCE MATCHER")
    print("======================================")

    print("\nDataset contains", len(sentences), "sentences.")

    # Accept query
    query = input(
        "\nEnter your query: "
    )

    print("\nQuery:", query)

    # Generate query embedding
    query_embedding = generate_query_embedding(
        query
    )

    # Calculate similarities
    results = find_similar_sentences(
        query_embedding
    )

    # Rank results
    ranked_results = rank_results(
        results
    )

    # Display Top 5
    display_top_5(
        ranked_results
    )


# --------------------------------------------------
# Run program
# --------------------------------------------------

if __name__ == "__main__":
    main()