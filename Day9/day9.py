import numpy as np


# --------------------------------------------------
# 1. Create document dataset
# --------------------------------------------------

documents = {
    "Python Basics": {
        "text": "Python is a popular programming language used for software development.",
        "vector": [8, 7, 2, 1, 1]
    },

    "Python Data Science": {
        "text": "Python is widely used for data analysis, NumPy, Pandas and data science.",
        "vector": [9, 8, 3, 1, 1]
    },

    "Artificial Intelligence": {
        "text": "Artificial intelligence enables computers to perform tasks that require human intelligence.",
        "vector": [7, 8, 2, 1, 2]
    },

    "Machine Learning": {
        "text": "Machine learning allows computers to learn patterns from data and make predictions.",
        "vector": [6, 9, 2, 1, 2]
    },

    "Deep Learning": {
        "text": "Deep learning uses neural networks to solve complex artificial intelligence problems.",
        "vector": [5, 9, 2, 1, 2]
    },

    "Football": {
        "text": "Football is a popular sport played by two teams trying to score goals.",
        "vector": [1, 1, 9, 2, 1]
    },

    "Cricket": {
        "text": "Cricket is a team sport involving batting, bowling and fielding.",
        "vector": [1, 1, 8, 3, 1]
    },

    "Travel": {
        "text": "Travel allows people to visit new places and experience different cultures.",
        "vector": [1, 1, 1, 9, 7]
    },

    "Tourism": {
        "text": "Tourism involves visiting destinations, exploring attractions and experiencing cultures.",
        "vector": [1, 1, 1, 8, 8]
    },

    "Adventure Travel": {
        "text": "Adventure travel includes hiking, camping and exploring exciting destinations.",
        "vector": [1, 1, 2, 8, 9]
    }
}


# --------------------------------------------------
# 2. Display documents
# --------------------------------------------------

def display_documents():

    print("\n===== DOCUMENTS =====")

    for name, document in documents.items():

        print(f"\n{name}")
        print("Text:", document["text"])
        print("Vector:", document["vector"])


# --------------------------------------------------
# 3. Display vector dimensions
# --------------------------------------------------

def display_dimensions():

    vectors = np.array([
        document["vector"]
        for document in documents.values()
    ])

    print("\n===== VECTOR INFORMATION =====")

    print("Number of documents:", vectors.shape[0])
    print("Vector dimensions:", vectors.shape[1])
    print("Dataset shape:", vectors.shape)


# --------------------------------------------------
# 4. Calculate cosine similarity
# --------------------------------------------------

def cosine_similarity(vector1, vector2):

    vector1 = np.array(vector1, dtype=float)
    vector2 = np.array(vector2, dtype=float)

    dot_product = np.dot(vector1, vector2)

    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    similarity = dot_product / (magnitude1 * magnitude2)

    return similarity


# --------------------------------------------------
# 5. Accept query vector
# --------------------------------------------------

def get_query_vector():

    print("\n===== QUERY VECTOR =====")

    print("Enter 5 numerical values.")
    print("Example: 8 8 2 1 1")

    while True:

        try:

            values = input("Enter query vector: ").split()

            query_vector = np.array(
                [float(value) for value in values]
            )

            if len(query_vector) != 5:
                print("Please enter exactly 5 values.")
                continue

            return query_vector

        except ValueError:

            print("Please enter numbers only.")


# --------------------------------------------------
# 6. Compare query with all documents
# --------------------------------------------------

def calculate_similarities(query_vector):

    results = []

    for name, document in documents.items():

        document_vector = document["vector"]

        similarity = cosine_similarity(
            query_vector,
            document_vector
        )

        results.append(
            (name, similarity)
        )

    return results


# --------------------------------------------------
# 7. Rank documents
# --------------------------------------------------

def rank_documents(results):

    ranked_results = sorted(
        results,
        key=lambda x: x[1],
        reverse=True
    )

    return ranked_results


# --------------------------------------------------
# 8. Display Top 3
# --------------------------------------------------

def display_top_3(ranked_results):

    print("\n===== TOP 3 MOST SIMILAR DOCUMENTS =====")

    for rank, (name, similarity) in enumerate(
        ranked_results[:3],
        start=1
    ):

        print(
            f"{rank}. {name:<25} "
            f"Similarity: {similarity:.4f}"
        )


# --------------------------------------------------
# 9. Display all ranked documents
# --------------------------------------------------

def display_all_results(ranked_results):

    print("\n===== ALL DOCUMENTS RANKED =====")

    for rank, (name, similarity) in enumerate(
        ranked_results,
        start=1
    ):

        print(
            f"{rank}. {name:<25} "
            f"Similarity: {similarity:.4f}"
        )


# --------------------------------------------------
# 10. Create similarity matrix
# --------------------------------------------------

def create_similarity_matrix():

    names = list(documents.keys())

    vectors = np.array([
        document["vector"]
        for document in documents.values()
    ])

    matrix = np.zeros(
        (len(vectors), len(vectors))
    )

    for i in range(len(vectors)):

        for j in range(len(vectors)):

            matrix[i][j] = cosine_similarity(
                vectors[i],
                vectors[j]
            )

    print("\n===== SIMILARITY MATRIX =====")

    print("Document names:")
    for i, name in enumerate(names):
        print(f"{i}: {name}")

    print("\nMatrix:")

    np.set_printoptions(
        precision=3,
        suppress=True
    )

    print(matrix)

    return matrix


# --------------------------------------------------
# 11. Main program
# --------------------------------------------------

def main():

    print("====================================")
    print("     DOCUMENT SIMILARITY ANALYZER")
    print("====================================")

    display_documents()

    display_dimensions()

    query_vector = get_query_vector()

    results = calculate_similarities(
        query_vector
    )

    ranked_results = rank_documents(
        results
    )

    display_all_results(
        ranked_results
    )

    display_top_3(
        ranked_results
    )

    create_similarity_matrix()


# --------------------------------------------------
# Run program
# --------------------------------------------------

if __name__ == "__main__":
    main()