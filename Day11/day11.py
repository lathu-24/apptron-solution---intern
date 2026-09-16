from sentence_transformers import SentenceTransformer
import numpy as np


# ==================================================
# 1. KNOWLEDGE BASE
# ==================================================

documents = [

    # ---------------- PYTHON ----------------

    {
        "id": 1,
        "category": "Python",
        "text": "Python is a high-level programming language known for its simple syntax."
    },

    {
        "id": 2,
        "category": "Python",
        "text": "Python is widely used for web development, automation, data science and artificial intelligence."
    },

    {
        "id": 3,
        "category": "Python",
        "text": "Python uses indentation to define blocks of code."
    },

    {
        "id": 4,
        "category": "Python",
        "text": "Python supports object-oriented programming and functional programming."
    },

    {
        "id": 5,
        "category": "Python",
        "text": "Python lists are ordered collections that can store multiple values."
    },

    {
        "id": 6,
        "category": "Python",
        "text": "Python dictionaries store data using key-value pairs."
    },

    {
        "id": 7,
        "category": "Python",
        "text": "Python functions are reusable blocks of code designed to perform specific tasks."
    },

    {
        "id": 8,
        "category": "Python",
        "text": "Python modules allow programmers to organize reusable code."
    },

    {
        "id": 9,
        "category": "Python",
        "text": "Python exception handling uses try and except blocks to handle errors."
    },

    {
        "id": 10,
        "category": "Python",
        "text": "Python is commonly used to build machine learning applications."
    },


    # ---------------- NUMPY ----------------

    {
        "id": 11,
        "category": "NumPy",
        "text": "NumPy is a Python library used for numerical computing."
    },

    {
        "id": 12,
        "category": "NumPy",
        "text": "NumPy provides powerful multidimensional arrays for numerical data."
    },

    {
        "id": 13,
        "category": "NumPy",
        "text": "NumPy arrays are generally faster and more efficient for numerical operations than Python lists."
    },

    {
        "id": 14,
        "category": "NumPy",
        "text": "NumPy provides mathematical functions for working with arrays."
    },

    {
        "id": 15,
        "category": "NumPy",
        "text": "NumPy supports vectorized operations that allow calculations without explicit loops."
    },

    {
        "id": 16,
        "category": "NumPy",
        "text": "NumPy broadcasting allows operations between arrays with compatible shapes."
    },

    {
        "id": 17,
        "category": "NumPy",
        "text": "NumPy can calculate statistical measures such as mean, median and standard deviation."
    },

    {
        "id": 18,
        "category": "NumPy",
        "text": "NumPy provides functions for reshaping and transposing arrays."
    },

    {
        "id": 19,
        "category": "NumPy",
        "text": "NumPy can generate random numbers for simulations and machine learning experiments."
    },

    {
        "id": 20,
        "category": "NumPy",
        "text": "NumPy is an important foundation for scientific computing in Python."
    },


    # ---------------- PANDAS ----------------

    {
        "id": 21,
        "category": "Pandas",
        "text": "Pandas is a Python library used for data analysis and data manipulation."
    },

    {
        "id": 22,
        "category": "Pandas",
        "text": "Pandas provides DataFrame and Series data structures."
    },

    {
        "id": 23,
        "category": "Pandas",
        "text": "Pandas can read data from CSV files."
    },

    {
        "id": 24,
        "category": "Pandas",
        "text": "Pandas can handle missing values in datasets."
    },

    {
        "id": 25,
        "category": "Pandas",
        "text": "Pandas allows users to filter rows based on conditions."
    },

    {
        "id": 26,
        "category": "Pandas",
        "text": "Pandas can group data and calculate summary statistics."
    },

    {
        "id": 27,
        "category": "Pandas",
        "text": "Pandas supports sorting and ranking data."
    },

    {
        "id": 28,
        "category": "Pandas",
        "text": "Pandas can merge multiple datasets using common columns."
    },

    {
        "id": 29,
        "category": "Pandas",
        "text": "Pandas provides tools for cleaning and preprocessing data."
    },

    {
        "id": 30,
        "category": "Pandas",
        "text": "Pandas DataFrames are useful for exploring structured datasets."
    },


    # ---------------- MACHINE LEARNING ----------------

    {
        "id": 31,
        "category": "Machine Learning",
        "text": "Machine learning allows computers to learn patterns from data."
    },

    {
        "id": 32,
        "category": "Machine Learning",
        "text": "Supervised learning uses labeled data to train machine learning models."
    },

    {
        "id": 33,
        "category": "Machine Learning",
        "text": "Unsupervised learning finds patterns in data without labeled target values."
    },

    {
        "id": 34,
        "category": "Machine Learning",
        "text": "Classification models predict categories or classes."
    },

    {
        "id": 35,
        "category": "Machine Learning",
        "text": "Regression models predict continuous numerical values."
    },

    {
        "id": 36,
        "category": "Machine Learning",
        "text": "Training data is used to teach a machine learning model."
    },

    {
        "id": 37,
        "category": "Machine Learning",
        "text": "Testing data is used to evaluate how well a machine learning model performs."
    },

    {
        "id": 38,
        "category": "Machine Learning",
        "text": "Feature engineering creates useful input variables for machine learning models."
    },

    {
        "id": 39,
        "category": "Machine Learning",
        "text": "Overfitting occurs when a model learns training data too closely."
    },

    {
        "id": 40,
        "category": "Machine Learning",
        "text": "Scikit-learn provides many tools for building machine learning models."
    },


    # ---------------- AI ----------------

    {
        "id": 41,
        "category": "AI",
        "text": "Artificial intelligence enables computers to perform tasks that normally require human intelligence."
    },

    {
        "id": 42,
        "category": "AI",
        "text": "AI systems can be used for image recognition."
    },

    {
        "id": 43,
        "category": "AI",
        "text": "Natural language processing allows computers to work with human language."
    },

    {
        "id": 44,
        "category": "AI",
        "text": "Computer vision enables machines to understand and analyze images."
    },

    {
        "id": 45,
        "category": "AI",
        "text": "Deep learning uses neural networks with multiple layers."
    },

    {
        "id": 46,
        "category": "AI",
        "text": "Generative AI can create text, images, audio and other types of content."
    },

    {
        "id": 47,
        "category": "AI",
        "text": "AI chatbots can understand questions and generate natural language responses."
    },

    {
        "id": 48,
        "category": "AI",
        "text": "AI recommendation systems can suggest products or content to users."
    },

    {
        "id": 49,
        "category": "AI",
        "text": "AI models often require large amounts of data for training."
    },

    {
        "id": 50,
        "category": "AI",
        "text": "Embeddings represent text as numerical vectors that capture semantic relationships."
    }
]


# ==================================================
# 2. LOAD EMBEDDING MODEL
# ==================================================

print("Loading embedding model...")

model = SentenceTransformer("all-MiniLM-L6-v2")

print("Model loaded successfully.")


# ==================================================
# 3. GENERATE DOCUMENT EMBEDDINGS
# ==================================================

print("\nGenerating document embeddings...")

texts = [
    document["text"]
    for document in documents
]

document_embeddings = model.encode(
    texts,
    convert_to_numpy=True
)

print("Embeddings generated.")
print("Number of documents:", len(documents))
print("Embedding shape:", document_embeddings.shape)


# ==================================================
# 4. COSINE SIMILARITY
# ==================================================

def cosine_similarity(vector1, vector2):

    dot_product = np.dot(vector1, vector2)

    magnitude1 = np.linalg.norm(vector1)
    magnitude2 = np.linalg.norm(vector2)

    if magnitude1 == 0 or magnitude2 == 0:
        return 0

    return dot_product / (
        magnitude1 * magnitude2
    )


# ==================================================
# 5. GENERATE QUERY EMBEDDING
# ==================================================

def generate_query_embedding(query):

    return model.encode(
        query,
        convert_to_numpy=True
    )


# ==================================================
# 6. SEARCH DOCUMENTS
# ==================================================

def search_documents(query, category=None):

    query_embedding = generate_query_embedding(query)

    results = []

    for i, document in enumerate(documents):

        # Category filter
        if category is not None:
            if document["category"].lower() != category.lower():
                continue

        similarity = cosine_similarity(
            query_embedding,
            document_embeddings[i]
        )

        results.append(
            {
                "id": document["id"],
                "category": document["category"],
                "text": document["text"],
                "similarity": similarity
            }
        )

    # Sort from highest similarity to lowest
    results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return results


# ==================================================
# 7. DISPLAY TOP 5 RESULTS
# ==================================================

def display_results(results):

    print("\n======================================")
    print("         TOP 5 SEARCH RESULTS")
    print("======================================")

    if not results:
        print("No matching documents found.")
        return

    for rank, result in enumerate(
        results[:5],
        start=1
    ):

        print(f"\n{rank}. {result['text']}")
        print(f"   Category: {result['category']}")
        print(f"   Similarity: {result['similarity']:.4f}")


# ==================================================
# 8. ADD NEW DOCUMENT
# ==================================================

def add_document():

    print("\n======================================")
    print("           ADD DOCUMENT")
    print("======================================")

    category = input(
        "Enter category: "
    ).strip()

    text = input(
        "Enter document text: "
    ).strip()

    if not category or not text:

        print("Category and text cannot be empty.")
        return

    new_id = len(documents) + 1

    new_document = {
        "id": new_id,
        "category": category,
        "text": text
    }

    documents.append(new_document)

    # Generate embedding for new document
    new_embedding = model.encode(
        text,
        convert_to_numpy=True
    )

    global document_embeddings

    document_embeddings = np.vstack(
        [
            document_embeddings,
            new_embedding
        ]
    )

    print("\nDocument added successfully.")


# ==================================================
# 9. DISPLAY CATEGORIES
# ==================================================

def display_categories():

    categories = sorted(
        set(
            document["category"]
            for document in documents
        )
    )

    print("\nAvailable categories:")

    for category in categories:
        print("-", category)


# ==================================================
# 10. MAIN MENU
# ==================================================

def main():

    while True:

        print("\n======================================")
        print("       MINI SEMANTIC SEARCH ENGINE")
        print("======================================")

        print("1. Search")
        print("2. Search by category")
        print("3. Add new document")
        print("4. Display categories")
        print("5. Exit")

        choice = input(
            "\nEnter your choice: "
        )

        if choice == "1":

            query = input(
                "\nEnter your search query: "
            )

            results = search_documents(
                query
            )

            display_results(
                results
            )

        elif choice == "2":

            display_categories()

            category = input(
                "\nEnter category: "
            )

            query = input(
                "Enter search query: "
            )

            results = search_documents(
                query,
                category
            )

            display_results(
                results
            )

        elif choice == "3":

            add_document()

        elif choice == "4":

            display_categories()

        elif choice == "5":

            print("\nExiting program...")
            break

        else:

            print("\nInvalid choice. Please try again.")


# ==================================================
# RUN PROGRAM
# ==================================================

if __name__ == "__main__":
    main()