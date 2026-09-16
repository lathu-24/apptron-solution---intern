import numpy as np
import matplotlib.pyplot as plt


# --------------------------------------------------
# 1. Create text-to-vector dataset
# --------------------------------------------------

data = {
    # Technology
    "computer": [2, 8],
    "laptop": [2, 7],
    "smartphone": [3, 8],
    "software": [4, 9],

    # Food
    "pizza": [7, 2],
    "burger": [7, 3],
    "rice": [8, 2],
    "pasta": [8, 3],

    # Animals
    "cat": [2, 3],
    "dog": [3, 4],
    "lion": [4, 5],
    "tiger": [4, 6],

    # Sports
    "football": [6, 7],
    "cricket": [5, 8],
    "tennis": [6, 8],
    "basketball": [7, 7],

    # Vehicles
    "car": [9, 1],
    "bus": [8, 2],
    "bike": [9, 2],
    "train": [10, 3]
}


# Convert dictionary vectors into NumPy array
words = list(data.keys())
vectors = np.array(list(data.values()), dtype=float)


# --------------------------------------------------
# 2. Display each word and its vector
# --------------------------------------------------

def display_vectors():
    print("\n===== TEXT TO VECTOR DATASET =====")

    for word, vector in data.items():
        print(f"{word:12} -> {vector}")


# --------------------------------------------------
# 3. Display vector dimensions
# --------------------------------------------------

def display_dimensions():
    print("\n===== VECTOR DIMENSIONS =====")

    print("Number of vectors:", vectors.shape[0])
    print("Vector dimensions:", vectors.shape[1])

    for word, vector in data.items():
        print(f"{word:12} -> Dimension: {len(vector)}")


# --------------------------------------------------
# 4. Compare two vectors
# --------------------------------------------------

def compare_vectors(word1, word2):
    vector1 = np.array(data[word1])
    vector2 = np.array(data[word2])

    print("\n===== VECTOR COMPARISON =====")

    print(f"{word1}: {vector1}")
    print(f"{word2}: {vector2}")

    print("Difference:", vector1 - vector2)


# --------------------------------------------------
# 5. Calculate Euclidean distance
# --------------------------------------------------

def calculate_distance(word1, word2):
    vector1 = np.array(data[word1])
    vector2 = np.array(data[word2])

    distance = np.linalg.norm(vector1 - vector2)

    print("\n===== DISTANCE =====")
    print(f"Distance between '{word1}' and '{word2}': {distance:.2f}")

    return distance


# --------------------------------------------------
# 6. Find closest vectors
# --------------------------------------------------

def find_closest_vectors():
    print("\n===== CLOSEST VECTOR PAIRS =====")

    closest_pairs = []

    min_distance = float("inf")

    for i in range(len(words)):
        for j in range(i + 1, len(words)):

            distance = np.linalg.norm(
                vectors[i] - vectors[j]
            )

            if distance < min_distance:
                min_distance = distance
                closest_pairs = [
                    (words[i], words[j], distance)
                ]

            elif distance == min_distance:
                closest_pairs.append(
                    (words[i], words[j], distance)
                )

    for word1, word2, distance in closest_pairs:
        print(
            f"{word1} <-> {word2} = {distance:.2f}"
        )


# --------------------------------------------------
# 7. 2D Visualization
# --------------------------------------------------

def visualize_vectors():

    plt.figure(figsize=(10, 7))

    for word, vector in data.items():

        x = vector[0]
        y = vector[1]

        plt.scatter(x, y)

        plt.annotate(
            word,
            (x, y),
            xytext=(5, 5),
            textcoords="offset points"
        )

    plt.xlabel("Vector Dimension 1")
    plt.ylabel("Vector Dimension 2")
    plt.title("Text-to-Vector Explorer")
    plt.grid(True)

    plt.show()


# --------------------------------------------------
# 8. Main program
# --------------------------------------------------

def main():

    display_vectors()

    display_dimensions()

    # Compare two example vectors
    compare_vectors("cat", "dog")

    # Calculate distance
    calculate_distance("cat", "dog")

    calculate_distance("car", "bus")

    # Find closest vectors
    find_closest_vectors()

    # Visualization
    visualize_vectors()


# Run program
if __name__ == "__main__":
    main()