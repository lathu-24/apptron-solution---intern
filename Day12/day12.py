import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

EMBEDDING_MODEL = "all-MiniLM-L6-v2"

# Change this if you want a different chunk size
CHUNK_SIZE = 500

# Number of search results
TOP_K = 5


# ============================================================
# SAMPLE KNOWLEDGE BASE
# 100+ DOCUMENTS
# ============================================================

documents = [

    # ========================================================
    # PYTHON - 20 DOCUMENTS
    # ========================================================

    {
        "name": "python_basics.txt",
        "category": "Python",
        "text": """
        Python is a high-level programming language known for its simple
        syntax and readability. Python is widely used in software development,
        data science, artificial intelligence, automation, and web development.
        """
    },

    {
        "name": "python_variables.txt",
        "category": "Python",
        "text": """
        Variables in Python are used to store data values. Python variables
        do not require explicit type declarations. A variable can store
        strings, integers, floating point numbers, lists, dictionaries,
        and other Python objects.
        """
    },

    {
        "name": "python_data_types.txt",
        "category": "Python",
        "text": """
        Common Python data types include integers, floats, strings, booleans,
        lists, tuples, sets, and dictionaries. Choosing an appropriate data
        type helps programs represent and process information effectively.
        """
    },

    {
        "name": "python_functions.txt",
        "category": "Python",
        "text": """
        Functions allow developers to organize reusable pieces of code.
        Python functions are defined using the def keyword. Functions can
        accept parameters and return values using the return statement.
        """
    },

    {
        "name": "python_loops.txt",
        "category": "Python",
        "text": """
        Python provides for loops and while loops for repetition. A for loop
        is commonly used to iterate over sequences such as lists and strings,
        while a while loop continues execution while a condition is true.
        """
    },

    {
        "name": "python_conditions.txt",
        "category": "Python",
        "text": """
        Conditional statements allow Python programs to make decisions.
        The if, elif, and else statements execute different blocks of code
        depending on whether conditions are true or false.
        """
    },

    {
        "name": "python_lists.txt",
        "category": "Python",
        "text": """
        A Python list is an ordered and mutable collection. Lists can contain
        multiple values and support indexing, slicing, adding elements,
        removing elements, and sorting.
        """
    },

    {
        "name": "python_dictionary.txt",
        "category": "Python",
        "text": """
        Dictionaries store data using key-value pairs. They are useful when
        information needs to be accessed using meaningful keys rather than
        numerical indexes.
        """
    },

    {
        "name": "python_classes.txt",
        "category": "Python",
        "text": """
        Classes are used to implement object-oriented programming in Python.
        A class defines attributes and methods that describe the behavior
        and properties of objects.
        """
    },

    {
        "name": "python_exceptions.txt",
        "category": "Python",
        "text": """
        Exception handling allows Python programs to handle runtime errors.
        The try, except, else, and finally blocks can be used to detect and
        manage exceptions without unexpectedly terminating the application.
        """
    },

    {
        "name": "python_modules.txt",
        "category": "Python",
        "text": """
        Python modules allow developers to organize reusable code into
        separate files. Modules can be imported using the import statement.
        Python also provides a large standard library.
        """
    },

    {
        "name": "python_file_handling.txt",
        "category": "Python",
        "text": """
        Python can read and write files using the open function. Common
        operations include reading text, writing new content, and appending
        information to existing files.
        """
    },

    {
        "name": "python_virtual_environment.txt",
        "category": "Python",
        "text": """
        Virtual environments provide isolated Python environments for
        projects. They allow each project to use its own dependencies
        without affecting packages installed for other projects.
        """
    },

    {
        "name": "python_packages.txt",
        "category": "Python",
        "text": """
        Python packages contain reusable modules and functionality.
        The pip package manager is commonly used to install third-party
        Python libraries.
        """
    },

    {
        "name": "python_lambda.txt",
        "category": "Python",
        "text": """
        Lambda expressions provide a short way to create small anonymous
        functions. They are often used with functions such as map, filter,
        and sorted.
        """
    },

    {
        "name": "python_comprehensions.txt",
        "category": "Python",
        "text": """
        List comprehensions provide a concise way to create lists from
        existing sequences. They can include expressions and conditions
        in a single readable statement.
        """
    },

    {
        "name": "python_testing.txt",
        "category": "Python",
        "text": """
        Python applications can be tested using frameworks such as unittest
        and pytest. Automated tests help developers detect bugs and maintain
        software quality.
        """
    },

    {
        "name": "python_debugging.txt",
        "category": "Python",
        "text": """
        Debugging is the process of identifying and fixing errors in programs.
        Python developers can use print statements, logging, IDE debuggers,
        and exception tracebacks to investigate problems.
        """
    },

    {
        "name": "python_json.txt",
        "category": "Python",
        "text": """
        Python provides the json module for working with JSON data.
        JSON is commonly used for configuration files, APIs, and exchanging
        structured information between applications.
        """
    },

    {
        "name": "python_api.txt",
        "category": "Python",
        "text": """
        Python can communicate with web APIs using libraries such as requests.
        APIs allow applications to send requests and receive data from
        external services.
        """
    },


    # ========================================================
    # AI / MACHINE LEARNING - 20 DOCUMENTS
    # ========================================================

    {
        "name": "machine_learning_basics.txt",
        "category": "AI/ML",
        "text": """
        Machine learning is a field of artificial intelligence where
        algorithms learn patterns from data and use those patterns to make
        predictions or decisions.
        """
    },

    {
        "name": "supervised_learning.txt",
        "category": "AI/ML",
        "text": """
        Supervised learning uses labeled training data. Common supervised
        learning tasks include classification and regression.
        """
    },

    {
        "name": "unsupervised_learning.txt",
        "category": "AI/ML",
        "text": """
        Unsupervised learning works with data that does not have predefined
        labels. Clustering and dimensionality reduction are common examples.
        """
    },

    {
        "name": "classification.txt",
        "category": "AI/ML",
        "text": """
        Classification predicts discrete categories. Examples include spam
        detection, disease classification, and customer churn prediction.
        """
    },

    {
        "name": "regression.txt",
        "category": "AI/ML",
        "text": """
        Regression models predict continuous numerical values. Examples
        include house price prediction, sales forecasting, and temperature
        prediction.
        """
    },

    {
        "name": "training_data.txt",
        "category": "AI/ML",
        "text": """
        Training data is used by machine learning algorithms to learn
        relationships between input features and target values.
        """
    },

    {
        "name": "test_data.txt",
        "category": "AI/ML",
        "text": """
        Test data is used to evaluate how well a trained machine learning
        model performs on previously unseen examples.
        """
    },

    {
        "name": "validation_data.txt",
        "category": "AI/ML",
        "text": """
        Validation data helps developers tune machine learning models and
        hyperparameters before evaluating the final model on test data.
        """
    },

    {
        "name": "overfitting.txt",
        "category": "AI/ML",
        "text": """
        Overfitting occurs when a model learns the training data too closely
        and performs poorly on unseen data. Regularization, more training
        data, and simpler models can help reduce overfitting.
        """
    },

    {
        "name": "underfitting.txt",
        "category": "AI/ML",
        "text": """
        Underfitting happens when a machine learning model is too simple
        to capture important patterns in the data.
        """
    },

    {
        "name": "feature_engineering.txt",
        "category": "AI/ML",
        "text": """
        Feature engineering involves creating useful input variables from
        raw data. Good features can improve the performance of machine
        learning models.
        """
    },

    {
        "name": "data_preprocessing.txt",
        "category": "AI/ML",
        "text": """
        Data preprocessing prepares raw data for machine learning.
        Common steps include handling missing values, encoding categories,
        scaling numerical values, and removing invalid records.
        """
    },

    {
        "name": "random_forest.txt",
        "category": "AI/ML",
        "text": """
        Random Forest is an ensemble machine learning algorithm that combines
        multiple decision trees. It can be used for both classification
        and regression problems.
        """
    },

    {
        "name": "decision_tree.txt",
        "category": "AI/ML",
        "text": """
        Decision trees make predictions by repeatedly splitting data using
        feature-based conditions. They are relatively easy to interpret.
        """
    },

    {
        "name": "logistic_regression.txt",
        "category": "AI/ML",
        "text": """
        Logistic regression is commonly used for binary classification.
        It estimates the probability that an observation belongs to a class.
        """
    },

    {
        "name": "neural_networks.txt",
        "category": "AI/ML",
        "text": """
        Neural networks consist of interconnected computational units
        organized into layers. They are widely used in computer vision,
        natural language processing, and other AI applications.
        """
    },

    {
        "name": "model_evaluation.txt",
        "category": "AI/ML",
        "text": """
        Machine learning models can be evaluated using metrics such as
        accuracy, precision, recall, F1 score, mean squared error, and
        mean absolute error.
        """
    },

    {
        "name": "cross_validation.txt",
        "category": "AI/ML",
        "text": """
        Cross-validation evaluates machine learning models using multiple
        training and validation splits. It provides a more reliable estimate
        of model performance.
        """
    },

    {
        "name": "hyperparameters.txt",
        "category": "AI/ML",
        "text": """
        Hyperparameters are configuration values selected before model
        training. Examples include learning rate, tree depth, number of
        estimators, and regularization strength.
        """
    },

    {
        "name": "machine_learning_pipeline.txt",
        "category": "AI/ML",
        "text": """
        A machine learning pipeline commonly includes data collection,
        preprocessing, feature engineering, model training, evaluation,
        deployment, and monitoring.
        """
    },


    # ========================================================
    # NUMPY - 20 DOCUMENTS
    # ========================================================

    {
        "name": "numpy_introduction.txt",
        "category": "NumPy",
        "text": """
        NumPy is a Python library designed for numerical computing.
        It provides efficient multidimensional arrays and mathematical
        operations.
        """
    },

    {
        "name": "numpy_arrays.txt",
        "category": "NumPy",
        "text": """
        NumPy arrays store numerical data efficiently. Unlike Python lists,
        NumPy arrays support fast vectorized mathematical operations.
        """
    },

    {
        "name": "numpy_dimensions.txt",
        "category": "NumPy",
        "text": """
        NumPy arrays can have one or more dimensions. A one-dimensional
        array represents a sequence, while a two-dimensional array can
        represent rows and columns.
        """
    },

    {
        "name": "numpy_shape.txt",
        "category": "NumPy",
        "text": """
        The shape attribute of a NumPy array describes the size of each
        dimension. For example, an array with shape 10 by 5 contains
        10 rows and 5 columns.
        """
    },

    {
        "name": "numpy_indexing.txt",
        "category": "NumPy",
        "text": """
        NumPy supports indexing to access individual array elements.
        Indexing begins from zero, similar to Python lists.
        """
    },

    {
        "name": "numpy_slicing.txt",
        "category": "NumPy",
        "text": """
        Array slicing allows developers to select portions of NumPy arrays.
        Slicing can select rows, columns, or ranges of elements.
        """
    },

    {
        "name": "numpy_statistics.txt",
        "category": "NumPy",
        "text": """
        NumPy provides statistical functions such as mean, median, minimum,
        maximum, variance, and standard deviation.
        """
    },

    {
        "name": "numpy_mean.txt",
        "category": "NumPy",
        "text": """
        The NumPy mean function calculates the arithmetic average of
        numerical values in an array.
        """
    },

    {
        "name": "numpy_sum.txt",
        "category": "NumPy",
        "text": """
        np.sum calculates the total of elements in a NumPy array.
        It can also calculate sums across specific axes.
        """
    },

    {
        "name": "numpy_max_min.txt",
        "category": "NumPy",
        "text": """
        NumPy provides np.max and np.min to find the largest and smallest
        values in an array.
        """
    },

    {
        "name": "numpy_sorting.txt",
        "category": "NumPy",
        "text": """
        NumPy provides sorting functions that can arrange numerical values.
        Sorting can be performed along different axes of an array.
        """
    },

    {
        "name": "numpy_random.txt",
        "category": "NumPy",
        "text": """
        NumPy provides random number generation tools for simulations,
        experiments, machine learning datasets, and testing.
        """
    },

    {
        "name": "numpy_reshape.txt",
        "category": "NumPy",
        "text": """
        The reshape operation changes the dimensions of an array without
        changing the underlying data, as long as the total number of
        elements remains compatible.
        """
    },

    {
        "name": "numpy_transpose.txt",
        "category": "NumPy",
        "text": """
        Transpose changes the orientation of array dimensions. For a
        two-dimensional matrix, rows become columns and columns become rows.
        """
    },

    {
        "name": "numpy_broadcasting.txt",
        "category": "NumPy",
        "text": """
        Broadcasting allows NumPy to perform operations between arrays
        with compatible but different shapes.
        """
    },

    {
        "name": "numpy_boolean_indexing.txt",
        "category": "NumPy",
        "text": """
        Boolean indexing allows developers to select array elements based
        on conditions. It is useful for filtering numerical datasets.
        """
    },

    {
        "name": "numpy_linear_algebra.txt",
        "category": "NumPy",
        "text": """
        NumPy provides linear algebra operations including matrix
        multiplication, vector norms, determinants, and decompositions.
        """
    },

    {
        "name": "numpy_dot_product.txt",
        "category": "NumPy",
        "text": """
        np.dot can calculate the dot product of vectors and perform
        matrix multiplication depending on the input dimensions.
        """
    },

    {
        "name": "numpy_performance.txt",
        "category": "NumPy",
        "text": """
        NumPy improves numerical computing performance by implementing
        many operations in optimized low-level code and supporting
        vectorized computation.
        """
    },

    {
        "name": "numpy_machine_learning.txt",
        "category": "NumPy",
        "text": """
        NumPy is commonly used in machine learning for numerical data,
        feature arrays, mathematical calculations, and preprocessing.
        """
    },


    # ========================================================
    # PANDAS - 20 DOCUMENTS
    # ========================================================

    {
        "name": "pandas_introduction.txt",
        "category": "Pandas",
        "text": """
        Pandas is a Python library for data manipulation and analysis.
        It provides powerful structures such as Series and DataFrame.
        """
    },

    {
        "name": "pandas_dataframe.txt",
        "category": "Pandas",
        "text": """
        A Pandas DataFrame is a two-dimensional labeled data structure
        containing rows and columns. It is commonly used for tabular data.
        """
    },

    {
        "name": "pandas_series.txt",
        "category": "Pandas",
        "text": """
        A Pandas Series is a one-dimensional labeled data structure.
        It can contain numerical, textual, or other types of data.
        """
    },

    {
        "name": "pandas_read_csv.txt",
        "category": "Pandas",
        "text": """
        Pandas provides read_csv for loading CSV files into a DataFrame.
        CSV files are commonly used to store structured tabular data.
        """
    },

    {
        "name": "pandas_missing_values.txt",
        "category": "Pandas",
        "text": """
        Missing values can be handled using Pandas methods such as
        isna, fillna, and dropna. These operations help prepare data
        for analysis and machine learning.
        """
    },

    {
        "name": "pandas_filtering.txt",
        "category": "Pandas",
        "text": """
        Pandas supports filtering rows using Boolean conditions.
        Filtering is useful for selecting records that satisfy specific
        requirements.
        """
    },

    {
        "name": "pandas_sorting.txt",
        "category": "Pandas",
        "text": """
        DataFrames can be sorted using sort_values. Sorting can be performed
        using one or more columns in ascending or descending order.
        """
    },

    {
        "name": "pandas_groupby.txt",
        "category": "Pandas",
        "text": """
        The groupby operation groups records based on one or more columns.
        It is useful for calculating statistics for different categories.
        """
    },

    {
        "name": "pandas_merge.txt",
        "category": "Pandas",
        "text": """
        Pandas merge combines DataFrames using common columns or indexes.
        It is similar to joining tables in a relational database.
        """
    },

    {
        "name": "pandas_concat.txt",
        "category": "Pandas",
        "text": """
        Pandas concat combines Series or DataFrames along rows or columns.
        It is useful when datasets need to be appended together.
        """
    },

    {
        "name": "pandas_duplicates.txt",
        "category": "Pandas",
        "text": """
        Pandas provides duplicated and drop_duplicates for identifying
        and removing duplicate records from a dataset.
        """
    },

    {
        "name": "pandas_columns.txt",
        "category": "Pandas",
        "text": """
        DataFrame columns can be selected, renamed, added, or removed.
        Column operations are fundamental to data preparation.
        """
    },

    {
        "name": "pandas_statistics.txt",
        "category": "Pandas",
        "text": """
        Pandas provides descriptive statistics through methods such as
        describe, mean, median, min, max, and standard deviation.
        """
    },

    {
        "name": "pandas_apply.txt",
        "category": "Pandas",
        "text": """
        The apply method allows functions to be applied to DataFrame rows,
        columns, or Series values.
        """
    },

    {
        "name": "pandas_transform.txt",
        "category": "Pandas",
        "text": """
        Pandas can transform columns using mathematical operations,
        functions, mappings, and conditional logic.
        """
    },

    {
        "name": "pandas_datetime.txt",
        "category": "Pandas",
        "text": """
        Pandas provides datetime functionality for working with dates
        and times. Datetime values can be extracted into useful features.
        """
    },

    {
        "name": "pandas_excel.txt",
        "category": "Pandas",
        "text": """
        Pandas can read and write Excel files using functions such as
        read_excel and to_excel.
        """
    },

    {
        "name": "pandas_data_cleaning.txt",
        "category": "Pandas",
        "text": """
        Pandas is frequently used for data cleaning tasks including
        handling missing values, removing duplicates, correcting data
        types, and filtering invalid records.
        """
    },

    {
        "name": "pandas_ml.txt",
        "category": "Pandas",
        "text": """
        Pandas is commonly used before machine learning to load datasets,
        clean data, create features, and prepare training and testing data.
        """
    },

    {
        "name": "pandas_visualization.txt",
        "category": "Pandas",
        "text": """
        Pandas provides basic plotting capabilities and can work with
        visualization libraries such as Matplotlib for data exploration.
        """
    },


    # ========================================================
    # COMPANY POLICIES - 10 DOCUMENTS
    # ========================================================

    {
        "name": "attendance_policy.txt",
        "category": "Company Policies",
        "text": """
        Employees and interns are expected to attend scheduled working
        sessions on time. Any planned absence should be communicated to
        the relevant supervisor in advance.
        """
    },

    {
        "name": "remote_work_policy.txt",
        "category": "Company Policies",
        "text": """
        Remote employees should maintain a professional working environment,
        remain available during agreed working hours, and communicate
        regularly with their team.
        """
    },

    {
        "name": "leave_policy.txt",
        "category": "Company Policies",
        "text": """
        Employees should submit leave requests through the approved process.
        Planned leave should normally be requested before the absence.
        """
    },

    {
        "name": "security_policy.txt",
        "category": "Company Policies",
        "text": """
        Company systems and credentials must be protected. Passwords,
        authentication information, and confidential company data must
        not be shared with unauthorized people.
        """
    },

    {
        "name": "password_policy.txt",
        "category": "Company Policies",
        "text": """
        Employees should use strong passwords and avoid reusing company
        passwords on personal services. Authentication information should
        remain confidential.
        """
    },

    {
        "name": "data_privacy_policy.txt",
        "category": "Company Policies",
        "text": """
        Personal and confidential information should only be collected,
        processed, and shared for legitimate business purposes.
        """
    },

    {
        "name": "communication_policy.txt",
        "category": "Company Policies",
        "text": """
        Professional and respectful communication is expected when using
        company email, messaging systems, meetings, and collaboration tools.
        """
    },

    {
        "name": "equipment_policy.txt",
        "category": "Company Policies",
        "text": """
        Company equipment should be used responsibly and protected from
        damage, unauthorized access, and loss.
        """
    },

    {
        "name": "code_of_conduct.txt",
        "category": "Company Policies",
        "text": """
        Employees are expected to behave professionally, respectfully,
        ethically, and responsibly while representing the organization.
        """
    },

    {
        "name": "confidentiality_policy.txt",
        "category": "Company Policies",
        "text": """
        Confidential company information must only be accessed by authorized
        individuals and should not be disclosed outside approved channels.
        """
    },


    # ========================================================
    # TRAINING - 10 DOCUMENTS
    # ========================================================

    {
        "name": "training_orientation.txt",
        "category": "Training",
        "text": """
        New interns should complete the organization orientation program.
        The orientation introduces company culture, tools, communication
        processes, and workplace expectations.
        """
    },

    {
        "name": "training_python.txt",
        "category": "Training",
        "text": """
        Python training covers variables, data types, conditions, loops,
        functions, object-oriented programming, modules, and file handling.
        """
    },

    {
        "name": "training_numpy.txt",
        "category": "Training",
        "text": """
        NumPy training introduces arrays, indexing, slicing, broadcasting,
        mathematical operations, statistics, and numerical data processing.
        """
    },

    {
        "name": "training_pandas.txt",
        "category": "Training",
        "text": """
        Pandas training covers DataFrames, data loading, filtering,
        cleaning, grouping, merging, sorting, and data analysis.
        """
    },

    {
        "name": "training_machine_learning.txt",
        "category": "Training",
        "text": """
        Machine learning training introduces supervised learning,
        unsupervised learning, preprocessing, model training, evaluation,
        and model improvement.
        """
    },

    {
        "name": "training_git.txt",
        "category": "Training",
        "text": """
        Git training covers repositories, commits, branches, merging,
        remote repositories, pushing, pulling, and collaborative development.
        """
    },

    {
        "name": "training_github.txt",
        "category": "Training",
        "text": """
        GitHub training explains how developers store projects online,
        collaborate through repositories, review code, and manage issues.
        """
    },

    {
        "name": "training_debugging.txt",
        "category": "Training",
        "text": """
        Debugging training teaches developers how to reproduce problems,
        inspect errors, use debugging tools, identify root causes,
        and verify fixes.
        """
    },

    {
        "name": "training_testing.txt",
        "category": "Training",
        "text": """
        Software testing training introduces unit testing, integration
        testing, test cases, assertions, and automated testing practices.
        """
    },

    {
        "name": "training_best_practices.txt",
        "category": "Training",
        "text": """
        Development best practices include writing readable code,
        meaningful variable names, documentation, testing, version control,
        and regular code reviews.
        """
    }
]


# ============================================================
# GLOBAL VARIABLES
# ============================================================

model = None
knowledge_base = []


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):
    """
    Clean unnecessary spaces and line breaks.
    """

    text = text.strip()

    # Replace multiple spaces/newlines with a single space
    text = re.sub(r"\s+", " ", text)

    return text


# ============================================================
# TEXT CHUNKING
# ============================================================

def split_into_chunks(text, chunk_size=CHUNK_SIZE):
    """
    Split text into smaller chunks.
    """

    text = clean_text(text)

    words = text.split()

    chunks = []

    current_chunk = []
    current_length = 0

    for word in words:

        word_length = len(word) + 1

        if current_length + word_length > chunk_size:

            if current_chunk:
                chunks.append(" ".join(current_chunk))

            current_chunk = [word]
            current_length = word_length

        else:
            current_chunk.append(word)
            current_length += word_length

    if current_chunk:
        chunks.append(" ".join(current_chunk))

    return chunks


# ============================================================
# CREATE KNOWLEDGE BASE
# ============================================================

def create_knowledge_base():

    global knowledge_base

    knowledge_base = []

    print("\nProcessing documents...")

    for document in documents:

        text = clean_text(document["text"])

        chunks = split_into_chunks(text)

        for index, chunk in enumerate(chunks):

            knowledge_base.append({
                "chunk_id": len(knowledge_base) + 1,
                "chunk": chunk,
                "category": document["category"],
                "document_name": document["name"],
                "chunk_number": index + 1
            })

    print(f"Documents processed: {len(documents)}")
    print(f"Total chunks created: {len(knowledge_base)}")


# ============================================================
# GENERATE EMBEDDINGS
# ============================================================

def generate_embeddings():

    global knowledge_base

    print("\nGenerating embeddings...")

    texts = [
        item["chunk"]
        for item in knowledge_base
    ]

    embeddings = model.encode(
        texts,
        convert_to_numpy=True,
        show_progress_bar=True
    )

    for i, embedding in enumerate(embeddings):

        knowledge_base[i]["embedding"] = embedding

    print("Embedding generation completed.")

    print(
        "Embedding shape:",
        embeddings.shape
    )


# ============================================================
# COSINE SIMILARITY
# ============================================================

def cosine_similarity(vector_a, vector_b):

    norm_a = np.linalg.norm(vector_a)
    norm_b = np.linalg.norm(vector_b)

    if norm_a == 0 or norm_b == 0:
        return 0.0

    return np.dot(vector_a, vector_b) / (norm_a * norm_b)


# ============================================================
# SEMANTIC SEARCH
# ============================================================

def semantic_search(query, category=None, top_k=TOP_K):

    query_embedding = model.encode(
        query,
        convert_to_numpy=True
    )

    results = []

    for item in knowledge_base:

        # Category filter
        if category is not None:

            if item["category"].lower() != category.lower():
                continue

        score = cosine_similarity(
            query_embedding,
            item["embedding"]
        )

        results.append({
            "score": float(score),
            "chunk": item["chunk"],
            "category": item["category"],
            "document_name": item["document_name"],
            "chunk_number": item["chunk_number"]
        })

    # Sort highest similarity first
    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# ============================================================
# DISPLAY SEARCH RESULTS
# ============================================================

def display_results(query, results):

    print("\n")
    print("=" * 70)
    print("SEMANTIC SEARCH RESULTS")
    print("=" * 70)

    print("\nQuery:")
    print(query)

    print("\nTop Results")
    print("-" * 70)

    for index, result in enumerate(results, start=1):

        print(f"\nResult {index}")
        print("-" * 40)

        print(
            f"Similarity Score: {result['score']:.4f}"
        )

        print(
            f"Category: {result['category']}"
        )

        print(
            f"Source Document: {result['document_name']}"
        )

        print(
            f"Chunk: {result['chunk_number']}"
        )

        print(
            f"Relevant Text: {result['chunk']}"
        )


# ============================================================
# BUILD CONTEXT FOR AI
# ============================================================

def build_context(results):

    context_parts = []

    for index, result in enumerate(results, start=1):

        context_parts.append(
            f"""
SOURCE {index}
Document: {result['document_name']}
Category: {result['category']}
Similarity: {result['score']:.4f}

Content:
{result['chunk']}
"""
        )

    return "\n".join(context_parts)


# ============================================================
# AI ANSWER
# ============================================================

def generate_ai_answer(query, results):

    """
    Generate an answer using an LLM.

    This function uses the OpenAI Python SDK.
    Set your API key before running:

        export OPENAI_API_KEY="your-api-key"

    The retrieved chunks are provided as context.
    """

    api_key = os.getenv("OPENAI_API_KEY")

    if not api_key:

        print("\nAI Answer")
        print("-" * 70)

        print(
            "OPENAI_API_KEY is not configured."
        )

        print(
            "\nThe retrieved knowledge can still be used "
            "as the answer context."
        )

        print("\nRelevant Knowledge:")

        for result in results:

            print(
                f"- {result['chunk']}"
            )

        return

    try:

        from openai import OpenAI

        client = OpenAI(api_key=api_key)

        context = build_context(results)

        prompt = f"""
You are an AI assistant for a company and educational
knowledge base.

Answer the user's question using ONLY the information
provided in the retrieved knowledge.

If the retrieved knowledge does not contain enough
information, clearly say that the knowledge base does
not provide enough information.

Do not invent information.

User Question:
{query}

Retrieved Knowledge:
{context}
"""

        response = client.responses.create(
            model="gpt-5.6-luna",
            input=prompt
        )

        answer = response.output_text

        print("\n")
        print("=" * 70)
        print("AI GENERATED ANSWER")
        print("=" * 70)

        print(answer)

    except Exception as error:

        print("\nAI answer generation failed.")

        print("Error:", error)

        print("\nRetrieved information:")

        for result in results:

            print(
                f"- {result['chunk']}"
            )


# ============================================================
# ADD NEW DOCUMENT
# ============================================================

def add_document():

    global documents
    global knowledge_base

    print("\n")
    print("=" * 70)
    print("ADD NEW DOCUMENT")
    print("=" * 70)

    document_name = input(
        "Document name: "
    ).strip()

    category = input(
        "Category: "
    ).strip()

    print("\nEnter document text.")
    print("Press ENTER twice when finished.")

    lines = []

    while True:

        line = input()

        if line == "":
            break

        lines.append(line)

    text = " ".join(lines)

    if not document_name:
        print("Document name cannot be empty.")
        return

    if not category:
        print("Category cannot be empty.")
        return

    if not text:
        print("Document text cannot be empty.")
        return

    # Add document
    documents.append({
        "name": document_name,
        "category": category,
        "text": text
    })

    # Split into chunks
    chunks = split_into_chunks(text)

    # Generate embeddings
    texts = np.array(chunks)

    embeddings = model.encode(
        list(texts),
        convert_to_numpy=True
    )

    # Add chunks
    for index, chunk in enumerate(chunks):

        knowledge_base.append({
            "chunk_id": len(knowledge_base) + 1,
            "chunk": chunk,
            "category": category,
            "document_name": document_name,
            "chunk_number": index + 1,
            "embedding": embeddings[index]
        })

    print("\nDocument added successfully.")

    print(
        f"Chunks created: {len(chunks)}"
    )


# ============================================================
# DISPLAY CATEGORIES
# ============================================================

def display_categories():

    categories = {}

    for document in documents:

        category = document["category"]

        if category not in categories:
            categories[category] = 0

        categories[category] += 1

    print("\n")
    print("=" * 50)
    print("KNOWLEDGE BASE CATEGORIES")
    print("=" * 50)

    for category, count in categories.items():

        print(
            f"{category}: {count} documents"
        )


# ============================================================
# KNOWLEDGE BASE STATISTICS
# ============================================================

def display_statistics():

    print("\n")
    print("=" * 50)
    print("KNOWLEDGE BASE STATISTICS")
    print("=" * 50)

    print(
        f"Documents: {len(documents)}"
    )

    print(
        f"Chunks: {len(knowledge_base)}"
    )

    if knowledge_base:

        embedding_size = len(
            knowledge_base[0]["embedding"]
        )

        print(
            f"Embedding dimensions: {embedding_size}"
        )


# ============================================================
# SEARCH MENU
# ============================================================

def search_menu():

    query = input(
        "\nEnter your question: "
    ).strip()

    if not query:

        print("Query cannot be empty.")

        return

    results = semantic_search(
        query,
        top_k=TOP_K
    )

    if not results:

        print("No results found.")

        return

    display_results(
        query,
        results
    )

    generate_ai_answer(
        query,
        results
    )


# ============================================================
# CATEGORY SEARCH
# ============================================================

def category_search():

    display_categories()

    category = input(
        "\nEnter category: "
    ).strip()

    query = input(
        "Enter your question: "
    ).strip()

    if not query:

        print("Query cannot be empty.")

        return

    results = semantic_search(
        query,
        category=category,
        top_k=TOP_K
    )

    if not results:

        print(
            "\nNo documents found for this category."
        )

        return

    display_results(
        query,
        results
    )

    generate_ai_answer(
        query,
        results
    )


# ============================================================
# MENU
# ============================================================

def display_menu():

    print("\n")
    print("=" * 70)
    print("       AI KNOWLEDGE BASE SEARCH SYSTEM")
    print("=" * 70)

    print("1. Semantic Search")
    print("2. Search by Category")
    print("3. Add New Document")
    print("4. Display Categories")
    print("5. Display Statistics")
    print("6. Exit")

    print("=" * 70)


# ============================================================
# MAIN
# ============================================================

def main():

    global model

    print("\nLoading embedding model...")

    model = SentenceTransformer(
        EMBEDDING_MODEL
    )

    print("Embedding model loaded.")

    # Create knowledge base
    create_knowledge_base()

    # Generate embeddings
    generate_embeddings()

    display_statistics()

    while True:

        display_menu()

        choice = input(
            "Enter your choice: "
        ).strip()

        if choice == "1":

            search_menu()

        elif choice == "2":

            category_search()

        elif choice == "3":

            add_document()

        elif choice == "4":

            display_categories()

        elif choice == "5":

            display_statistics()

        elif choice == "6":

            print("\nThank you for using the AI Knowledge Base.")
            break

        else:

            print(
                "\nInvalid choice. Please try again."
            )


# ============================================================
# PROGRAM START
# ============================================================

if __name__ == "__main__":
    main()