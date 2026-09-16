import numpy as np


# ==========================================================
# 1. DATA COLLECTION
# ==========================================================

def create_dataset():

    np.random.seed(42)

    student_ids = np.arange(1001, 1101)

    ages = np.random.randint(17, 26, 100)

    math_marks = np.random.randint(40, 101, 100).astype(float)

    science_marks = np.random.randint(40, 101, 100).astype(float)

    english_marks = np.random.randint(40, 101, 100).astype(float)

    ai_marks = np.random.randint(40, 101, 100).astype(float)

    study_hours = np.random.randint(1, 11, 100).astype(float)

    attendance = np.random.randint(50, 101, 100).astype(float)

    # ---------------------------------------------
    # Introduce some invalid values
    # ---------------------------------------------

    math_marks[5] = -10
    science_marks[20] = -5
    study_hours[30] = -2

    # ---------------------------------------------
    # Introduce missing values using NaN
    # ---------------------------------------------

    math_marks[10] = np.nan
    science_marks[25] = np.nan
    english_marks[40] = np.nan
    study_hours[50] = np.nan

    # ---------------------------------------------
    # Combine all columns
    # ---------------------------------------------

    data = np.column_stack((
        student_ids,
        ages,
        math_marks,
        science_marks,
        english_marks,
        ai_marks,
        study_hours,
        attendance
    ))

    return data


# ==========================================================
# 2. DATA EXPLORATION
# ==========================================================

def explore_dataset(data):

    print("\n======================================")
    print("        DATASET EXPLORATION")
    print("======================================")

    print("Shape      :", data.shape)
    print("Dimensions :", data.ndim)
    print("Size       :", data.size)
    print("Data Type  :", data.dtype)

    print("\nFirst 5 Records:")
    print(data[:5])

    print("\nLast 5 Records:")
    print(data[-5:])


# ==========================================================
# 3. DATA CLEANING
# ==========================================================

def clean_dataset(data):

    cleaned_data = data.copy()

    # ---------------------------------------------
    # Replace missing values
    # ---------------------------------------------

    for column in range(cleaned_data.shape[1]):

        column_data = cleaned_data[:, column]

        valid_values = column_data[~np.isnan(column_data)]

        if len(valid_values) > 0:

            column_mean = np.mean(valid_values)

            cleaned_data[:, column] = np.where(
                np.isnan(column_data),
                column_mean,
                column_data
            )

    # ---------------------------------------------
    # Replace invalid negative values
    # ---------------------------------------------

    # Math Marks
    math_mean = np.mean(cleaned_data[:, 2])

    cleaned_data[:, 2] = np.where(
        cleaned_data[:, 2] < 0,
        math_mean,
        cleaned_data[:, 2]
    )

    # Science Marks
    science_mean = np.mean(cleaned_data[:, 3])

    cleaned_data[:, 3] = np.where(
        cleaned_data[:, 3] < 0,
        science_mean,
        cleaned_data[:, 3]
    )

    # Study Hours
    study_mean = np.mean(cleaned_data[:, 6])

    cleaned_data[:, 6] = np.where(
        cleaned_data[:, 6] < 0,
        study_mean,
        cleaned_data[:, 6]
    )

    return cleaned_data


# ==========================================================
# 4. FILTER RECORDS
# ==========================================================

def filter_students(data):

    # Students with average marks above 75

    average_marks = (
        data[:, 2] +
        data[:, 3] +
        data[:, 4] +
        data[:, 5]
    ) / 4

    high_performers = data[average_marks > 75]

    print("\n======================================")
    print("        HIGH PERFORMING STUDENTS")
    print("======================================")

    print("Number of students:", len(high_performers))

    print("\nFirst 10 high performers:")
    print(high_performers[:10])

    return high_performers


# ==========================================================
# 5. STATISTICAL ANALYSIS
# ==========================================================

def statistical_analysis(data):

    print("\n======================================")
    print("        STATISTICAL ANALYSIS")
    print("======================================")

    subjects = {
        "Math": 2,
        "Science": 3,
        "English": 4,
        "AI": 5
    }

    for subject, column in subjects.items():

        values = data[:, column]

        print(f"\n{subject}:")

        print(f"Mean   : {np.mean(values):.2f}")
        print(f"Median : {np.median(values):.2f}")
        print(f"Minimum: {np.min(values):.2f}")
        print(f"Maximum: {np.max(values):.2f}")
        print(f"Std Dev: {np.std(values):.2f}")


# ==========================================================
# 6. BROADCASTING / DATA TRANSFORMATION
# ==========================================================

def transform_data(data):

    transformed_data = data.copy()

    # ---------------------------------------------
    # Select marks columns
    # ---------------------------------------------

    marks = transformed_data[:, 2:6]

    # ---------------------------------------------
    # Broadcasting
    # Add 5 bonus marks to every subject
    # ---------------------------------------------

    marks += 5

    # Make sure marks don't exceed 100
    marks = np.clip(marks, 0, 100)

    transformed_data[:, 2:6] = marks

    return transformed_data


# ==========================================================
# 7. FEATURE SELECTION
# ==========================================================

def select_features(data):

    # Select:
    # Age
    # Math
    # Science
    # English
    # AI
    # Study Hours
    # Attendance

    features = data[:, 1:8]

    return features


# ==========================================================
# 8. RESHAPING
# ==========================================================

def reshape_data(data):

    print("\n======================================")
    print("        RESHAPING DATA")
    print("======================================")

    print("Original Shape:", data.shape)

    # Reshape into 3D

    reshaped = data.reshape(100, 8, 1)

    print("3D Shape      :", reshaped.shape)
    print("Dimensions    :", reshaped.ndim)

    return reshaped


# ==========================================================
# 9. SUMMARY STATISTICS
# ==========================================================

def generate_summary(data):

    print("\n======================================")
    print("        SUMMARY STATISTICS")
    print("======================================")

    marks = data[:, 2:6]

    all_marks = marks.flatten()

    print("Total Students :", len(data))

    print("Average Mark   :", np.mean(all_marks))

    print("Highest Mark   :", np.max(all_marks))

    print("Lowest Mark    :", np.min(all_marks))

    print("Variance       :", np.var(all_marks))

    print("Std Deviation  :", np.std(all_marks))


# ==========================================================
# 10. SAVE DATASET
# ==========================================================

def save_dataset(data):

    filename = "processed_student_data.csv"

    np.savetxt(
        filename,
        data,
        delimiter=",",
        fmt="%.2f"
    )

    print("\n======================================")
    print("        DATASET SAVED")
    print("======================================")

    print("File:", filename)


# ==========================================================
# 11. MAIN PIPELINE
# ==========================================================

def main():

    print("==============================================")
    print("      AI DATA PREPROCESSING PIPELINE")
    print("      STUDENT PERFORMANCE DATA")
    print("==============================================")

    # ------------------------------------------
    # Step 1: Data Collection
    # ------------------------------------------

    data = create_dataset()

    print("\n100 student records created successfully.")

    # ------------------------------------------
    # Step 2: Data Exploration
    # ------------------------------------------

    explore_dataset(data)

    # ------------------------------------------
    # Step 3: Data Cleaning
    # ------------------------------------------

    data = clean_dataset(data)

    print("\nData cleaning completed.")

    # ------------------------------------------
    # Step 4: Statistical Analysis
    # ------------------------------------------

    statistical_analysis(data)

    # ------------------------------------------
    # Step 5: Filtering
    # ------------------------------------------

    filter_students(data)

    # ------------------------------------------
    # Step 6: Data Transformation
    # ------------------------------------------

    data = transform_data(data)

    print("\nData transformation completed.")

    # ------------------------------------------
    # Step 7: Feature Selection
    # ------------------------------------------

    features = select_features(data)

    print("\nSelected Feature Shape:", features.shape)

    # ------------------------------------------
    # Step 8: Reshaping
    # ------------------------------------------

    reshape_data(data)

    # ------------------------------------------
    # Step 9: Summary Statistics
    # ------------------------------------------

    generate_summary(data)

    # ------------------------------------------
    # Step 10: Save Final Dataset
    # ------------------------------------------

    save_dataset(data)

    print("\n==============================================")
    print("       PREPROCESSING PIPELINE COMPLETE")
    print("       Dataset Ready for AI/ML")
    print("==============================================")


# ==========================================================
# RUN PROGRAM
# ==========================================================

if __name__ == "__main__":
    main()