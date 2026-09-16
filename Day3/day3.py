import numpy as np


# -----------------------------------------
# Create Student Dataset
# -----------------------------------------
def create_dataset():
    students = np.array([
        [101, 19, 85, 78, 82, 90],
        [102, 20, 72, 81, 75, 88],
        [103, 19, 91, 89, 86, 95],
        [104, 21, 68, 74, 70, 79],
        [105, 20, 77, 80, 78, 85],
        [106, 19, 95, 92, 90, 98],
        [107, 22, 80, 76, 84, 87],
        [108, 20, 88, 85, 89, 91],
        [109, 21, 60, 65, 68, 72],
        [110, 19, 74, 79, 73, 81],
        [111, 20, 83, 77, 80, 89],
        [112, 21, 69, 72, 75, 78],
        [113, 19, 92, 88, 91, 96],
        [114, 20, 78, 82, 79, 86],
        [115, 22, 86, 90, 85, 93],
        [116, 19, 71, 68, 74, 80],
        [117, 21, 89, 84, 88, 94],
        [118, 20, 63, 70, 66, 75],
        [119, 22, 81, 79, 83, 88],
        [120, 19, 76, 85, 77, 84]
    ])

    return students


# -----------------------------------------
# Display Dataset
# -----------------------------------------
def display_dataset(students):

    print("\n========== STUDENT DATASET ==========")

    print(
        "Student ID | Age | Math | Science | English | AI"
    )

    print("-" * 55)

    for student in students:
        print(
            f"{student[0]:10} | "
            f"{student[1]:3} | "
            f"{student[2]:4} | "
            f"{student[3]:7} | "
            f"{student[4]:7} | "
            f"{student[5]:2}"
        )


# -----------------------------------------
# Display NumPy Information
# -----------------------------------------
def display_numpy_information(students):

    print("\n========== NUMPY INFORMATION ==========")

    print("Shape      :", students.shape)
    print("Dimensions :", students.ndim)
    print("Size       :", students.size)
    print("Data Type  :", students.dtype)


# -----------------------------------------
# Compare Python List and NumPy Array
# -----------------------------------------
def compare_list_numpy():

    python_list = [10, 20, 30, 40, 50]

    numpy_array = np.array([10, 20, 30, 40, 50])

    print("\n========== PYTHON LIST VS NUMPY ARRAY ==========")

    print("\nPython List:")
    print(python_list)
    print("Type:", type(python_list))

    print("\nNumPy Array:")
    print(numpy_array)
    print("Type:", type(numpy_array))

    # List multiplication
    print("\nList * 2:")
    print(python_list * 2)

    # NumPy multiplication
    print("\nNumPy Array * 2:")
    print(numpy_array * 2)


# -----------------------------------------
# Create 3D Classroom Dataset
# -----------------------------------------
def create_classrooms(students):

    # Split 20 students into two classrooms
    classroom_1 = students[:10]
    classroom_2 = students[10:]

    classrooms = np.array([
        classroom_1,
        classroom_2
    ])

    print("\n========== 3D CLASSROOM DATASET ==========")

    print("3D Array:")
    print(classrooms)

    print("\nShape      :", classrooms.shape)
    print("Dimensions :", classrooms.ndim)
    print("Size       :", classrooms.size)

    return classrooms


# -----------------------------------------
# Main Function
# -----------------------------------------
def main():

    print("==========================================")
    print("     STUDENT DATASET MANAGEMENT SYSTEM")
    print("==========================================")

    # Create dataset
    students = create_dataset()

    # Display dataset
    display_dataset(students)

    # Display NumPy information
    display_numpy_information(students)

    # Compare List and NumPy Array
    compare_list_numpy()

    # Bonus: 3D classroom array
    create_classrooms(students)


# -----------------------------------------
# Run Program
# -----------------------------------------
if __name__ == "__main__":
    main()