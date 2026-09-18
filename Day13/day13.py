import pandas as pd


# Create student data
student_names = pd.Series([
    "Kamal",
    "Arun",
    "Siva",
    "John",
    "David",
    "Ravi",
    "Nimal",
    "Alex",
    "Kevin",
    "Daniel"
])

ages = pd.Series([
    20, 21, 19, 22, 20,
    21, 23, 19, 22, 20
])

python_marks = pd.Series([
    85, 72, 91, 68, 77,
    95, 80, 88, 60, 74
])

ai_marks = pd.Series([
    90, 75, 94, 70, 82,
    92, 85, 89, 65, 78
])


# Create Student IDs
student_ids = [
    "ST001", "ST002", "ST003", "ST004", "ST005",
    "ST006", "ST007", "ST008", "ST009", "ST010"
]


# --------------------------------------------------
# Display all students
# --------------------------------------------------

def display_students():

    students = pd.DataFrame({
        "Name": student_names,
        "Age": ages,
        "Python": python_marks,
        "AI": ai_marks
    }, index=student_ids)

    print("\n===== ALL STUDENTS =====")
    print(students)


# --------------------------------------------------
# Highest marks
# --------------------------------------------------

def highest_marks():

    python_index = python_marks.idxmax()
    ai_index = ai_marks.idxmax()

    print("\n===== HIGHEST MARKS =====")

    print(
        f"Highest Python Mark: "
        f"{python_marks.max()} "
        f"({student_names[python_index]})"
    )

    print(
        f"Highest AI Mark: "
        f"{ai_marks.max()} "
        f"({student_names[ai_index]})"
    )


# --------------------------------------------------
# Lowest marks
# --------------------------------------------------

def lowest_marks():

    python_index = python_marks.idxmin()
    ai_index = ai_marks.idxmin()

    print("\n===== LOWEST MARKS =====")

    print(
        f"Lowest Python Mark: "
        f"{python_marks.min()} "
        f"({student_names[python_index]})"
    )

    print(
        f"Lowest AI Mark: "
        f"{ai_marks.min()} "
        f"({student_names[ai_index]})"
    )


# --------------------------------------------------
# Average marks
# --------------------------------------------------

def average_marks():

    print("\n===== AVERAGE MARKS =====")

    print(
        f"Average Python Mark: "
        f"{python_marks.mean():.2f}"
    )

    print(
        f"Average AI Mark: "
        f"{ai_marks.mean():.2f}"
    )


# --------------------------------------------------
# Students scoring above 75
# --------------------------------------------------

def above_75():

    print("\n===== STUDENTS SCORING ABOVE 75 =====")

    python_students = student_names[python_marks > 75]
    ai_students = student_names[ai_marks > 75]

    print("\nPython:")
    print(python_students)

    print("\nAI:")
    print(ai_students)


# --------------------------------------------------
# Number of passing students
# --------------------------------------------------

def passing_students():

    # Pass mark = 50
    passed = (python_marks >= 50) & (ai_marks >= 50)

    print("\n===== PASSING STUDENTS =====")

    print(
        f"Number of passing students: "
        f"{passed.sum()}"
    )

    print("\nPassing Students:")

    print(student_names[passed])


# --------------------------------------------------
# Main
# --------------------------------------------------

def main():

    print("========================================")
    print("       STUDENT DATA EXPLORER")
    print("========================================")

    display_students()
    highest_marks()
    lowest_marks()
    average_marks()
    above_75()
    passing_students()


if __name__ == "__main__":
    main()