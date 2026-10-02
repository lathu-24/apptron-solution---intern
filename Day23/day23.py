import pandas as pd
import matplotlib.pyplot as plt

# ==========================================
# Day 23 - Student Performance Visualizer
# ==========================================

# ==========================================
# 1. Create Student Dataset
# ==========================================

data = {
    "Student Name": [
        "Liam", "Emma", "Noah", "Olivia", "William",
        "Ava", "James", "Sophia", "Benjamin", "Isabella",
        "Lucas", "Mia", "Henry", "Charlotte", "Alexander",
        "Amelia", "Daniel", "Harper", "Michael", "Evelyn"
    ],

    "Math": [
        85, 92, 76, 88, 95,
        78, 89, 96, 72, 84,
        91, 87, 79, 93, 98,
        81, 86, 90, 75, 94
    ],

    "Science": [
        82, 95, 79, 90, 93,
        80, 85, 98, 74, 88,
        89, 91, 83, 94, 97,
        84, 87, 92, 78, 96
    ],

    "Python": [
        88, 94, 81, 92, 96,
        85, 90, 99, 77, 86,
        93, 89, 84, 95, 98,
        87, 91, 94, 80, 97
    ],

    "AI": [
        90, 96, 78, 94, 97,
        82, 92, 98, 75, 89,
        95, 91, 80, 96, 99,
        86, 90, 93, 79, 98
    ]
}

df = pd.DataFrame(data)


# ==========================================
# 2. Calculate Total and Average Marks
# ==========================================

subjects = [
    "Math",
    "Science",
    "Python",
    "AI"
]

df["Total Marks"] = df[subjects].sum(axis=1)

df["Average Marks"] = df[subjects].mean(axis=1).round(2)


# ==========================================
# 3. Assign Grades
# ==========================================

def calculate_grade(average):

    if average >= 90:
        return "A"

    elif average >= 80:
        return "B"

    elif average >= 70:
        return "C"

    elif average >= 60:
        return "D"

    else:
        return "F"


df["Grade"] = df["Average Marks"].apply(
    calculate_grade
)


# ==========================================
# 4. Display Dataset
# ==========================================

print("=" * 60)
print("STUDENT PERFORMANCE DATA")
print("=" * 60)

print(df.to_string(index=False))


# ==========================================
# 5. Line Chart
# Student vs Total Marks
# ==========================================

plt.figure(figsize=(12, 6))

plt.plot(
    df["Student Name"],
    df["Total Marks"],
    marker="o",
    label="Total Marks"
)

plt.title("Student vs Total Marks")
plt.xlabel("Student Name")
plt.ylabel("Total Marks")
plt.xticks(rotation=45)
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.show()


# ==========================================
# 6. Bar Chart
# Student vs Average Marks
# ==========================================

plt.figure(figsize=(12, 6))

plt.bar(
    df["Student Name"],
    df["Average Marks"],
    label="Average Marks"
)

plt.title("Student vs Average Marks")
plt.xlabel("Student Name")
plt.ylabel("Average Marks")
plt.xticks(rotation=45)
plt.legend()

plt.tight_layout()
plt.show()


# ==========================================
# 7. Pie Chart
# Grade Distribution
# ==========================================

grade_distribution = df["Grade"].value_counts()

plt.figure(figsize=(7, 7))

plt.pie(
    grade_distribution.values,
    labels=grade_distribution.index,
    autopct="%1.1f%%",
    startangle=90
)

plt.title("Student Grade Distribution")
plt.legend(
    title="Grades",
    loc="best"
)

plt.show()


# ==========================================
# 8. Top 5 Students
# ==========================================

top_5 = df.sort_values(
    "Total Marks",
    ascending=False
).head(5)

print("\n" + "=" * 60)
print("TOP 5 STUDENTS")
print("=" * 60)

print(
    top_5[
        [
            "Student Name",
            "Total Marks",
            "Average Marks",
            "Grade"
        ]
    ].to_string(index=False)
)


# ==========================================
# 9. Top 5 Visualization
# ==========================================

plt.figure(figsize=(9, 6))

plt.bar(
    top_5["Student Name"],
    top_5["Total Marks"],
    label="Total Marks"
)

plt.title("Top 5 Students - Total Marks")
plt.xlabel("Student Name")
plt.ylabel("Total Marks")
plt.legend()

plt.tight_layout()
plt.show()


# ==========================================
# 10. Save Dataset
# ==========================================

df.to_csv(
    "student_performance.csv",
    index=False
)

top_5.to_csv(
    "top_5_students.csv",
    index=False
)

print("\nFiles generated:")
print("1. student_performance.csv")
print("2. top_5_students.csv")

print("\nStudent performance visualization completed!")