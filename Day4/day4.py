import numpy as np


# ------------------------------------------------
# Create Patient Dataset
# ------------------------------------------------
def create_dataset():

    patients = np.array([
        [1001, 25, 118, 85, 72, 22.5],
        [1002, 34, 125, 92, 78, 24.1],
        [1003, 45, 140, 110, 85, 27.8],
        [1004, 52, 135, 105, 88, 29.2],
        [1005, 67, 155, 130, 92, 31.5],
        [1006, 29, 120, 88, 75, 23.4],
        [1007, 72, 160, 145, 95, 33.1],
        [1008, 41, 128, 98, 80, 26.2],
        [1009, 63, 148, 120, 89, 30.4],
        [1010, 55, 138, 115, 87, 28.9],
        [1011, 38, 122, 90, 76, 25.1],
        [1012, 69, 152, 135, 91, 32.2],
        [1013, 47, 130, 100, 82, 27.0],
        [1014, 61, 145, 125, 90, 30.1],
        [1015, 75, 165, 150, 98, 34.5],
        [1016, 33, 119, 87, 73, 22.8],
        [1017, 58, 142, 108, 86, 29.7],
        [1018, 66, 150, 128, 93, 31.8],
        [1019, 44, 126, 95, 79, 25.8],
        [1020, 70, 158, 140, 94, 32.7]
    ])

    return patients


# ------------------------------------------------
# Display Dataset
# ------------------------------------------------
def display_dataset(patients):

    print("\n========== PATIENT DATASET ==========")

    print(
        "Patient ID | Age | Blood Pressure | "
        "Sugar Level | Heart Rate | BMI"
    )

    print("-" * 70)

    for patient in patients:
        print(
            f"{patient[0]:10} | "
            f"{patient[1]:3} | "
            f"{patient[2]:14} | "
            f"{patient[3]:11} | "
            f"{patient[4]:10} | "
            f"{patient[5]:.1f}"
        )


# ------------------------------------------------
# First and Last Patient
# ------------------------------------------------
def first_last_patient(patients):

    print("\n========== FIRST PATIENT ==========")
    print(patients[0])

    print("\n========== LAST PATIENT ==========")
    print(patients[-1])


# ------------------------------------------------
# Display All Blood Pressure Values
# ------------------------------------------------
def display_blood_pressure(patients):

    blood_pressure = patients[:, 2]

    print("\n========== BLOOD PRESSURE ==========")
    print(blood_pressure)


# ------------------------------------------------
# Display Age and BMI
# ------------------------------------------------
def display_age_bmi(patients):

    age_bmi = patients[:, [1, 5]]

    print("\n========== AGE AND BMI ==========")
    print(age_bmi)


# ------------------------------------------------
# Retrieve Patients 5-15
# ------------------------------------------------
def retrieve_patients(patients):

    selected = patients[4:15]

    print("\n========== PATIENTS 5-15 ==========")
    print(selected)


# ------------------------------------------------
# High Blood Pressure Patients
# ------------------------------------------------
def high_blood_pressure(patients):

    high_bp = patients[patients[:, 2] >= 140]

    print("\n========== HIGH BLOOD PRESSURE ==========")
    print(high_bp)


# ------------------------------------------------
# Senior Citizens
# ------------------------------------------------
def senior_citizens(patients):

    seniors = patients[patients[:, 1] > 60]

    print("\n========== SENIOR CITIZENS ==========")
    print(seniors)


# ------------------------------------------------
# High Blood Pressure AND High Sugar
# ------------------------------------------------
def high_bp_and_sugar(patients):

    result = patients[
        (patients[:, 2] >= 140) &
        (patients[:, 3] >= 120)
    ]

    print("\n========== HIGH BP AND HIGH SUGAR ==========")
    print(result)


# ------------------------------------------------
# Main Function
# ------------------------------------------------
def main():

    print("==========================================")
    print("   HOSPITAL PATIENT RECORD EXPLORER")
    print("==========================================")

    patients = create_dataset()

    display_dataset(patients)

    first_last_patient(patients)

    display_blood_pressure(patients)

    display_age_bmi(patients)

    retrieve_patients(patients)

    high_blood_pressure(patients)

    senior_citizens(patients)

    high_bp_and_sugar(patients)


# ------------------------------------------------
# Run Program
# ------------------------------------------------
if __name__ == "__main__":
    main()