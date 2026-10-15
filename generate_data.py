"""
generate_data.py
----------------
Fallback synthetic dataset generator for Placement Prediction.
Generates realistic student records matching the 32-column schema
and writes to data/placement_data.csv when the dataset is missing.
"""

from pathlib import Path
import random
import argparse
import pandas as pd
import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parent
DATA_DIR = PROJECT_ROOT / "data"
DEFAULT_OUTPUT = DATA_DIR / "placement_data.csv"

CITIES = ["Ahmedabad", "Bangalore", "Chennai", "Delhi", "Hyderabad", "Kolkata", "Mumbai", "Pune"]
COLLEGE_TIERS = ["Tier1", "Tier2", "Tier3"]
STREAMS = ["CS", "IT", "ECE", "EEE", "Mechanical", "Civil"]
SPECIALISATIONS = ["DataScience", "AI_ML", "WebDevelopment", "CloudComputing", "Networking", "CyberSecurity", "General"]
GENDERS = ["Male", "Female"]
YES_NO = ["Yes", "No"]


def generate_synthetic_data(num_samples: int = 5000, output_path: Path = DEFAULT_OUTPUT, random_state: int = 42) -> pd.DataFrame:
    """
    Generates synthetic placement dataset with realistic distributions and correlations.
    """
    np.random.seed(random_state)
    random.seed(random_state)
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    records = []
    for i in range(1, num_samples + 1):
        student_id = i
        gender = random.choices(GENDERS, weights=[0.55, 0.45])[0]
        city = random.choice(CITIES)
        college_tier = random.choices(COLLEGE_TIERS, weights=[0.25, 0.45, 0.30])[0]
        stream = random.choices(STREAMS, weights=[0.35, 0.25, 0.15, 0.10, 0.10, 0.05])[0]
        specialisation = random.choice(SPECIALISATIONS)
        hostel = random.choice(YES_NO)
        backlog_history = random.choices(YES_NO, weights=[0.20, 0.80])[0]

        # Base academic aptitude (latent variable)
        base_ability = np.random.normal(loc=0.0, scale=1.0)
        if college_tier == "Tier1":
            base_ability += 0.5
        elif college_tier == "Tier3":
            base_ability -= 0.4

        # Generate SGPAs and CGPA
        sgpas = []
        for sem in range(1, 9):
            sem_sgpa = np.clip(np.random.normal(loc=6.8 + base_ability * 0.8, scale=0.9), 4.0, 10.0)
            sgpas.append(round(float(sem_sgpa), 2))
        cgpa = round(float(np.mean(sgpas)), 2)

        # Categorize CGPA Tier
        cgpa_tier = "High" if cgpa >= 8.0 else ("Mid" if cgpa >= 6.5 else "Low")

        attendance = round(float(np.clip(np.random.normal(loc=75 + base_ability * 5, scale=10), 45.0, 100.0)), 1)
        internships = int(np.clip(np.random.poisson(lam=max(0.2, 1.0 + base_ability * 0.6)), 0, 6))
        projects = int(np.clip(np.random.poisson(lam=max(0.5, 2.0 + base_ability * 0.5)), 0, 8))
        workshops = int(np.clip(np.random.poisson(lam=1.5), 0, 6))
        certifications = int(np.clip(np.random.poisson(lam=max(0.2, 1.2 + base_ability * 0.4)), 0, 7))
        publications = int(np.clip(np.random.poisson(lam=0.2 if college_tier == "Tier1" else 0.05), 0, 4))

        aptitude_score = round(float(np.clip(np.random.normal(loc=60 + base_ability * 12, scale=12), 20.0, 100.0)), 1)
        soft_skills = round(float(np.clip(np.random.normal(loc=3.2 + base_ability * 0.4, scale=0.7), 1.0, 5.0)), 1)
        coding_score = round(float(np.clip(np.random.normal(loc=55 + base_ability * 15, scale=15), 10.0, 100.0)), 1)
        mock_interview = round(float(np.clip(np.random.normal(loc=58 + base_ability * 10, scale=12), 20.0, 100.0)), 1)
        extracurricular = random.choices([0, 1], weights=[0.4, 0.6])[0]

        # Calculate Placement Probability using logistic sigmoid
        logit = (
            -5.5
            + 0.55 * (cgpa - 5.0)
            + 0.03 * (coding_score - 50)
            + 0.02 * (aptitude_score - 50)
            + 0.50 * internships
            + 0.25 * projects
            + 0.20 * certifications
            + (0.6 if college_tier == "Tier1" else (-0.5 if college_tier == "Tier3" else 0.0))
            - (1.2 if backlog_history == "Yes" else 0.0)
        )
        prob = 1.0 / (1.0 + np.exp(-logit))
        placement_status = 1 if np.random.rand() < prob else 0

        # Salary calculation (only for placed candidates)
        if placement_status == 1:
            base_salary = 3.5 + 0.6 * (cgpa - 6.0) + 0.05 * coding_score + 0.8 * internships
            if college_tier == "Tier1":
                base_salary += 2.5
            salary_package = round(float(max(3.0, base_salary + np.random.normal(0, 0.8))), 2)
        else:
            salary_package = 0.0

        is_anomaly = 1 if (placement_status == 0 and cgpa > 9.0 and coding_score > 85) else 0

        record = {
            "StudentID": student_id,
            "Gender": gender,
            "City": city,
            "CollegeTier": college_tier,
            "Stream": stream,
            "Specialisation": specialisation,
            "Hostel": hostel,
            "HistoryOfBacklogs": backlog_history,
            "SGPA_Sem1": sgpas[0],
            "SGPA_Sem2": sgpas[1],
            "SGPA_Sem3": sgpas[2],
            "SGPA_Sem4": sgpas[3],
            "SGPA_Sem5": sgpas[4],
            "SGPA_Sem6": sgpas[5],
            "SGPA_Sem7": sgpas[6],
            "SGPA_Sem8": sgpas[7],
            "CGPA": cgpa,
            "AttendancePercent": attendance,
            "Internships": internships,
            "Projects": projects,
            "Workshops": workshops if random.random() > 0.05 else np.nan,
            "Certifications": certifications,
            "Publications": publications,
            "AptitudeTestScore": aptitude_score if random.random() > 0.03 else np.nan,
            "SoftSkillsRating": soft_skills if random.random() > 0.03 else np.nan,
            "CodingTestScore": coding_score if random.random() > 0.03 else np.nan,
            "MockInterviewScore": mock_interview if random.random() > 0.03 else np.nan,
            "ExtraCurricular": extracurricular,
            "CGPA_Tier": cgpa_tier,
            "PlacementStatus": placement_status,
            "IsAnomaly": is_anomaly,
            "Salary Package": salary_package,
        }
        records.append(record)

    df = pd.DataFrame(records)
    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        print(f"Generated {len(df)} synthetic records -> {output_path}")

    return df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Synthetic Placement Dataset Generator")
    parser.add_argument("--samples", type=int, default=5000, help="Number of records to generate")
    parser.add_argument("--output", type=str, default=str(DEFAULT_OUTPUT), help="Output CSV path")
    args = parser.parse_args()

    generate_synthetic_data(num_samples=args.samples, output_path=Path(args.output))
