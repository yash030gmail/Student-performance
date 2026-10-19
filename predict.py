"""
Inference and Early Intervention Prediction Script.
Uses trained AIML models to predict:
  1. Estimated Scores in Math, Reading, Writing, and Average Overall Score.
  2. Academic Risk Status (Pass / Fail Probability).
  3. Performance Tier (High Performer, Moderate Performer, At-Risk).
  4. Personalized Pedagogical Recommendations for Teachers & Counselors.
"""

import os
import argparse
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(__file__)
MODEL_DIR = os.path.join(BASE_DIR, "models")

# Load trained pipelines
reg_model = joblib.load(os.path.join(MODEL_DIR, "best_overall_score_regressor.joblib"))
multi_reg_model = joblib.load(os.path.join(MODEL_DIR, "multi_subject_regressor.joblib"))
cls_model = joblib.load(os.path.join(MODEL_DIR, "best_pass_fail_classifier.joblib"))
tier_model = joblib.load(os.path.join(MODEL_DIR, "tier_classifier.joblib"))
math_predictor = joblib.load(os.path.join(MODEL_DIR, "math_score_predictor.joblib"))

def predict_student(gender, race_ethnicity, parental_education, lunch, test_prep, reading_score=None, writing_score=None):
    """
    Generate complete AI performance predictions for an individual student.
    """
    input_data = pd.DataFrame([{
        'gender': gender,
        'race/ethnicity': race_ethnicity,
        'parental level of education': parental_education,
        'lunch': lunch,
        'test preparation course': test_prep
    }])

    # 1. Multi-subject score predictions from demographic factors
    multi_preds = multi_reg_model.predict(input_data)[0]
    pred_math_demo = max(0.0, min(100.0, multi_preds[0]))
    pred_reading_demo = max(0.0, min(100.0, multi_preds[1]))
    pred_writing_demo = max(0.0, min(100.0, multi_preds[2]))
    pred_overall = (pred_math_demo + pred_reading_demo + pred_writing_demo) / 3.0

    # 2. If reading & writing scores are already known, predict refined math score
    refined_math = None
    if reading_score is not None and writing_score is not None:
        refined_input = pd.DataFrame([{
            'gender': gender,
            'race/ethnicity': race_ethnicity,
            'parental level of education': parental_education,
            'lunch': lunch,
            'test preparation course': test_prep,
            'reading score': float(reading_score),
            'writing score': float(writing_score)
        }])
        refined_math = max(0.0, min(100.0, math_predictor.predict(refined_input)[0]))

    # 3. Pass/Fail Probability
    pass_prob = cls_model.predict_proba(input_data)[0][1]
    is_passing = bool(pass_prob >= 0.5)

    # 4. Performance Tier
    tier = tier_model.predict(input_data)[0]

    # 5. Early Intervention Recommendations
    recommendations = []
    if test_prep == 'none':
        recommendations.append("Enroll in Test Preparation Course: Data indicates an expected gain of +7.6 to +10 points across subjects.")
    if lunch == 'free/reduced':
        recommendations.append("Socioeconomic Assistance: Ensure student has access to subsidized nutrition, learning materials, and after-school study spaces.")
    if (refined_math and refined_math < 60) or pred_math_demo < 60:
        recommendations.append("Targeted Math Support: Flag for foundational math tutorials and peer mentoring.")
    if pred_overall >= 80:
        recommendations.append("Advanced Placement: Candidate for honors / accelerated academic challenges.")

    result = {
        "student_profile": {
            "gender": gender,
            "race/ethnicity": race_ethnicity,
            "parental level of education": parental_education,
            "lunch": lunch,
            "test preparation course": test_prep
        },
        "forecasted_scores": {
            "math_score": float(round(pred_math_demo, 1)),
            "reading_score": float(round(pred_reading_demo, 1)),
            "writing_score": float(round(pred_writing_demo, 1)),
            "overall_average_score": float(round(pred_overall, 1)),
            "refined_math_score_given_reading_writing": float(round(refined_math, 1)) if refined_math is not None else None
        },
        "academic_risk_assessment": {
            "pass_probability": float(round(pass_prob * 100, 1)),
            "pass_status": "Likely to Pass" if is_passing else "At-Risk of Failing",
            "predicted_tier": str(tier)
        },
        "actionable_recommendations": recommendations
    }

    return result

def print_student_report(res):
    print("=" * 65)
    print("      STUDENT PERFORMANCE AI PREDICTION REPORT")
    print("=" * 65)
    prof = res['student_profile']
    print(f"Profile: {prof['gender'].capitalize()} | {prof['race/ethnicity'].title()} | {prof['parental level of education']} | Lunch: {prof['lunch']} | Test Prep: {prof['test preparation course']}")
    print("-" * 65)
    
    scores = res['forecasted_scores']
    print("PREDICTED SUBJECT SCORES (Demographic Baseline):")
    print(f"  * Mathematics Score:  {scores['math_score']:5.1f} / 100")
    print(f"  * Reading Score:      {scores['reading_score']:5.1f} / 100")
    print(f"  * Writing Score:      {scores['writing_score']:5.1f} / 100")
    print(f"  * Overall Average:    {scores['overall_average_score']:5.1f} / 100")
    
    if scores['refined_math_score_given_reading_writing'] is not None:
        print(f"  * Refined Math Score (using Reading/Writing): {scores['refined_math_score_given_reading_writing']:5.1f} / 100")

    risk = res['academic_risk_assessment']
    print("\nACADEMIC RISK & TIER CLASSIFICATION:")
    print(f"  * Pass Probability:   {risk['pass_probability']}%")
    print(f"  * Status:             {risk['pass_status']}")
    print(f"  * Performance Tier:   {risk['predicted_tier']}")

    recs = res['actionable_recommendations']
    if recs:
        print("\nACTIONABLE INTERVENTIONS:")
        for idx, rec in enumerate(recs, 1):
            print(f"  [{idx}] {rec}")
    print("=" * 65)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Student Performance Prediction CLI")
    parser.add_argument("--gender", type=str, default="female", choices=["female", "male"])
    parser.add_argument("--race", type=str, default="group C", choices=["group A", "group B", "group C", "group D", "group E"])
    parser.add_argument("--education", type=str, default="some college",
                        choices=["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"])
    parser.add_argument("--lunch", type=str, default="standard", choices=["standard", "free/reduced"])
    parser.add_argument("--prep", type=str, default="completed", choices=["none", "completed"])
    parser.add_argument("--reading", type=float, default=None, help="Optional known reading score (0-100)")
    parser.add_argument("--writing", type=float, default=None, help="Optional known writing score (0-100)")

    args = parser.parse_args()

    # Run single student test
    result = predict_student(
        gender=args.gender,
        race_ethnicity=args.race,
        parental_education=args.education,
        lunch=args.lunch,
        test_prep=args.prep,
        reading_score=args.reading,
        writing_score=args.writing
    )

    print_student_report(result)
