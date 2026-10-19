"""
Exploratory Data Analysis (EDA) for Student Performance Prediction.
Generates comprehensive statistical summaries and publication-quality visual plots.
"""

import os
import json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np

# Set aesthetic style
sns.set_theme(style="whitegrid", palette="muted")
plt.rcParams.update({'font.sans-serif': 'Arial', 'font.size': 11})

DATA_PATH = os.path.join(os.path.dirname(__file__), "StudentsPerformance.csv")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "visualizations")
os.makedirs(OUTPUT_DIR, exist_ok=True)

def run_eda():
    df = pd.read_csv(DATA_PATH)
    print(f"Dataset successfully loaded: {df.shape[0]} rows, {df.shape[1]} columns")

    # Add engineered features
    df['total_score'] = df['math score'] + df['reading score'] + df['writing score']
    df['average_score'] = df['total_score'] / 3.0
    df['passed_all'] = (
        (df['math score'] >= 50) & 
        (df['reading score'] >= 50) & 
        (df['writing score'] >= 50)
    ).astype(int)

    def categorize_tier(avg):
        if avg >= 80:
            return 'High Performer'
        elif avg >= 60:
            return 'Moderate Performer'
        else:
            return 'At-Risk / Needs Support'

    df['performance_tier'] = df['average_score'].apply(categorize_tier)

    # 1. Statistical Summary
    stats_dict = {
        "dataset_shape": list(df.shape),
        "columns": list(df.columns),
        "score_summary": {
            col: {
                "mean": float(df[col].mean()),
                "std": float(df[col].std()),
                "median": float(df[col].median()),
                "min": float(df[col].min()),
                "max": float(df[col].max()),
                "q25": float(df[col].quantile(0.25)),
                "q75": float(df[col].quantile(0.75))
            } for col in ['math score', 'reading score', 'writing score', 'average_score']
        },
        "pass_rate_all_subjects": float(df['passed_all'].mean() * 100),
        "tier_distribution": df['performance_tier'].value_counts().to_dict(),
        "gender_breakdown": df.groupby('gender')[['math score', 'reading score', 'writing score', 'average_score']].mean().to_dict(),
        "lunch_breakdown": df.groupby('lunch')[['math score', 'reading score', 'writing score', 'average_score']].mean().to_dict(),
        "prep_course_breakdown": df.groupby('test preparation course')[['math score', 'reading score', 'writing score', 'average_score']].mean().to_dict(),
        "education_breakdown": df.groupby('parental level of education')[['math score', 'reading score', 'writing score', 'average_score']].mean().to_dict()
    }

    with open(os.path.join(OUTPUT_DIR, "eda_statistics.json"), "w") as f:
        json.dump(stats_dict, f, indent=4)
    print("Saved eda_statistics.json")

    # ---------------- PLOT 1: Score Distributions and KDEs ----------------
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    scores = ['math score', 'reading score', 'writing score']
    colors = ['#2b5c8f', '#2a9d8f', '#e76f51']

    for i, col in enumerate(scores):
        sns.histplot(df[col], kde=True, ax=axes[i], color=colors[i], bins=20, edgecolor='black', alpha=0.6)
        axes[i].axvline(df[col].mean(), color='red', linestyle='--', linewidth=1.5, label=f"Mean: {df[col].mean():.1f}")
        axes[i].axvline(df[col].median(), color='green', linestyle=':', linewidth=1.5, label=f"Median: {df[col].median():.1f}")
        axes[i].set_title(f"Distribution of {col.title()}", fontsize=13, fontweight='bold')
        axes[i].set_xlabel("Score (0 - 100)")
        axes[i].set_ylabel("Count")
        axes[i].legend()

    plt.tight_layout()
    plot1_path = os.path.join(OUTPUT_DIR, "01_score_distributions.png")
    plt.savefig(plot1_path, dpi=300)
    plt.close()
    print(f"Saved {plot1_path}")

    # ---------------- PLOT 2: Correlation Heatmap ----------------
    fig, ax = plt.subplots(figsize=(7, 6))
    corr = df[scores + ['average_score']].corr()
    sns.heatmap(corr, annot=True, cmap="YlGnBu", fmt=".3f", square=True, 
                cbar_kws={'shrink': .8}, linewidths=1, ax=ax)
    ax.set_title("Correlation Heatmap Across Exam Scores", fontsize=13, fontweight='bold', pad=12)
    plt.tight_layout()
    plot2_path = os.path.join(OUTPUT_DIR, "02_correlation_heatmap.png")
    plt.savefig(plot2_path, dpi=300)
    plt.close()
    print(f"Saved {plot2_path}")

    # ---------------- PLOT 3: Impact of Socioeconomic & Preparation Factors ----------------
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))

    # Test Prep vs Average Score
    sns.boxplot(x='test preparation course', y='average_score', hue='gender', data=df, 
                palette=['#3a86ff', '#ff006e'], ax=axes[0, 0])
    axes[0, 0].set_title("Impact of Test Preparation Course & Gender", fontsize=12, fontweight='bold')
    axes[0, 0].set_ylabel("Average Score")
    axes[0, 0].set_xlabel("Test Preparation Course")

    # Lunch vs Average Score
    sns.boxplot(x='lunch', y='average_score', data=df, palette=['#06d6a0', '#ef476f'], ax=axes[0, 1])
    axes[0, 1].set_title("Impact of Lunch Type (Socioeconomic Indicator)", fontsize=12, fontweight='bold')
    axes[0, 1].set_ylabel("Average Score")
    axes[0, 1].set_xlabel("Lunch Type")

    # Parental Level of Education
    edu_order = ["some high school", "high school", "some college", "associate's degree", "bachelor's degree", "master's degree"]
    sns.barplot(x='parental level of education', y='average_score', data=df, order=edu_order,
                palette="Blues_d", ax=axes[1, 0], errorbar=None)
    axes[1, 0].set_title("Average Score by Parental Level of Education", fontsize=12, fontweight='bold')
    axes[1, 0].set_ylabel("Average Score")
    axes[1, 0].set_xlabel("Parental Education")
    axes[1, 0].tick_params(axis='x', rotation=25)

    # Race/Ethnicity vs Average Score
    race_order = sorted(df['race/ethnicity'].unique())
    sns.barplot(x='race/ethnicity', y='average_score', data=df, order=race_order,
                palette="Spectral", ax=axes[1, 1], errorbar=None)
    axes[1, 1].set_title("Average Score Across Race / Ethnicity Groups", fontsize=12, fontweight='bold')
    axes[1, 1].set_ylabel("Average Score")
    axes[1, 1].set_xlabel("Race/Ethnicity Group")

    plt.tight_layout()
    plot3_path = os.path.join(OUTPUT_DIR, "03_socioeconomic_factors.png")
    plt.savefig(plot3_path, dpi=300)
    plt.close()
    print(f"Saved {plot3_path}")

    # ---------------- PLOT 4: Academic Performance Tiers & Pass Rates ----------------
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Tier Pie chart
    tier_counts = df['performance_tier'].value_counts()
    colors_pie = ['#2a9d8f', '#e9c46a', '#e76f51']
    axes[0].pie(tier_counts, labels=tier_counts.index, autopct='%1.1f%%', startangle=140, 
                colors=colors_pie, wedgeprops={'edgecolor': 'white', 'linewidth': 2})
    axes[0].set_title("Student Distribution by Performance Tier", fontsize=13, fontweight='bold')

    # Pass Rate by Lunch & Test Prep
    pass_summary = df.groupby(['lunch', 'test preparation course'])['passed_all'].mean().reset_index()
    pass_summary['pass_percentage'] = pass_summary['passed_all'] * 100
    sns.barplot(x='lunch', y='pass_percentage', hue='test preparation course', data=pass_summary,
                palette=['#f4a261', '#264653'], ax=axes[1])
    axes[1].set_title("Pass Rate (Scores >= 50 in All Subjects) (%)", fontsize=13, fontweight='bold')
    axes[1].set_ylabel("Pass Rate (%)")
    axes[1].set_ylim(0, 105)
    for p in axes[1].patches:
        height = p.get_height()
        if height > 0:
            axes[1].annotate(f'{height:.1f}%',
                            (p.get_x() + p.get_width() / 2., height / 2),
                            ha='center', va='center', color='white', fontweight='bold', fontsize=11)

    plt.tight_layout()
    plot4_path = os.path.join(OUTPUT_DIR, "04_tiers_and_pass_rates.png")
    plt.savefig(plot4_path, dpi=300)
    plt.close()
    print(f"Saved {plot4_path}")

    print("EDA completed successfully!")

if __name__ == "__main__":
    run_eda()
