import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# Configuration for beautiful plots
plt.style.use('seaborn-v0_8-whitegrid')
sns.set_palette("viridis")

# Load data
print("Loading data...")
try:
    df = pd.read_csv("dados_grade_invertida.csv")
except FileNotFoundError:
    try:
        df = pd.read_csv("Student_performance_data.csv")
    except FileNotFoundError:
        print("Error: Could not find the dataset CSV file.")
        exit()

# If GPA is present, drop it since we discussed it's a direct mapping of GradeClass
if "GPA" in df.columns:
    df = df.drop("GPA", axis=1)

# Drop StudentID for visualizations as it's not a real feature
if "StudentID" in df.columns:
    df = df.drop("StudentID", axis=1)


# ==========================================
# 1. Target Distribution (GradeClass)
# ==========================================
plt.figure(figsize=(8, 6))
sns.countplot(data=df, x='GradeClass', hue='GradeClass', palette="rocket", legend=False)
plt.title('Distribution of Grade Classes', fontsize=16)
plt.xlabel('Grade Class')
plt.ylabel('Number of Students')
plt.tight_layout()
plt.savefig("vis_1_grade_distribution.png", dpi=300)
plt.close()
print("Saved vis_1_grade_distribution.png")

# ==========================================
# 2. Study Time vs Grade Class
# ==========================================
plt.figure(figsize=(8, 6))
sns.boxplot(data=df, x='GradeClass', y='StudyTimeWeekly', hue='GradeClass', palette="mako", legend=False)
plt.title('Weekly Study Time by Grade Class', fontsize=16)
plt.xlabel('Grade Class')
plt.ylabel('Study Hours per Week')
plt.tight_layout()
plt.savefig("vis_2_study_time.png", dpi=300)
plt.close()
print("Saved vis_2_study_time.png")

# ==========================================
# 3. Absences vs Grade Class
# ==========================================
plt.figure(figsize=(8, 6))
sns.violinplot(data=df, x='GradeClass', y='Absences', hue='GradeClass', palette="magma", legend=False)
plt.title('Absences by Grade Class', fontsize=16)
plt.xlabel('Grade Class')
plt.ylabel('Total Absences')
plt.tight_layout()
plt.savefig("vis_3_absences.png", dpi=300)
plt.close()
print("Saved vis_3_absences.png")

# ==========================================
# 4. Impact of Parental Support
# ==========================================
plt.figure(figsize=(8, 6))
parent_support_avg = df.groupby('ParentalSupport')['GradeClass'].mean().reset_index()
sns.barplot(data=parent_support_avg, x='ParentalSupport', y='GradeClass', hue='ParentalSupport', palette="crest", legend=False)
plt.title('Average Grade Class by Parental Support', fontsize=16)
plt.xlabel('Parental Support Level (0=None, 4=High)')
plt.ylabel('Average Grade Class')
plt.tight_layout()
plt.savefig("vis_4_parental_support.png", dpi=300)
plt.close()
print("Saved vis_4_parental_support.png")

# ==========================================
# 5. Correlation Heatmap (Numerical features)
# ==========================================
plt.figure(figsize=(8, 6))
corr_cols = ['Age', 'StudyTimeWeekly', 'Absences', 'ParentalEducation', 'ParentalSupport', 'GradeClass']
corr_matrix = df[corr_cols].corr()
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', fmt='.2f', linewidths=0.5)
plt.title('Feature Correlation Heatmap', fontsize=16)
plt.tight_layout()
plt.savefig("vis_5_correlation.png", dpi=300)
plt.close()
print("Saved vis_5_correlation.png")

# ==========================================
# 6. Tutoring Impact
# ==========================================
plt.figure(figsize=(8, 6))
sns.histplot(data=df, x='GradeClass', hue='Tutoring', multiple='dodge', shrink=.8, palette='Set2')
plt.title('Grade Distribution by Tutoring Status', fontsize=16)
plt.xlabel('Grade Class')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig("vis_6_tutoring.png", dpi=300)
plt.close()
print("Saved vis_6_tutoring.png")

print("All individual visualizations successfully generated!")