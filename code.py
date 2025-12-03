import pandas as pd
import statsmodels.formula.api as smf

df = pd.read_csv('attendance_rollcall.csv')
df = df.rename(columns={df.columns[0]: 'user_id'})

# Convert to long (panel) format
long = df.melt(id_vars=['user_id'],
               var_name='lecture_date',
               value_name='attended')

# Convert attendance to 0/1 for the regression
long['attended'] = (
    long['attended']
    .astype(str)
    .str.lower()
    .map({'yes': 1, 'no': 0})
)

#removing nan values and converting to int type
long = long.dropna(subset=['attended'])
long['attended'] = long['attended'].astype(int)

print(f"Mean Attendance = {long['attended'].mean()}")


# Parsing dates
long['lecture_date'] = pd.to_datetime(long['lecture_date'])
long['date_str']     = long['lecture_date'].dt.strftime('%Y-%m-%d')

# ---- ESTIMATE FIXED-EFFECTS LINEAR PROBABILITY MODEL ----
#this model drops the first entries in both the student and lecture fixed effects as baseline to avoid perfect multicollinearity
#hence the fixed effects for the respective first entries are 0

model = smf.ols("attended ~ C(user_id) + C(date_str)", data=long).fit(cov_type='HC1')

# Extract FE coefficients
params = model.params

student_fe = params[params.index.str.startswith("C(user_id)")]
date_fe    = params[params.index.str.startswith("C(date_str)")]

# ---- Cleaning data for the output ----
student_fe.index = (
    student_fe.index
    .str.replace("C(user_id)[T.", "", regex=False)
    .str.replace("]", "", regex=False)
)

date_fe.index = (
    date_fe.index
    .str.replace("C(date_str)[T.", "", regex=False)
    .str.replace("]", "", regex=False)
)

# Output DataFrames
student_df = student_fe.reset_index()
student_df.columns = ["student_id", "student_fixed_effect"]

date_df = date_fe.reset_index()
date_df.columns = ["lecture_date", "lecture_fixed_effect"]


pd.set_option('display.max_rows', 200)
pd.set_option('display.max_columns', 20)

print("\n===== STUDENT FIXED EFFECTS =====")
display(student_df)

print("\n===== LECTURE DATE FIXED EFFECTS =====")
display(date_df)

print("\n===== MODEL SUMMARY (TOP LINES) =====")
print(model.summary())

# Save student fixed effects
student_df.to_csv("student_fixed_effects.csv", index=False)

# Save date fixed effects
date_df.to_csv("date_fixed_effects.csv", index=False)

# Download the files
from google.colab import files
files.download("student_fixed_effects.csv")
files.download("date_fixed_effects.csv")

# Save the OLS regression summary to a text file
with open("ols_summary.txt", "w") as f:
    f.write(model.summary().as_text())

from google.colab import files
files.download("ols_summary.txt")