import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, r2_score
import pickle, os

np.random.seed(42)
n = 2000

roles       = ["Data Analyst","ML Engineer","Software Engineer","Data Scientist","DevOps Engineer","Backend Developer","Frontend Developer","Full Stack Developer"]
industries  = ["IT","Finance","Healthcare","E-Commerce","Consulting","Startup","Government"]
locations   = ["Bengaluru","Hyderabad","Mumbai","Delhi","Chennai","Pune","Remote"]
educations  = ["B.Sc","B.Tech","M.Sc","MCA","MBA","PhD"]
skills_list = ["Python","SQL","AWS","ML","React","Java","Docker","Tableau"]

role_base = {
    "Data Analyst": 5.5, "ML Engineer": 11.0, "Software Engineer": 8.0,
    "Data Scientist": 12.0, "DevOps Engineer": 10.0, "Backend Developer": 9.0,
    "Frontend Developer": 7.5, "Full Stack Developer": 10.5
}
edu_boost = {"B.Sc": 0.0, "B.Tech": 0.5, "M.Sc": 0.8, "MCA": 0.6, "MBA": 1.2, "PhD": 2.0}
loc_boost = {"Bengaluru": 1.5, "Hyderabad": 1.2, "Mumbai": 1.3, "Delhi": 1.1, "Chennai": 0.9, "Pune": 0.8, "Remote": 0.5}
ind_boost = {"IT": 0.8, "Finance": 1.5, "Healthcare": 0.5, "E-Commerce": 1.0, "Consulting": 1.2, "Startup": 0.3, "Government": -0.5}

rows = []
for _ in range(n):
    role      = np.random.choice(roles)
    industry  = np.random.choice(industries)
    location  = np.random.choice(locations)
    education = np.random.choice(educations)
    exp       = np.random.randint(0, 20)
    skills    = np.random.randint(1, 6)

    salary = (role_base[role]
              + edu_boost[education]
              + loc_boost[location]
              + ind_boost[industry]
              + exp * 0.45
              + skills * 0.3
              + np.random.normal(0, 1.2))
    salary = max(2.5, round(salary, 2))
    rows.append([role, industry, location, education, exp, skills, salary])

df = pd.DataFrame(rows, columns=["role","industry","location","education","experience","num_skills","salary_lpa"])

encoders = {}
for col in ["role","industry","location","education"]:
    le = LabelEncoder()
    df[col+"_enc"] = le.fit_transform(df[col])
    encoders[col] = le

features = ["role_enc","industry_enc","location_enc","education_enc","experience","num_skills"]
X = df[features]
y = df["salary_lpa"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = GradientBoostingRegressor(n_estimators=200, max_depth=4, learning_rate=0.08, random_state=42)
model.fit(X_train, y_train)

preds = model.predict(X_test)
print(f"MAE : {mean_absolute_error(y_test, preds):.2f} LPA")
print(f"R²  : {r2_score(y_test, preds):.3f}")

os.makedirs("models", exist_ok=True)
with open("models/salary_model.pkl", "wb") as f:
    pickle.dump({"model": model, "encoders": encoders,
                 "roles": roles, "industries": industries,
                 "locations": locations, "educations": educations}, f)
print("Model saved → models/salary_model.pkl")
