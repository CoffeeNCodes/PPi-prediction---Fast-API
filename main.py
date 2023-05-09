from fastapi import FastAPI
import uvicorn
from pydantic import BaseModel
from sklearn.linear_model import LogisticRegression
import pandas as pd
import os
import numpy as np
from sklearn.metrics import f1_score, roc_auc_score
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from imblearn.over_sampling import SMOTE

from sklearn.preprocessing import StandardScaler, MinMaxScaler

working_directory = os.getcwd()
print(working_directory)
path = 'Pacemaker_tavr_v2.csv'
df = pd.read_csv(path)

X = df.drop("PacemakerImplantation", axis= 1)
y = df['PacemakerImplantation']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state= 42)

smote = SMOTE(random_state=42)

X_train_res, y_train_res = smote.fit_resample(X_train, y_train)
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
model = LogisticRegression(random_state=42,solver='saga', max_iter=10000, C=1.0)
model.fit(X_train, y_train)
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

prediction = model.predict(X_test)

auc_score = roc_auc_score(y_test, y_pred)
fpr, tpr, thresholds = roc_curve(y_test, y_pred)
plt.plot(fpr, tpr, label=f'AUC={auc_score:.2f}')
plt.plot([0, 1], [0, 1], linestyle='--', color='gray')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic (ROC) Curve')
plt.legend()
plt.show()

gender_counts = df['Sex'].value_counts()
sns.countplot(x='Sex', data=df)
plt.xlabel('Sex')
plt.ylabel('Count')
plt.title('Number of Males and Females')
plt.show()

df['Sex'].value_counts()
sns.countplot(x='Sex',hue ='PacemakerImplantation',dodge= False, data=df)
plt.xlabel('Sex')
plt.ylabel('pacemaker implantation')
plt.show()



class TAVR(BaseModel):
    Age: float
    Sex: int
    BSA: float
    BMI: float
    HTN: int
    CAD: int
    DM: int
    COPD: int
    AF: int
    PVD: int
    CVA: int
    Hemodialysis: float
    PreviousHeartSurgery_Intervention: int
    SymptomaticAS : float
    ACEi_ARB: int
    Beta_Blocker: int
    Aldosteroneantagonist: int
    CCB: int
    AntiPlateletotherthanASA: int
    ASA: int
    AntiplateletTherapy: int
    Diuretics: int
    LVEF: float
    SystolicBP: float
    DiastolicBP: float 
    LVOT: float
    ValveCode: int
    ValveSize: int
    BaselineRhythm: int
    PR: float
    QRS: int
    QRSmorethan120: int
    FirstdegreeAVblock: float
    Baseline_conduction_disorder: int
    BaselineRBBB: int
    DeltaPR: float
    DeltaQRS: int
    New_Onset_LBBB: int
    PPMdays: int

app = FastAPI()
@app.get("/")
def main():
    return 'Pacemaker risk report'

@app.post('/predict')

def predict(request: TAVR):
    input_data = request.dict()
    prediction = model.predict([list(input_data.values())])
    return {
        "prediction": int(prediction)
    }






if __name__=='__main__':
    uvicorn.run("main:app", host="127.0.0.1", port=8000,reload = True)
    