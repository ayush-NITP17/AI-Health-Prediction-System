import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression,LogisticRegressionCV,LassoCV,Lasso
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier,BaggingClassifier,AdaBoostClassifier,VotingClassifier,StackingClassifier
from sklearn.model_selection import train_test_split,GridSearchCV
from sklearn.metrics import precision_score,recall_score,accuracy_score,f1_score,confusion_matrix,classification_report
df=pd.read_csv("novagen_dataset.csv")
#now after doing feature engineering
#creating new feature
df["internal_body_health"]=df["Blood_Pressure"]+df["Heart_Rate"]+df["Cholesterol"]+df["Glucose_Level"]+df["Stress_Level"]+df["BMI"]
df["daily_activity"]=df["Sleep_Hours"]+df["Exercise_Hours"]+df["Water_Intake"]
df["Medical_status"]=df["MentalHealth"]+df["PhysicalActivity"]+df["MedicalHistory"]+df["Allergies"]
df = pd.get_dummies(
    df,
    columns=[
        "Diet_Type__Vegan",
        "Diet_Type__Vegetarian",
        "Blood_Group_AB",
        "Blood_Group_B",
        "Blood_Group_O"
    ],
    drop_first=True,
    dtype=int
)
x=df.drop("Target",axis=1)
y=df["Target"]
x_train,x_test,y_train,y_test=train_test_split(
    x,y,test_size=0.2,stratify=y,random_state=42
)
#scaling x train and test dataset
se=StandardScaler()
x_train_scaled=se.fit_transform(x_train)
x_test_scaled=se.transform(x_test)
full_tree=DecisionTreeClassifier(random_state=42)
full_tree.fit(x_train,y_train)

#prunning cost complexity

path=full_tree.cost_complexity_pruning_path(x_train,y_train)
ccp_alpha=path.ccp_alphas
#train our model for all alphas
tree=[]
for alpha in ccp_alpha:
    model=DecisionTreeClassifier(random_state=42,ccp_alpha=alpha)
    model.fit(x_train,y_train)
    acc=model.score(x_test,y_test)

    tree.append((model,alpha))

best_acc=0
best_alpha=0

for model,alpha in tree:
    curr_acc=model.score(x_test,y_test)
    if curr_acc>best_acc:
        best_acc=curr_acc
        best_alpha=alpha
#model 12 with adaboost after feature engineering
base_model=DecisionTreeClassifier(ccp_alpha=best_alpha)
model=AdaBoostClassifier(estimator=base_model,n_estimators=100,random_state=42)
model.fit(x_train,y_train)
y_pred=model.predict(x_test)
cm=confusion_matrix(y_test,y_pred)
print(cm)
cr=classification_report(y_test,y_pred)
print(cr)
print("accuracy of = ",accuracy_score(y_test,y_pred)*100)
print("precision of = ",precision_score(y_test,y_pred)*100)
print("recall of = ",recall_score(y_test,y_pred)*100)
print("f1 score of = ",f1_score(y_test,y_pred)*100)
import joblib

joblib.dump(model,"health_model.pkl")
joblib.dump(x.columns.tolist(),"feature_columns.pkl")
print(x.columns.tolist())