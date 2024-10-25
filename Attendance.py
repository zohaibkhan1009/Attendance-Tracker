#import the packages
import pandas as pd
import numpy as np
import seaborn as sns
import os
import pylab as pl
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, confusion_matrix

os.chdir("C:\\Users\\zohaib khan\\OneDrive\\Desktop\\USE ME\\dump\\zk")

data = pd.read_csv("attendance.csv")

data.head()

pd.set_option('display.max_columns', None)

data.head()

#Droping the Unnamed Column
data.drop('Unnamed: 12',axis=1,inplace=True)

#Checking the basic info of the dataset
data.info()

data.describe()


#convert the date 
data['date'] = pd.to_datetime(data['date'], format='%Y-%m-%d')

#Check for missing values
data.isnull().sum()*100

data.isnull().mean()*100


#Check for duplicate values
data[data.duplicated()]

data.drop_duplicates(inplace=True)  

data.dropna(inplace=True)

data.drop('year',axis=1,inplace=True)

data.drop('incident_type',axis=1,inplace=True)


#transforming the month and date columns for further analysis
data['month'] = data['date'].dt.month

data['year'] = data['date'].dt.year

grouped_date = data.groupby(['year','month'])

data.info()

# Group by student ID and calculate the required measures
attendance_summary = data.groupby('student_id').agg(
    total_school_days=('date', 'nunique'),
    total_days_attended=('attendance_status', lambda x: (x == 'Present').sum())
).reset_index()


# Merge the summary back into the original dataset
data = data.merge(attendance_summary, on='student_id', how='left')

data.head()

data.columns

#outlier detection Using Box-plot and removing outliers
def zohaib (data,age):
 Q1 = data[age].quantile(0.25)
 Q3 = data[age].quantile(0.75)
 IQR = Q3 - Q1
 data= data.loc[~((data[age] < (Q1 - 1.5 * IQR)) | (data[age] > (Q3 + 1.5 * IQR))),]
 return data

data.boxplot(column=["student_id"])

data = zohaib(data,"student_id")

data.boxplot(column=["age"])

data = zohaib(data,"age")

data.boxplot(column=["total_days_attended"])

data = zohaib(data,"total_days_attended")


data.boxplot(column=["total_school_days"])

data = zohaib(data,"total_school_days")


data.boxplot(column=["assignment"])

data = zohaib(data,"assignment")


data.boxplot(column=["test"])

data = zohaib(data,"test")



#Label Encoding
from sklearn.preprocessing import LabelEncoder

encoder = LabelEncoder()

encoder.fit(data["attendance_status"])

data["attendance_status"] = encoder.transform(data["attendance_status"])

data["attendance_status"].unique()

data.head()

data.info()


#Automating the EDA process using autoviz
from autoviz.AutoViz_Class import AutoViz_Class 
AV = AutoViz_Class()
import matplotlib.pyplot as plt
%matplotlib INLINE
filename = 'attendance.csv'
sep =","
dft = AV.AutoViz(
    filename  
)


data.drop('date',axis=1,inplace=True)


# changing the datatype
data = data.astype({col: 'int32' for col in data.select_dtypes(include='float64').columns})


#changing the datatype to make it uniform
data = data.astype({col: 'int32' for col in data.select_dtypes(include='int64').columns})

data.info()


#Segregating into X and Y
X = data.drop("test", axis = 1)

y = data["test"]

X.head()

y.head()

#Splitting the dataset into testing and training

from sklearn.model_selection import train_test_split

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 0)

#Scale the dataset
from sklearn.preprocessing import StandardScaler
sc = StandardScaler()
X_train = sc.fit_transform(X_train)
X_test = sc.transform(X_test)

#convertig the variable based on threshold
def categorize_test_score(score):
    if score >= 0.8:
        return 'high'
    elif score >= 0.6:
        return 'medium'
    else:
        return 'low'

data['test_category'] = data['test'].apply(categorize_test_score)

data.head()

#convert the test_category to numerics
label_encoder = LabelEncoder()

data['test_category'] = label_encoder.fit_transform(data['test_category'])



#calling the algorithm
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

#initialising the algorithm
log_model = LogisticRegression()
rf_model = RandomForestClassifier()
svm_model = SVC()
knn_model = KNeighborsClassifier()
gnb_model = GaussianNB()

# logistic regression model fitting
log_model.fit(X_train, y_train)

# random forest classifier model fitting
rf_model.fit(X_train, y_train)

# k nearest neighbour model fitting
knn_model.fit(X_train, y_train)

# gaussian naive bayes model fitting
gnb_model.fit(X_train, y_train)

# model testing libraries
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# logistic regression model testing
y_pred_log = log_model.predict(X_test)
print("Logistic Regression")
print(f"Accuracy: {accuracy_score(y_test, y_pred_log)}")
print(f"Precision: {precision_score(y_test, y_pred_log, average='weighted')}")
print(f"Recall: {recall_score(y_test, y_pred_log, average='weighted')}")
print(f"F1 score: {f1_score(y_test, y_pred_log, average='weighted')}")


# Random Forest model testing

y_pred_rf = rf_model.predict(X_test)
print("Random Forest Classifier")
print(f"Accuracy: {accuracy_score(y_test, y_pred_rf)}")
print(f"Precision: {precision_score(y_test, y_pred_rf, average='weighted')}")
print(f"Recall: {recall_score(y_test, y_pred_rf, average='weighted')}")
print(f"F1 score: {f1_score(y_test, y_pred_rf, average='weighted')}")

# k nearest neighbour model testing
y_pred_knn = knn_model.predict(X_test)
print("K Nearest Neighbours")
print(f"Accuracy: {accuracy_score(y_test, y_pred_knn)}")
print(f"Precision: {precision_score(y_test, y_pred_knn, average='weighted')}")
print(f"Recall: {recall_score(y_test, y_pred_knn, average='weighted')}")
print(f"F1 score: {f1_score(y_test, y_pred_knn, average='weighted')}")

# gaussian naive bayes model testing
y_pred_gnb = gnb_model.predict(X_test)
print("Gaussian Naive Bayes")
print(f"Accuracy: {accuracy_score(y_test, y_pred_gnb)}")
print(f"Precision: {precision_score(y_test, y_pred_gnb, average='weighted')}")
print(f"Recall: {recall_score(y_test, y_pred_gnb, average='weighted')}")
print(f"F1 score: {f1_score(y_test, y_pred_gnb, average='weighted')}")



#Once more checking the accuracy
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier


# Define the models
models = {
    "Logistic Regression": LogisticRegression(),
    "Decision Tree": DecisionTreeClassifier(),
    "Support Vector Classifier": SVC(),
    "Gradient Boosting Classifier": GradientBoostingClassifier(),
    "K-Nearest Neighbors": KNeighborsClassifier(),
    "MLP Classifier": MLPClassifier()
}


# Train and evaluate each model
for name, model in models.items():
    model.fit(X_train, y_train)
    score = model.score(X_test, y_test)
    print(f"{name} Accuracy: {score * 100:.2f}%")
    
#To check whether there is overfitting or not doing K-Fold Cross Validation on random model Decision Tree
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import classification_report, confusion_matrix


decision_tree_model = DecisionTreeClassifier()

cv_scores = cross_val_score(decision_tree_model, X, y, cv=10, scoring='accuracy')

print(f'Cross-Validation Scores: {cv_scores}')
print(f'Mean Accuracy: {np.mean(cv_scores)}')
print(f'Standard Deviation: {np.std(cv_scores)}')

#There is no overfitting or underfitting
