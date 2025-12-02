from google.colab import files
uploaded = files.upload()

###################################
import pandas as pd
df = pd.read_csv('mushrooms.csv')
df.head()
###################################

df.isnull().sum()

###################################
le = LabelEncoder()
for col in df.columns:
    df[col] = le.fit_transform(df[col])
df.head()
###################################
#222222222222
###################################
import seaborn as sns
import matplotlib.pyplot as plt
sns.countplot(df['class'])
plt.title("Yesa bo'ladigon (0) va Zaharli (1) qo'ziqorinlar")
plt.show()
df['class'].value_counts()
###################################

columns_to_plot = ['cap-shape', 'cap-surface', 'cap-color', 'odor']
for col in columns_to_plot:
    plt.figure(figsize=(5,4))
    sns.countplot(df[col])
    plt.title(f"{col} tarqalishi")
    plt.show()
###################################

for col in columns_to_plot:
    plt.figure(figsize=(5,4))
    sns.countplot(x=df[col], hue=df['class'])
    plt.title(f"{col} → class ta’siri")
    plt.show()

###################################
plt.figure(figsize=(12,10))
sns.heatmap(df.corr(), cmap="coolwarm")
plt.title("Xususiyatlarning o'zaro bog'liqligi issiqlik xaritasi")
plt.show()
###################################
# 33333333
###################################
from sklearn.model_selection import train_test_split
X = df.drop('class', axis=1)
y = df['class']
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)
print("Train shape:", X_train.shape)
print("Test shape:", X_test.shape)
###################################

from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns
import matplotlib.pyplot as plt
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)
preds_knn = knn.predict(X_test)
cm = confusion_matrix(y_test, preds_knn)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("KNN - klassifikatsiyasi")
plt.show()
print(classification_report(y_test, preds_knn))

###################################
from sklearn.tree import DecisionTreeClassifier
dt = DecisionTreeClassifier()
dt.fit(X_train, y_train)
preds_dt = dt.predict(X_test)
cm = confusion_matrix(y_test, preds_dt)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Decision Tree - klassifikatsiyasi")
plt.show()
print(classification_report(y_test, preds_dt))
###################################

from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier()
rf.fit(X_train, y_train)
preds_rf = rf.predict(X_test)
cm = confusion_matrix(y_test, preds_rf)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Random Forest - klassifikatsiyasi")
plt.show()
print(classification_report(y_test, preds_rf))
###################################

from sklearn.linear_model import LogisticRegression
lr = LogisticRegression(max_iter=200)
lr.fit(X_train, y_train)
preds_lr = lr.predict(X_test)
cm = confusion_matrix(y_test, preds_lr)
plt.figure(figsize=(4,3))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
plt.title("Logistic Regression - klassifikatsiyasi")
plt.show()
print(classification_report(y_test, preds_lr))
###################################

