import pandas as pd
import joblib

df = pd.read_csv('pose_data_augmented.csv')
print(df.head())
print(df.count())

x = df.drop("label", axis=1)
y = df["label"]

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y_encoded = le.fit_transform(y)

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded)

from sklearn.ensemble import RandomForestClassifier
clf = RandomForestClassifier(oob_score=True, n_estimators=100, max_depth=5, random_state=42)

from sklearn.model_selection import cross_val_score
scores = cross_val_score(clf, x_train, y_train, cv=5)
print("Cross-validation accuracy train set:", scores.mean()*100)

clf.fit(x_train, y_train)

joblib.dump(clf,"pose_classifier.pkl")
joblib.dump(le, "label_encoder.pkl")

print("OOB score:", clf.oob_score_*100)

from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
y_pred = clf.predict(x_test)
acc = accuracy_score(y_test, y_pred)
prec = precision_score(y_test, y_pred, average='macro')
rec = recall_score(y_test, y_pred, average='macro')
print("số x_train = {}".format(len(x_train)))
print("số x_test = {}".format(len(x_test)))
print("Độ chính xác Accuracy là {:.2f}%".format(acc*100))
print("Precision score là {:.2f}%".format(prec*100))
print("Recall = {:.2f}%".format((rec*100)))
print("Confusion matrix:")
print(confusion_matrix(y_test, y_pred))


