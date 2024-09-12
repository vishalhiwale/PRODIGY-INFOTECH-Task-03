import pandas as pd
from sklearn.tree import DecisionTreeClassifier, plot_tree
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import matplotlib.pyplot as plt

url = "C:/Users/ASUS/Documents/GitHub/PRODIGY-INFOTECH-Task-03/bank.csv" # bank.csv file's path from your device
data = pd.read_csv(url, sep=';')
data = pd.get_dummies(data, drop_first=True)
X = data.drop('y_yes', axis=1)
y = data['y_yes']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

clf = DecisionTreeClassifier(random_state=42)
clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print('Classification Report:')
print(classification_report(y_test, y_pred))
print('Confusion Matrix:')
print(confusion_matrix(y_test, y_pred))

param_grid = {
    'max_depth': [3, 5, 10, None],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [2, 2, 4]
}

grid_search = GridSearchCV(DecisionTreeClassifier(random_state=42), param_grid, cv=5, n_jobs=-1, verbose=1)
grid_search.fit(X_train, y_train)

print(f'Best Parameters: {grid_search.best_params_}')

y_pred_optimized = grid_search.best_estimator_.predict(X_test)
print(f'Optimized Accuracy: {accuracy_score(y_test, y_pred_optimized)}')
print('Optimized Classification Report:')
print(classification_report(y_test, y_pred_optimized))

plt.figure(figsize=(20,10))
plot_tree(clf, feature_names=X.columns, class_names=['No', 'Yes'], filled=True)
plt.show()
