import numpy as np
import matplotlib.pyplot as plt
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from utilities import visualize_classifier

# Вхідний файл, який містить дані
input_file = 'data_multivar_nb.txt'

# Завантаження даних із вхідного файлу
data = np.loadtxt(input_file, delimiter=',')
X, y = data[:, :-1], data[:, -1]

# Розбивка даних на навчальний та тестовий набори
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=3)

# Наївний Байєсівський Класифікатор
classifier_nb = GaussianNB()
classifier_nb.fit(X_train, y_train)
y_test_pred_nb = classifier_nb.predict(X_test)

# Прогнозування та обчислення якості для NB
accuracy_nb = 100.0 * (y_test == y_test_pred_nb).sum() / X_test.shape[0]
print("Accuracy of the Naive Bayes classifier =", round(accuracy_nb, 2), "%")

# Крос-валідація для NB
num_folds = 3
accuracy_values_nb = cross_val_score(classifier_nb, X, y, scoring='accuracy', cv=num_folds)
print("Naive Bayes Accuracy: " + str(round(100 * accuracy_values_nb.mean(), 2)) + "%")
precision_values_nb = cross_val_score(classifier_nb, X, y,scoring='precision_weighted', cv=num_folds)
print("Naive Bayes Precision: " + str(round(100 * precision_values_nb.mean(), 2)) + "%")
recall_values_nb = cross_val_score(classifier_nb, X, y, scoring='recall_weighted', cv=num_folds)
print("Naive Bayes Recall: " + str(round(100 * recall_values_nb.mean(), 2)) + "%")
f1_values_nb = cross_val_score(classifier_nb, X, y, scoring='f1_weighted', cv=num_folds)
print("Naive Bayes F1: " + str(round(100 * f1_values_nb.mean(), 2)) + "%")

# Візуалізація результатів NB
visualize_classifier(classifier_nb, X_test, y_test)

# Support Vector Machine (SVM) Класифікатор
classifier_svm = SVC(kernel='linear', random_state=3)
classifier_svm.fit(X_train, y_train)
y_test_pred_svm = classifier_svm.predict(X_test)

# Прогнозування та обчислення якості для SVM
accuracy_svm = 100.0 * (y_test == y_test_pred_svm).sum() / X_test.shape[0]
print("\nAccuracy of the SVM classifier =", round(accuracy_svm, 2), "%")

# Крос-валідація для SVM
accuracy_values_svm = cross_val_score(classifier_svm, X, y, scoring='accuracy', cv=num_folds)
print("SVM Accuracy: " + str(round(100 * accuracy_values_svm.mean(), 2)) + "%")
precision_values_svm = cross_val_score(classifier_svm, X, y, scoring='precision_weighted', cv=num_folds)
print("SVM Precision: " + str(round(100 * precision_values_svm.mean(), 2)) + "%")
recall_values_svm = cross_val_score(classifier_svm, X, y, scoring='recall_weighted', cv=num_folds)
print("SVM Recall: " + str(round(100 * recall_values_svm.mean(), 2)) + "%")
f1_values_svm = cross_val_score(classifier_svm, X, y, scoring='f1_weighted', cv=num_folds)
print("SVM F1: " + str(round(100 * f1_values_svm.mean(), 2)) + "%")

# Візуалізація результатів SVM
visualize_classifier(classifier_svm, X_test, y_test)

# Порівняння результатів
print("\nComparison between Naive Bayes and SVM:")
print(f"Naive Bayes Accuracy: {round(accuracy_nb, 2)}%, SVM Accuracy: {round(accuracy_svm, 2)}%")
print(f"Naive Bayes F1: {round(100 * f1_values_nb.mean(), 2)}%, SVM F1: {round(100 * f1_values_svm.mean(), 2)}%")