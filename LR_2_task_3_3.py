# Завантаження бібліотек
from pandas import read_csv
from pandas.plotting import scatter_matrix
from matplotlib import pyplot
from sklearn.model_selection import train_test_split
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.datasets import load_iris
iris_dataset = load_iris()

print("Ключі iris_dataset: \n{}".format(iris_dataset.keys()))
print(iris_dataset['DESCR'][:193] + "\n...")
print("Назви відповідей: {}".format(iris_dataset['target_names']))
print("Назва ознак: \n{}".format(iris_dataset['feature_names']))
print("Тип масиву data: {}".format(type(iris_dataset['data'])))

# Виведемо форму масиву data
print("Форма масиву data: {}".format(iris_dataset['data'].shape))

# Виведемо значення ознак для перших п'яти прикладів
print("Значення ознак для перших п'яти прикладів:\n{}".format(iris_dataset['data'][:5]))
print("Тип масиву target: {}".format(type(iris_dataset['target'])))
print("Відповіді:\n{}".format(iris_dataset['target']))
# Завантаження датасету
url = "https://raw.githubusercontent.com/jbrownlee/Datasets/master/iris.csv"
names = ['sepal-length', 'sepal-width', 'petal-length', 'petal-width', 'class']
dataset = read_csv(url, names=names)

# Перегляд основних параметрів
print(dataset.shape)
print(dataset.head(20))
print(dataset.describe())
print(dataset.groupby('class').size())

# Діаграми
dataset.plot(kind='box', subplots=True, layout=(2, 2), sharex=False, sharey=False)
pyplot.show()
dataset.hist()
pyplot.show()
scatter_matrix(dataset)
pyplot.show()

# Розділення датасету на навчальну та контрольну вибірки
array = dataset.values
X = array[:, 0:4]
y = array[:, 4]

# Розподіл X і y на вибірки
X_train, X_validation, y_train, y_validation = train_test_split(X, y, test_size=0.20, random_state=1)
# Використання алгоритму SVM
model = SVC(gamma='auto')
# Налаштування стратифікованої крос-валідації
stratified_kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
# Оцінка моделі з використанням метрики точності
cv_results = cross_val_score(model, X_train, y_train, cv=stratified_kfold, scoring='accuracy')
# Виведення результатів точності
print(f"Середня точність: {cv_results.mean() * 100:.2f}%")
print(f"Стандартне відхилення точності: {cv_results.std() * 100:.2f}%")

# Навчання моделі на всіх навчальних даних
model.fit(X_train, y_train)
# Передбачення на тестовій вибірці
y_pred = model.predict(X_validation)
# Обчислення точності на тестових даних
accuracy = accuracy_score(y_validation, y_pred)
print(f"Точність на тестових даних: {accuracy * 100:.2f}%")
# Завантажуємо моделі
models = []
models.append(('LR', LogisticRegression(solver='liblinear')))

# multi_class removed
models.append(('LDA', LinearDiscriminantAnalysis()))
models.append(('KNN', KNeighborsClassifier()))
models.append(('CART', DecisionTreeClassifier()))
models.append(('NB', GaussianNB()))
models.append(('SVM', SVC(gamma='auto')))
# Оцінка моделей
results = []
names = []
for name, model in models:
    kfold = StratifiedKFold(n_splits=10, random_state=1, shuffle=True)
    cv_results = cross_val_score(model, X_train, y_train, cv=kfold, scoring='accuracy')
    results.append(cv_results)
    names.append(name)
    print(f'{name}: {cv_results.mean():.6f} ({cv_results.std():.6f})')

# Порівняння алгоритмів
pyplot.boxplot(results, tick_labels=names) # Changed to tick_labels
pyplot.title('Algorithm Comparison')
pyplot.show()

# Створюємо прогноз на контрольній вибірці
model = SVC(gamma='auto')
model.fit(X_train, y_train)
predictions = model.predict(X_validation)
# Оцінюємо прогноз
print(accuracy_score(y_validation, predictions))
print(confusion_matrix(y_validation, predictions))
print(classification_report(y_validation, predictions))
knn = KNeighborsClassifier()
knn.fit(X_train, y_train)

# Створюємо прогноз на нових даних (X_new має бути визначено)
X_new = [[5.0, 2.9, 1.0, 0.2]] # приклад нових даних
prediction = knn.predict(X_new)
# Оскільки prediction є масивом, потрібно використовувати перший елемент
predicted_class = prediction[0]
# Прогноз
print("Прогноз: {}".format(prediction))
# сам прогноз (наприклад, ['setosa'])
print("Спрогнозована мітка: {}".format(predicted_class))
# відображення класу