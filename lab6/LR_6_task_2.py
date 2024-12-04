import pandas as pd
from collections import defaultdict

# Завантаження даних і обробка
dataset_url = "https://raw.githubusercontent.com/susanli2016/Machine-Learning-with-Python/master/data/renfe_small.csv"
data = pd.read_csv(dataset_url)

# Зберігається відповідні стопці та опускайте пропущені значення
data = data[["price", "train_type", "origin", "destination", "train_class"]].dropna()

# Розділ цін на 'low', 'medium', 'high'
price_bins = [data["price"].min(), data["price"].quantile(0.33), data["price"].quantile(0.66), data["price"].max()]
data["price_category"] = pd.cut(data["price"], bins=price_bins, labels=["low", "medium", "high"])

# Обчислення частотних розподілів
frequency_distributions = defaultdict(lambda: defaultdict(int))
total_per_category = defaultdict(int)

for _, row in data.iterrows():
    category = row["price_category"]
    total_per_category[category] += 1
    for feature in ["train_type", "origin", "destination", "train_class"]:
        frequency_distributions[feature][(row[feature], category)] += 1

# Обчислення умовних ймовірностей
def conditional_probability(feature, value, category):
    match_count = frequency_distributions[feature].get((value, category), 0)
    category_total = total_per_category[category]
    return match_count / category_total if category_total else 0

# Визначення калькулятора ймовірностей за теоремою Байєса
def calculate_category_probability(category, conditions):
    prior = total_per_category[category] / len(data)
    likelihood = 1
    for feature, value in conditions.items():
        likelihood *= conditional_probability(feature, value, category)
    return prior * likelihood

# Визначте характеристики квитка для прогнозування
ticket_conditions = {
    "train_type": "AVE",
    "origin": "MADRID",
    "destination": "SEVILLA",
    "train_class": "Turista"
}

# Розрахувати апостеріорні ймовірності для кожної цінової категорії
posterior_probabilities = {}
for category in total_per_category.keys():
    posterior_probabilities[category] = calculate_category_probability(category, ticket_conditions)

# Нормалізація ймовірностей
total_probability = sum(posterior_probabilities.values())
normalized_probabilities = {cat: prob / total_probability for cat, prob in posterior_probabilities.items()}

print("Ймовірності для кожної категорії вартості квитка -->")
for category, probability in normalized_probabilities.items():
    print(f"{str(category).capitalize()}: {probability:.2f}")


