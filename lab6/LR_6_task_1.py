from collections import Counter
# Вхідні дані
data = [
    {"Outlook": "Sunny", "Humidity": "High", "Wind": "Weak", "Play": "No"},
    {"Outlook": "Sunny", "Humidity": "High", "Wind": "Strong", "Play": "No"},
    {"Outlook": "Overcast", "Humidity": "High", "Wind": "Weak", "Play": "Yes"},
    {"Outlook": "Rain", "Humidity": "High", "Wind": "Weak", "Play": "Yes"},
    {"Outlook": "Rain", "Humidity": "Normal", "Wind": "Weak", "Play": "Yes"},
    {"Outlook": "Rain", "Humidity": "Normal", "Wind": "Strong", "Play": "No"},
    {"Outlook": "Overcast", "Humidity": "Normal", "Wind": "Strong", "Play": "Yes"},
    {"Outlook": "Sunny", "Humidity": "High", "Wind": "Weak", "Play": "No"},
    {"Outlook": "Sunny", "Humidity": "Normal", "Wind": "Weak", "Play": "Yes"},
    {"Outlook": "Rain", "Humidity": "High", "Wind": "Weak", "Play": "Yes"},
    {"Outlook": "Sunny", "Humidity": "Normal", "Wind": "Strong", "Play": "Yes"},
    {"Outlook": "Overcast", "Humidity": "High", "Wind": "Strong", "Play": "Yes"},
    {"Outlook": "Overcast", "Humidity": "Normal", "Wind": "Weak", "Play": "Yes"},
    {"Outlook": "Rain", "Humidity": "High", "Wind": "Strong", "Play": "No"},
]

#Функція для підрахунку умовних ймовірностей
def calculate_conditional_probability(attribute, value, target, target_value, data):
    relevant = [row for row in data if row[target] == target_value]
    count_matches = sum(1 for row in relevant if row[attribute] == value)
    return count_matches / len(relevant) if relevant else 0

# Функція для підрахунку загальної ймовірності
def calculate_total_probability(target_value, conditions, data):
    target_count = sum(1 for row in data if row["Play"] == target_value)
    total_count = len(data)
    target_probability = target_count / total_count

    conditional_probabilities = [
        calculate_conditional_probability(attr, val, "Play", target_value, data)
        for attr, val in conditions.items()
    ]
    combined_probability = target_probability
    for prob in conditional_probabilities:
        combined_probability *= prob

    return combined_probability

# Умови для розгляду за варіантом
conditions = {"Outlook": "Rain", "Humidity": "High", "Wind": "Strong"}

# Обчислення ймовірностей
prob_yes = calculate_total_probability("Yes", conditions, data)
prob_no = calculate_total_probability("No", conditions, data)

# Нормалізація
total = prob_yes + prob_no

if total != 0:
    normalized_yes = prob_yes / total
    normalized_no = prob_no / total
else:
    normalized_yes = 0
    normalized_no = 0


# Вивід результатів
print("Чи відбудеться матч?")
print(f"Умови: {conditions}")
print(f"Ймовірність, що матч відбудеться 'Yes' --> {normalized_yes:.2f}")
print(f"Ймовірність, що матч НЕ відбудеться 'No' --> {normalized_no:.2f}")