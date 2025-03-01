import requests

# Определяем адрес API для отправки POST-запроса
endpoint = "http://127.0.0.1:8000/predict/"

# Подготавливаем данные для запроса
payload = {
    "sepal_length": 1.1,
    "sepal_width": 1.5,
    "petal_length": 5.4,
    "petal_width": 1.2
}

# Выполняем POST-запрос
result = requests.post(endpoint, json=payload)

# Печатаем ответ от сервера
print(result.json())
