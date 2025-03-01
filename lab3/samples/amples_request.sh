# Пример POST-запроса с использованием curl
curl --request POST \
  --url 'http://127.0.0.1:8000/predict/' \
  --header 'Content-Type: application/json' \
  --data '{
  "sepal_length": 5.1,
  "sepal_width": 3.5,
  "petal_length": 1.4,
  "petal_width": 0.2
}'
