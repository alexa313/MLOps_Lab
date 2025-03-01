import pickle
import pandas as pd
from fastapi import FastAPI
from pydantic import BaseModel
from sklearn.datasets import load_iris
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Определяем модель входящих данных
class IrisFeatures(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float

# Создаем экземпляр приложения FastAPI
app = FastAPI()

# Загружаем данные о ирисах
iris_data = load_iris()
X_train_set, X_test_set, y_train_set, y_test_set = train_test_split(
    iris_data.data, iris_data.target, test_size=0.2, random_state=42
)

# Сохраняем данные в CSV-файл
iris_dataframe = pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris_dataframe['target'] = iris_data.target
iris_dataframe.to_csv('data/datasets/iris_dataset.csv', index=False)

# Стандартизируем данные
scaler_model = StandardScaler()
X_train_scaled_set = scaler_model.fit_transform(X_train_set)
X_test_scaled_set = scaler_model.transform(X_test_set)

# Создаем и обучаем классификатор
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train_scaled_set, y_train_set)

# Сохраняем модель в файл с помощью pickle
with open('data/model/iris_model.pkl', 'wb') as model_file:
    pickle.dump((rf_model, scaler_model), model_file)

# Загружаем сохранённую модель
with open('data/model/iris_model.pkl', 'rb') as model_file:
    rf_model, scaler_model = pickle.load(model_file)

# Соответствие индексов классам
class_labels = {0: 'setosa', 1: 'versicolor', 2: 'virginica'}

# Создаем эндпоинт для предсказания
@app.post("/predict/")
async def classify_iris(features: IrisFeatures):
    input_features = [[features.sepal_length, features.sepal_width, features.petal_length, features.petal_width]]
    scaled_input = scaler_model.transform(input_features)
    predicted_index = rf_model.predict(scaled_input)[0]
    predicted_class = class_labels[predicted_index]
    return {"prediction": predicted_class}

# Создаем эндпоинт для отправки информационного сообщения
@app.get("/")
async def welcome_message():
    return {
        "message": "To classify an iris flower, send a POST request to the /predict endpoint.",
        "example_body": {
            "sepal_length": 1.6,
            "sepal_width": 4.4,
            "petal_length": 1.4,
            "petal_width": 3.6
        }
    }
