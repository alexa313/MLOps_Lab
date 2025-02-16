import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib
import os

def test_model():
    # Определение путей к файлам с использованием os.path
    base_dir = os.path.dirname(os.path.abspath(__file__))  # Получаем директорию скрипта
    data_path = os.path.join(base_dir, '../data/test/test_data.csv')  # Путь к тестовым данным
    model_path = os.path.join(base_dir, 'model.joblib')  # Если модель находится в той же директории


    # Загрузка тестовых данных
    test_data = pd.read_csv(data_path)

    # Определение признаков и целевой переменной
    X_test = test_data[['Средняя температура']]
    y_test = test_data['Эффективная температура']

    # Загрузка модели
    model = joblib.load(model_path)

    # Предсказание
    predictions = model.predict(X_test)

    # Оценка модели
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

if __name__ == "__main__":
    test_model()
