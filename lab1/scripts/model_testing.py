import joblib
import pandas as pd
import os
from sklearn.metrics import mean_squared_error, r2_score

def test_model():
    model_load_path = 'C:/py_projects/urfu/MLOps_Lab/lab1/scripts/model.joblib'
    
    # Проверка наличия модели
    if not os.path.isfile(model_load_path):
        print("Ошибка: Модель не найдена по указанному пути.")
        return

    print("Проверка наличия модели: True")
    
    # Загружаем модель
    model = joblib.load(model_load_path)
    
    # Указываем путь к тестовым данным
    test_file_path = 'C:/py_projects/urfu/MLOps_Lab/lab1/data/test/test_data_scaled.csv'
    
    # Проверка наличия тестовых данных
    if not os.path.isfile(test_file_path):
        print("Ошибка: Тестовые данные не найдены по указанному пути.")
        return

    print("Проверка наличия тестовых данных: True")
    
    # Загружаем тестовые данные
    test_df = pd.read_csv(test_file_path)

    # Проверяем, есть ли необходимые колонки в тестовых данных
    required_columns = ['Средняя температура', 'Эффективная температура']
    for column in required_columns:
        if column not in test_df.columns:
            print(f"Ошибка: Отсутствует колонка '{column}' в тестовых данных.")
            return

    # Извлекаем предикторы и целевую переменную
    X_test = test_df[['Средняя температура']]
    y_test = test_df['Эффективная температура']

    # Предсказания на тестовых данных
    predictions = model.predict(X_test)

    # Оценка модели
    mse = mean_squared_error(y_test, predictions)
    r2 = r2_score(y_test, predictions)

    print(f"Среднеквадратичная ошибка: {mse}")
    print(f"Коэффициент детерминации R^2: {r2}")

if __name__ == '__main__':
    test_model()
