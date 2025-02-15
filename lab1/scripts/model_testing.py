import pandas as pd
from sklearn.metrics import mean_squared_error
import joblib

def test_model():
    # Загрузка тестовых данных
    test_data = pd.read_csv('data/test/test_data.csv')

    # Предполагаем, что 'Эффективная температура' - это целевая переменная
    X_test = test_data[['Средняя температура']]
    y_test = test_data['Эффективная температура']

    # Загрузка модели
    model = joblib.load('model.joblib')  # Правильный путь к модели

    # Предсказание
    predictions = model.predict(X_test)

    # Оценка модели
    mse = mean_squared_error(y_test, predictions)
    print(f'Mean Squared Error: {mse}')

if __name__ == "__main__":
    test_model()
