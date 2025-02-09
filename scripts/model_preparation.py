import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib

def prepare_model():
    # Загружаем предобработанные данные
    train_df = pd.read_csv('data/train/train_data_scaled.csv')

    # Предполагаем, что 'Эффективная температура' - это целевая переменная
    X = train_df[['Средняя температура']]  # Все колонки, кроме целевой
    y = train_df['Эффективная температура']  # Целевая переменная

    # Создаем и обучаем модель линейной регрессии
    model = LinearRegression()
    model.fit(X, y)

    # Сохраняем модель
    joblib.dump(model, 'model.joblib')  # Сохраняем модель
    print("Модель создана и обучена.")

if __name__ == '__main__':
    prepare_model()
