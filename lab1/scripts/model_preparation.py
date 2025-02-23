import pandas as pd
from sklearn.linear_model import LinearRegression
import joblib
import os

def prepare_model():
    # Проверяем текущую директорию
    print("Текущая директория:", os.getcwd())
    
    # Указываем абсолютный путь к файлу CSV
    file_path = 'C:/py_projects/urfu/MLOps_Lab/lab1/data/train/train_data_scaled.csv'
    print("Проверка наличия файла:", os.path.isfile(file_path))
    
    # Загружаем предобработанные данные
    train_df = pd.read_csv(file_path)

    # Проверяем названия колонок
    print("Название колонок:", train_df.columns)

    # Определяем предикторы и целевую переменную
    X = train_df[['Средняя температура']]
    y = train_df['Эффективная температура']

    # Создаем и обучаем модель линейной регрессии
    model = LinearRegression()
    model.fit(X, y)

    # Сохраняем модель в указанном вами пути
    model_save_path = 'C:/py_projects/urfu/MLOps_Lab/lab1/scripts/model.joblib'
    joblib.dump(model, model_save_path)
    print("Модель создана и обучена, сохранена по пути", model_save_path)

if __name__ == '__main__':
        prepare_model()
