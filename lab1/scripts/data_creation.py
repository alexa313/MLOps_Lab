import pandas as pd
import numpy as np
import os

def create_data(num_samples=1000):
    dates = pd.date_range(start='2020-01-01', periods=num_samples)
    temperatures = np.random.normal(loc=20, scale=5, size=num_samples)  # Генерируем данные о температуре

    # Включение аномалий
    for i in np.random.choice(range(num_samples), size=10, replace=False):
        temperatures[i] += np.random.choice([-20, 20])  # Аномальные значения

    data = pd.DataFrame({
        'Дата': dates,
        'Средняя температура': temperatures,
        'Эффективная температура': np.maximum(temperatures, 0)  # Целевая переменная
    })

    # Сохранение данных
    os.makedirs('data/train', exist_ok=True)
    os.makedirs('data/test', exist_ok=True)

    # Сохраняем 80% в train и 20% в test
    train_data = data.sample(frac=0.8, random_state=1)
    test_data = data.drop(train_data.index)

    train_data.to_csv('data/train/train_data.csv', index=False)
    test_data.to_csv('data/test/test_data.csv', index=False)

    print("Данные созданы и сохранены.")

if __name__ == "__main__":
    create_data()
