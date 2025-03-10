import pandas as pd

from catboost.datasets import titanic
import pandas as pd

def download_and_save_titanic_data():
    # Загрузка данных
    data, _ = titanic()
    # Сохранение в CSV
    data.to_csv('../datasets/titanic_data.csv', index=False)

if __name__ == "__main__":
    download_and_save_titanic_data()
