import os
import pandas as pd
from scripts.download_data import download_and_save_titanic_data
from scripts.fill_missing_values import fill_missing_ages
from scripts.one_hot_encode import one_hot_encode_sex

def test_download_data():
    download_and_save_titanic_data()
    assert os.path.exists('../datasets/titanic_data.csv'), "CSV файл не был создан."

def test_fill_missing_ages():
    fill_missing_ages()
    df = pd.read_csv('../datasets/titanic_data.csv')
    assert df['Age'].isnull().sum() == 0, "Есть пропущенные значения 'Age'."

def test_one_hot_encode_sex():
    one_hot_encode_sex()
    df = pd.read_csv('../datasets/titanic_data.csv')
    assert 'Gender_female' in df.columns and 'Gender_male' in df.columns, "One-hot-кодирование не было выполнено."

def run_all_tests():
    test_download_data()
    test_fill_missing_ages()
    test_one_hot_encode_sex()
    print("Все тесты пройдены успешно!")

if __name__ == "__main__":
    run_all_tests()
