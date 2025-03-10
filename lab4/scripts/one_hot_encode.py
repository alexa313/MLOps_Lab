import pandas as pd

def one_hot_encode_sex():
    # Загрузка данных
    df = pd.read_csv('../datasets/titanic_data.csv')
    # Применение one-hot-кодирования к столбцу 'Sex'
    sex_encoded = pd.get_dummies(df['Sex'], prefix='Gender')
    # Объединение с исходным датасетом
    df = pd.concat([df, sex_encoded], axis=1)
    # Сохранение обновленного датасета
    df.to_csv('../datasets/titanic_data.csv', index=False)

if __name__ == "__main__":
    one_hot_encode_sex()
