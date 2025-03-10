import pandas as pd

def fill_missing_ages():
    # Загрузка датасета
    df = pd.read_csv('../datasets/titanic_data.csv')
    # Заполнение пропущенных значений 'Age'
    average_age = df['Age'].mean()
    df['Age'].fillna(average_age, inplace=True)
    # Сохранение обновленного датасета
    df.to_csv('../datasets/titanic_data.csv', index=False)

if __name__ == "__main__":
    fill_missing_ages()
