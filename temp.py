import pandas as pd

def load_temperature_data(file_path):
    """
    Загружает данные о температуре из указанного файла.
    """
    data = pd.read_csv(file_path)

    # Вывод названий столбцов
    print("Загруженные столбцы:", data.columns.tolist())  # Преобразуем в список для более удобного вывода

    return data

if __name__ == "__main__":
    # Укажите путь к вашему файлу CSV с данными о температуре
    file_path = r'R:/URFU/MLOps/Лабораторная 1/pogoda-service_all.csv'  # Обновите этот путь

    # Загружаем данные
    temperature_data = load_temperature_data(file_path)