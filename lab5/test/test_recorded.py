import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error

# Функция для генерации случайных данных
def generate_data(n_samples, noise_factor=0.0):
    X = np.random.rand(n_samples, 1) * 10
    y = 2 * X.squeeze() + 3 + np.random.randn(n_samples) * noise_factor
    return X, y

# Функция для оценки модели с помощью MSE
def evaluate_model(X_test, y_test, model):
    y_pred = model.predict(X_test)
    return mean_squared_error(y_test, y_pred)

# Главная функция для тестирования модели
def test_model_performance():
    model = LinearRegression()

    # Создаем три датасета с качественными данными и один зашумленный
    quality_datasets = [generate_data(100, noise_factor=0.5) for _ in range(3)]
    noisy_dataset = generate_data(100, noise_factor=5)

    # Обучаем модель на первом качественном датасете
    model.fit(*quality_datasets[0])

    # Вычисляем максимальное значение MSE на качественных данных
    max_mse_quality = max(mean_squared_error(y, model.predict(X)) for X, y in quality_datasets)

    # Функция для проверки MSE каждого датасета
    def check_mse(X_test, y_test, dataset_name):
        mse = evaluate_model(X_test, y_test, model)
        if mse <= max_mse_quality:
            print(f"Датасет {dataset_name}: MSE = {mse} (в пределах нормы)")
        else:
            print(f"Датасет {dataset_name}: MSE = {mse} (вышло за пределы нормы)")

    # Проверяем каждый из качественных датасетов
    for idx, (X, y) in enumerate(quality_datasets, start=1):
        check_mse(X, y, f"Качественный датасет {idx}")

    # Проверка на зашумленном датасете
    check_mse(*noisy_dataset, "Зашумленный датасет")

# Запуск тестирования модели
test_model_performance()

