#!/bin/bash

echo "Начинаем выполнение..."

# Создание данных
echo "Создание данных..."
python3 scripts/data_creation.py
echo "Данные созданы."

# Предобработка данных
echo "Начинаем предобработку данных..."
python3 scripts/model_preprocessing.py
echo "Данные предобработаны."

# Подготовка и обучение модели
echo "Начинаем подготовку и обучение модели..."
python3 scripts/model_preparation.py
echo "Модель подготовлена и обучена."

# Тестирование модели
echo "Начинаем тестирование модели..."
python3 scripts/model_testing.py
echo "Модель протестирована."

echo "Завершение выполнения."
