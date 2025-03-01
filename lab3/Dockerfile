# Устанавливаем базовый образ Python
FROM python:3.9-slim

# Задаем рабочую директорию
WORKDIR /app

# Копируем исходные коды приложения в контейнер
COPY . /app/

# Устанавливаем необходимые зависимости из файла
RUN pip install --no-cache-dir -r requirements.txt

# Определяем команду для запуска приложения
CMD ["uvicorn", "src.app:app", "--host", "0.0.0.0", "--port", "8000"]
