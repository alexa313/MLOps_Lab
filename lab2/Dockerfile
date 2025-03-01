# Используйте официальный образ Jenkins с поддержкой LTS и JDK 17
FROM jenkins/jenkins:lts-jdk21

# Переключаемся на пользователя root для установки пакетов
USER root

# Обновляем список пакетов
RUN apt-get update

# Устанавливаем wget и software-properties-common, необходимые для управления пакетами
RUN apt-get install -y wget software-properties-common

# Устанавливаем Python 3
RUN apt-get install -y python3

# Устанавливаем пакет apt-utils для более хорошей обработки установок
RUN apt-get install -y apt-utils

# Устанавливаем инструмент для создания виртуальных окружений в Python
RUN apt-get install -y python3-venv

# Устанавливаем OpenJDK 17 (необходимо, если вы хотите использовать Java)
RUN apt-get install -y openjdk-21-jdk

# Возвращаемся к пользователю jenkins после окончания установки
USER jenkins

