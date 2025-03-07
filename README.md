# Проект по выявлению аномалий в данных о просмотрах видео

## Цель проекта
Целью проекта является анализ данных о просмотрах видео с использованием алгоритмов машинного обучения для выявления аномальных паттернов, которые могут указывать на подозрительную активность, такую как накрутка просмотров или использование ботов.

## Задачи
- Проведение детального анализа данных для выявления необычных паттернов поведения.
- Использование методов машинного обучения для автоматического обнаружения аномалий.
- Разработка рекомендаций для предотвращения подобных активностей в будущем.

## Используемые технологии
- **Язык программирования:** Python
- **Библиотеки:**
  - Pandas (для работы с данными)
  - Scikit-learn (для машинного обучения)
  - Matplotlib / Seaborn (для визуализации данных)
- **Методы анализа:**
  - Описательная статистика
  - Кластеризация (например, K-means)
  - Алгоритмы обнаружения аномалий (например, Isolation Forest)

## Этапы проекта

### 1. Подготовка данных
- Загрузка данных из CSV (views_statistics.csv).
- Очистка данных от пропусков и дубликатов.
- Преобразование данных в подходящий формат для анализа.

### 2. Исследовательский анализ данных (EDA)
- Анализ активности по времени.
- Визуализация данных, включая графики и тепловые карты.

### 3. Выявление аномалий
- Применение алгоритмов машинного обучения для детекции аномальных действий.
- Использование алгоритма Isolation Forest для нахождения аномальных пользователей.

### 4. Визуализация результатов
- Построение графиков для представления аномальных данных.
- Создание дашборда для визуализации активности.

## Результаты

### Выявленные аномалии:
- Пользователи с аномально высокой активностью.
- Использование множества IP-адресов за короткий период времени.
- Одновременная активность с разных устройств (Android и iPhone).

### Статистика:
- Всего записей до очистки: 80342
- Аномальных записей (удалено): 32664
- Записей после очистки: 47678

### Очищенные данные:
- Сохранены в файле `cleaned_views_statistics.csv`.
  
## Визуализация данных

В рамках анализа данных был построен график, который показывает, в какие часы дня пользователи чаще всего смотрели видео. Этот график помогает выявить пики активности пользователей, что может быть полезным для анализа аномальной активности или выявления подозрительных паттернов поведения, таких как накрутка просмотров.

### graph.py

График отображает количество просмотров в различные часы суток, что позволяет нам понять, в какое время происходят максимальные пики активности. Это важная информация для мониторинга аномальных пиков или выявления активностей, которые могут быть связаны с ботами или накруткой.

## Пример кода

### anomaly.py

```python
import pandas as pd
from sklearn.ensemble import IsolationForest

# Загрузка данных
data = pd.read_csv('views_statistics.csv')

# Применение Isolation Forest для выявления аномалий
model = IsolationForest(contamination=0.1)
data['anomaly'] = model.fit_predict(data[['views_count', 'user_activity']])

# Сохранение очищенных данных
cleaned_data = data[data['anomaly'] == 1]
cleaned_data.to_csv('cleaned_views_statistics.csv', index=False) 

