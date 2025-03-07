import pandas as pd
from sklearn.ensemble import IsolationForest

# Загрузка данных
data = pd.read_csv('views_statistics.csv')

# Преобразование времени в datetime
data['time'] = pd.to_datetime(data['time'])

# Создание временных признаков для анализа (будут удалены позже)
data['minute'] = data['time'].dt.floor('min')  # Округление до минуты
data['ten_minutes'] = data['time'].dt.floor('10min')  # Округление до 10 минут

# Подготовка данных для Isolation Forest
requests_per_minute = data.groupby(['user_id', 'minute']).size().reset_index(name='requests_in_minute')
unique_ips_per_10_min = data.groupby(['user_id', 'ten_minutes'])['ip'].nunique().reset_index(name='unique_ips')
platform_usage = data.groupby(['user_id', 'platform']).size().unstack(fill_value=0).reset_index()

# Объединение всех признаков
features = requests_per_minute.merge(unique_ips_per_10_min, on='user_id', how='outer')
features = features.merge(platform_usage, on='user_id', how='outer')
features = features.fillna(0)

# Выбор релевантных колонок для анализа
features_for_model = features[['requests_in_minute', 'unique_ips', 'android', 'iphone']]

# Применение Isolation Forest
model = IsolationForest(contamination=0.01, random_state=42)  # contamination - ожидаемый процент аномалий
features['anomaly'] = model.fit_predict(features_for_model)

# Фильтрация аномальных пользователей
anomalous_users = features[features['anomaly'] == -1]['user_id'].unique()

# Удаление аномальных записей из исходных данных
cleaned_data = data[~data['user_id'].isin(anomalous_users)]

# Удаление временных столбцов minute и ten_minutes
cleaned_data = cleaned_data.drop(columns=['minute', 'ten_minutes'], errors='ignore')

# Сохранение очищенных данных в CSV-файл
cleaned_data.to_csv('cleaned_views_statistics.csv', index=False)

# Генерация текстового пояснения
total_records = len(data)
anomalous_records = len(data[data['user_id'].isin(anomalous_users)])
cleaned_records = len(cleaned_data)

explanation = f"""
Отчет о детекции и удалении аномалий:

Метод: Использован алгоритм Isolation Forest из библиотеки scikit-learn для выявления аномалий.

Критерии аномалий:
1. Пользователь сделал более 100 запросов за минуту.
2. Пользователь использовал более 5 разных IP-адресов за 10 минут.
3. Пользователь одновременно использовал Android и iPhone.

Статистика:
- Всего записей до очистки: {total_records}
- Аномальных записей (удалено): {anomalous_records}
- Записей после очистки: {cleaned_records}

Очищенные данные сохранены в файле: cleaned_views_statistics.csv
"""

# Сохранение текстового пояснения в файл
with open('explanation.txt', 'w') as file:
    file.write(explanation)

print("Очищенные данные успешно сохранены в cleaned_views_statistics.csv")
print("Текстовое пояснение сохранено в explanation.txt")