import matplotlib.pyplot as plt
import pandas as pd

# Загрузка данных
data = pd.read_csv('views_statistics.csv')

# Преобразование времени в формат datetime
data['time'] = pd.to_datetime(data['time'], errors='coerce')

# Добавление столбцов для дня недели и часа
data['hour'] = data['time'].dt.hour
data['day_of_week'] = data['time'].dt.dayofweek

# Визуализация просмотров по часам
plt.figure(figsize=(10, 6))
data['hour'].value_counts().sort_index().plot(kind='bar', color='skyblue')
plt.title('Распределение просмотров по часам')
plt.xlabel('Часы')
plt.ylabel('Количество просмотров')
plt.show()

