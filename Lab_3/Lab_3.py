import pandas as pd
import numpy as np

# task 9
def find_stable_objects(file_path, N, M):
    # Чтение CSV файла
    df = pd.read_csv(file_path, sep=';')

    # Получаем уникальные ID объектов
    unique_objects = df['frame'].unique()

    stable_objects = []

    for obj_id in unique_objects:
        # Фильтруем данные для конкретного объекта
        obj_data = df[df['frame'] == obj_id]

        # Проверяем, если у объекта достаточно кадров
        if len(obj_data) < N:
            continue

        # Считаем смещения между кадрами
        for i in range(len(obj_data) - N):
            # Получаем координаты для текущего и N-го кадров
            current = obj_data.iloc[i]
            future = obj_data.iloc[i + N]

            # Вычисляем смещение по координатам
            x_shift = abs(current['x_left'] - future['x_left'])
            y_shift = abs(current['y_top'] - future['y_top'])
            total_shift = np.sqrt(x_shift ** 2 + y_shift ** 2)

            # Проверяем, меньше ли смещение M
            if total_shift <= M:
                stable_objects.append(obj_id)
                break  # Выходим из цикла, если нашли стабильный объект

    return list(set(stable_objects))  # Уникальные ID стабильных объектов

# task 7
def calculate_average_lifetime(file_path, fps):
    # Чтение CSV файла
    df = pd.read_csv(file_path, sep=';')

    # Создаём словарь для хранения времени жизни объектов
    object_lifetimes = {}

    # Группируем данные по ID объекта и считаем количество уникальных кадров
    for obj_id in df['frame'].unique():
        # Получаем все кадры для текущего объекта
        frames = df[df['frame'] == obj_id]

        # Вычисляем время жизни как количество уникальных кадров
        lifetime = frames['ID'].nunique()  # Предполагается, что 'ID' - это столбец с ID кадров
        object_lifetimes[obj_id] = lifetime

    # Рассчитываем среднее время жизни в секундах
    total_lifetime = sum(object_lifetimes.values())
    total_objects = len(object_lifetimes)

    average_lifetime_frames = total_lifetime / total_objects if total_objects > 0 else 0
    average_lifetime_seconds = average_lifetime_frames / fps  # Преобразование в секунды

    return average_lifetime_seconds

# task 1
def count_large_objects(file_path, min_width, min_height):
    # Чтение CSV файла
    df = pd.read_csv(file_path, sep=';')

    print("Общее число объектов: ",df['frame'].nunique())

    # Вычисляем ширину и высоту объектов
    df['width'] = df['x_right'] - df['x_left']
    df['height'] = df['y_bottom'] - df['y_top']

    # Фильтруем объекты по заданным размерам
    large_objects = df[(df['width'] > min_width) & (df['height'] > min_height)]

    # Подсчитываем количество уникальных объектов по идентификаторам в столбце 'frame'
    count = large_objects['frame'].nunique()  # Используем 'frame' как идентификатор

    return count


# Пример использования
file_path = 'trajectories.csv'
N = 5  # Количество кадров
M = 3  # Максимальное смещение в пикселях
# stable_objects = find_stable_objects(file_path, N, M)
# print("Стабильные объекты (ID):", sorted(stable_objects))

# fps = 30  # Частота кадров
# average_lifetime = calculate_average_lifetime(file_path, fps)
# print("Среднее время нахождения в кадре (время жизни объектов, в секундах):", average_lifetime)

min_width = 120  # Минимальная ширина
min_height = 120  # Минимальная высота
large_object_count = count_large_objects(file_path, min_width, min_height)
print("Количество объектов с размерами больше заданных:", large_object_count)
