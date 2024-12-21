import pandas as pd
import numpy as np


def write_file(file_path, data):
    with open(file_path, 'w') as f:
        f.write(data)


# task 9
def find_stable_objects(file_path, n, m):
    df = pd.read_csv(file_path, sep=';')

    unique_objects = df['ID'].unique()

    stable_objects = []

    for obj_id in unique_objects:
        obj_data = df[df['ID'] == obj_id]

        if len(obj_data) < n:
            continue

        for i in range(len(obj_data) - n):
            current = obj_data.iloc[i]
            future = obj_data.iloc[i + n]

            x_shift = abs(current['x_left'] - future['x_left'])
            y_shift = abs(current['y_top'] - future['y_top'])
            total_shift = np.sqrt(x_shift ** 2 + y_shift ** 2)

            if total_shift <= m:
                stable_objects.append(obj_id)
                break

    write_file('task_9.txt', str(set(stable_objects)))

    return list(set(stable_objects))


# task 7
def calculate_average_lifetime(file_path, fps):
    df = pd.read_csv(file_path, sep=';')

    object_lifetimes = {}
    unique_objects = df['ID'].unique()

    for obj_id in unique_objects:
        frames = df[df['ID'] == obj_id]

        lifetime = frames['frame'].nunique()
        object_lifetimes[obj_id] = lifetime

    total_lifetime = sum(object_lifetimes.values())
    total_objects = len(object_lifetimes)

    average_lifetime_frames = total_lifetime / total_objects if total_objects > 0 else 0
    print("Среднее время нахождения в кадре (время жизни объектов, в кадрах):", average_lifetime_frames)
    average_lifetime_seconds = average_lifetime_frames / fps

    write_file('task_7.txt', str(average_lifetime_seconds))

    return average_lifetime_seconds


# task 1
def count_large_objects(file_path, min_width, min_height):
    df = pd.read_csv(file_path, sep=';')

    print("Общее число объектов: ", df['ID'].nunique())

    df['width'] = df['x_right'] - df['x_left']
    df['height'] = df['y_bottom'] - df['y_top']

    large_objects = df[(df['width'] > min_width) & (df['height'] > min_height)]

    count = large_objects['ID'].nunique()

    write_file('task_1.txt', str(count))

    return count


def main():
    file_path = 'trajectories.csv'
    # task 9
    n = 5  # Количество кадров
    m = 4  # Смещение в пикселях
    stable_objects = find_stable_objects(file_path, n, m)
    print("Стабильные объекты (ID):", sorted(stable_objects))

    # task 7
    fps = 30  # Частота кадров
    average_lifetime = calculate_average_lifetime(file_path, fps)
    print("Среднее время нахождения в кадре (время жизни объектов, в секундах):", average_lifetime)

    # task 1
    min_width = 120  # Ширина кадра
    min_height = 120  # Высота кадра
    large_object_count = count_large_objects(file_path, min_width, min_height)
    print("Количество объектов с размерами больше заданных:", large_object_count)


if __name__ == '__main__':
    main()
