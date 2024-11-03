import cv2
import numpy as np
import matplotlib.pyplot as plt

# Загрузка изображения
image_path = 'image/I22.BMP'
image = cv2.imread(image_path)

# Преобразование изображения в цветовое пространство HSV
image_hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Функция для отображения изображения
def display_image(title, img, cmap=None):
    plt.figure(figsize=(6, 6))
    if cmap:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB), cmap=cmap)
    else:
        plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title(title)
    plt.axis('off')
    plt.show()

# Отображаем исходное изображение
display_image('Original Image', image)

# Классификация по яркости в цветовой модели HSV (V-компонента)
v_channel = image_hsv[:, :, 2]

# Определение порогов для сегментации по яркости
thresholds = np.linspace(v_channel.min(), v_channel.max(), 5)  # 4 порога, 5 сегментов

# Функция для сегментации изображения по яркости
def segment_image(v_channel, thresholds):
    segments = np.digitize(v_channel, thresholds)
    return segments

# Сегментация изображения
segmented_image = segment_image(v_channel, thresholds)

# Отображение сегментированного изображения
plt.figure(figsize=(6, 6))
plt.imshow(segmented_image, cmap='gray')
plt.title('Segmented Image by Brightness (5 segments)')
plt.axis('off')
plt.show()

# Описание сегментов на основе яркости
def describe_segments(v_channel, thresholds):
    descriptions = []
    for i in range(1, len(thresholds)):
        mask = (v_channel >= thresholds[i-1]) & (v_channel < thresholds[i])
        segment_mean = np.mean(v_channel[mask]) if mask.any() else 0
        descriptions.append(f'Segment {i}: Brightness between {thresholds[i-1]:.2f} and {thresholds[i]:.2f}, mean: {segment_mean:.2f}')
    return descriptions

# Описание сегментов
segment_descriptions = describe_segments(v_channel, thresholds)

# Функция для визуализации текстовых выводов
def visualize_text_output(text_list, title):
    plt.figure(figsize=(8, 4))
    plt.text(0.5, 0.5, "\n".join(text_list), horizontalalignment='center', verticalalignment='center', 
             wrap=True, fontsize=12)
    plt.title(title)
    plt.axis('off')
    plt.show()

# Визуализируем текстовые выводы описания сегментов
visualize_text_output(segment_descriptions, 'Segment Descriptions')

# Функция для визуализации ошибок первого рода (false negatives)
def visualize_false_negatives(segmented_image, reference_mask):
    false_negatives_mask = (segmented_image != reference_mask) & (reference_mask != 0)
    
    # Создаем изображение, где отображаются только false negatives
    false_negatives_image = np.zeros_like(segmented_image, dtype=np.uint8)
    false_negatives_image[false_negatives_mask] = 255  # Выделяем ошибки белым цветом
    
    plt.figure(figsize=(6, 6))
    plt.imshow(false_negatives_image, cmap='gray')
    plt.title('False Negatives Visualization')
    plt.axis('off')
    plt.show()

# Модифицированная функция для расчета и визуализации ошибок первого рода
def calculate_and_visualize_errors(segmented_image):
    # Предположим, что у нас есть некоторая эталонная маска для проверки ошибок
    reference_mask = np.random.randint(1, 6, size=segmented_image.shape)

    # Ошибки первого рода - когда объект отнесен к другому классу (ложноотрицательные)
    false_negatives = np.sum((segmented_image != reference_mask) & (reference_mask != 0))
    
    # Визуализируем количество ошибок первого рода
    error_info = [f'False Negatives (Error Type I): {false_negatives}']
    visualize_text_output(error_info, 'Error Type I: False Negatives')
    
    # Визуализация false negatives
    visualize_false_negatives(segmented_image, reference_mask)

# Вызов функции для расчета и визуализации ошибок первого рода
calculate_and_visualize_errors(segmented_image)
