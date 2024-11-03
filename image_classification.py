import cv2
import numpy as np
import matplotlib.pyplot as plt
from skimage.feature import graycomatrix, graycoprops
from skimage.measure import shannon_entropy  # Для расчета энтропии

# Загрузка изображения
image_path = 'image/I22.BMP'  # Укажите путь к вашему изображению
image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)  # Используем серый масштаб для текстурного анализа

# Преобразование изображения в цветовое пространство HSV
image_hsv = cv2.cvtColor(cv2.imread(image_path), cv2.COLOR_BGR2HSV)
v_channel = image_hsv[:, :, 2]  # Яркостная составляющая

# Сегментация по яркости
thresholds = np.linspace(v_channel.min(), v_channel.max(), 5)
segmented_brightness = np.digitize(v_channel, thresholds)

# Функция для расчета текстурных характеристик (контраст, энергия, гомогенность и энтропия)
def calculate_texture_features(image, distances=[5], angles=[0]):
    glcm = graycomatrix(image, distances=distances, angles=angles, symmetric=True, normed=True)
    contrast = graycoprops(glcm, 'contrast')[0, 0]
    energy = graycoprops(glcm, 'energy')[0, 0]
    homogeneity = graycoprops(glcm, 'homogeneity')[0, 0]
    entropy = shannon_entropy(image)  # Расчет энтропии
    return contrast, energy, homogeneity, entropy

# Сегментация на основе реальных текстурных метрик
image_contrast = np.zeros(image.shape)
image_energy = np.zeros(image.shape)
image_homogeneity = np.zeros(image.shape)
image_entropy = np.zeros(image.shape)

# Скользящее окно для расчета текстурных метрик
window_size = 25
for i in range(0, image.shape[0] - window_size, window_size):
    for j in range(0, image.shape[1] - window_size, window_size):
        window = image[i:i + window_size, j:j + window_size]
        contrast, energy, homogeneity, entropy = calculate_texture_features(window)
        image_contrast[i:i + window_size, j:j + window_size] = contrast
        image_energy[i:i + window_size, j:j + window_size] = energy
        image_homogeneity[i:i + window_size, j:j + window_size] = homogeneity
        image_entropy[i:i + window_size, j:j + window_size] = entropy

# Визуализация нескольких графиков с легендами (colorbar)
fig, axs = plt.subplots(2, 3, figsize=(15, 10))

# Первая строка - структурные метрики
im1 = axs[0, 0].imshow(segmented_brightness, cmap='jet')
axs[0, 0].set_title('Segments classified by brightness')
fig.colorbar(im1, ax=axs[0, 0])

im2 = axs[0, 1].imshow(image_energy, cmap='jet')
axs[0, 1].set_title('Segments classified by energy (real)')
fig.colorbar(im2, ax=axs[0, 1])

im3 = axs[0, 2].imshow(image_contrast, cmap='jet')
axs[0, 2].set_title('Segments classified by contrast (real)')
fig.colorbar(im3, ax=axs[0, 2])

# Вторая строка - статистические метрики
im4 = axs[1, 0].imshow(image_homogeneity, cmap='jet')
axs[1, 0].set_title('Segments classified by homogeneity (real)')
fig.colorbar(im4, ax=axs[1, 0])

# Сегменты, классифицированные по энтропии
im5 = axs[1, 1].imshow(image_entropy, cmap='jet')
axs[1, 1].set_title('Segments classified by entropy (real)')
fig.colorbar(im5, ax=axs[1, 1])

# Отключение осей для всех графиков
for ax in axs.flat:
    ax.axis('off')

plt.tight_layout()
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
calculate_and_visualize_errors(segmented_brightness)