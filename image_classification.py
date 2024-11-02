import cv2
import numpy as np
import matplotlib.pyplot as plt

def classify_image_segments(image_path, block_sizes=[(8, 8), (16, 16)], thresholds_dict=None):
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        print("Error: Image not found.")
        return

    # Convert the image to HSV color space
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # Extract brightness (V) from the HSV channel
    brightness = hsv_image[:, :, 2]
    
    # Set brightness thresholds for different block sizes
    if thresholds_dict is None:
        thresholds_dict = {
            (8, 8): [80, 170, 255],  # Thresholds for block size 8x8
            (16, 16): [60, 130, 200]  # Thresholds for block size 16x16
        }
    
    # Step 1: Display the original image
    plt.figure()
    plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    plt.title("Original Image")
    plt.axis('off')
    plt.show()

    # Step 2: Process and classify each block size
    for block_size in block_sizes:
        block_height, block_width = block_size
        image_height, image_width = brightness.shape
        segmented_image = np.zeros_like(image)

        # Get thresholds for the current block size
        thresholds = thresholds_dict[block_size]
        
        # Loop through the image in blocks
        for y in range(0, image_height, block_height):
            for x in range(0, image_width, block_width):
                # Extract block
                block = brightness[y:y + block_height, x:x + block_width]
                
                # Calculate average brightness of the block
                avg_brightness = np.mean(block)
                
                # Classify block by average brightness
                if avg_brightness <= thresholds[0]:
                    color = [255, 0, 0]  # Dark segment - red
                elif avg_brightness <= thresholds[1]:
                    color = [0, 255, 0]  # Medium segment - green
                else:
                    color = [0, 0, 255]  # Bright segment - blue

                # Color the segmented block
                segmented_image[y:y + block_height, x:x + block_width] = color

        # Step 3: Display segmented image with legend
        plt.figure()
        plt.imshow(cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB))
        plt.title(f"Segmented Image (Block Size: {block_size[0]}x{block_size[1]})")
        plt.axis('off')

        # Add legend
        colors = [(255, 0, 0), (0, 255, 0), (0, 0, 255)]
        labels = ["Low Brightness", "Medium Brightness", "High Brightness"]
        for i, color in enumerate(colors):
            plt.scatter([], [], color=np.array(color) / 255.0, label=labels[i])
        plt.legend(title="Brightness Classification", loc='upper right')
        plt.show()

# Call the function with the path to the image and block sizes
image_path = r'image\I22.BMP'
classify_image_segments(image_path, block_sizes=[(8, 8), (16, 16)])
