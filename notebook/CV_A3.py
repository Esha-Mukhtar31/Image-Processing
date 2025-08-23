#!/usr/bin/env python
# coding: utf-8

# In[18]:


# Import necessary libraries
import numpy as np  # NumPy is used for numerical operations, such as handling arrays and performing mathematical operations.
from PIL import Image  # PIL (Python Imaging Library) is used for image processing tasks, such as opening, manipulating, and saving image files.
import matplotlib.pyplot as plt  # Matplotlib is a plotting library used for creating visualizations, such as graphs and images.


def otsu_thresholding(image):
    """Compute the optimal threshold using Otsu's method."""
    # Compute histogram
    hist, _ = np.histogram(image.flatten(), bins=256, range=[0, 256])
    total_pixels = np.sum(hist)
    
    # Initialize variables
    max_variance = 0
    optimal_threshold = 0
    cumulative_sum = np.cumsum(hist)
    cumulative_mean = np.cumsum(hist * np.arange(256))
    
    for i in range(1, 256):
        weight_background = cumulative_sum[i] / total_pixels
        weight_foreground = 1 - weight_background
        
        if weight_background == 0 or weight_foreground == 0:
            continue
        
        mean_background = cumulative_mean[i] / cumulative_sum[i]
        mean_foreground = (cumulative_mean[-1] - cumulative_mean[i]) / (total_pixels - cumulative_sum[i])
        
        between_class_variance = weight_background * weight_foreground * (mean_background - mean_foreground) ** 2
        
        if between_class_variance > max_variance:
            max_variance = between_class_variance
            optimal_threshold = i
    
    return optimal_threshold

def regenerate_img(img, threshold):
    """Binarize the image based on the threshold."""
    return np.where(img >= threshold, 255, 0).astype(np.uint8)

def main():
    # Load the original color image and convert to grayscale
    original_image_path = 'img.jfif'  # Replace with your image path
    img = Image.open(original_image_path)
    grayscale_image = img.convert('L')
    
    # Convert images to numpy arrays for processing
    img_array = np.array(img)
    grayscale_img_array = np.array(grayscale_image)
    
    # Compute Otsu's threshold
    otsu_threshold = otsu_thresholding(grayscale_img_array)
    print(f'Otsu\'s Optimal Threshold: {otsu_threshold}')
    
    # Binarize the grayscale image using Otsu's threshold
    binarized_image_otsu = regenerate_img(grayscale_img_array, otsu_threshold)
    
    # Binarize the grayscale image using a fixed threshold for comparison
    fixed_threshold = 128
    binarized_image_fixed = regenerate_img(grayscale_img_array, fixed_threshold)
    
    # Calculate the histogram of the grayscale image
    histogram, bin_edges = np.histogram(grayscale_img_array.flatten(), bins=256, range=[0, 256])
    
    # Plot the results
    plt.figure(figsize=(24, 8))
    
    # Plot the original color image
    plt.subplot(2, 3, 1)
    plt.imshow(img_array)
    plt.title('Original Color Image')
    plt.axis('off')
    
    # Plot the grayscale image
    plt.subplot(2, 3, 2)
    plt.imshow(grayscale_img_array, cmap='gray')
    plt.title('Grayscale Image')
    plt.axis('off')
    
    # Plot the histogram with Otsu's threshold
    plt.subplot(2, 3, 3)
    plt.hist(grayscale_img_array.flatten(), bins=256, range=[0, 256], color='gray')
    plt.axvline(otsu_threshold, color='red', linestyle='dashed', linewidth=1, label=f'Otsu Threshold ({otsu_threshold})')
    plt.title('Histogram with Otsu\'s Threshold')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    plt.legend()
    
    # Plot the histogram of grayscale image (separate)
    plt.subplot(2, 3, 4)
    plt.hist(grayscale_img_array.flatten(), bins=256, range=[0, 256], color='blue', alpha=0.7)
    plt.title('histogram of pixel intensities in the grayscale image.')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')
    
    # Plot the binarized image using fixed threshold
    plt.subplot(2, 3, 5)
    plt.imshow(binarized_image_fixed, cmap='gray')
    plt.title('Binarized Image (Fixed Threshold)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    main()


# In[ ]:




