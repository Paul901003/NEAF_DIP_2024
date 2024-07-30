import cv2
import numpy as np
import matplotlib.pyplot as plt

# Read the image as grayscale
img_ori = cv2.imread("input/lena_noise.png", cv2.IMREAD_GRAYSCALE)

# Apply mean blur with different kernel sizes
mean_blur_3 = cv2.blur(img_ori, (3, 3))
mean_blur_5 = cv2.blur(img_ori, (5, 5))
mean_blur_7 = cv2.blur(img_ori, (7, 7))

# Apply Gaussian blur with different kernel sizes and sigma
gaussian_blur_3 = cv2.GaussianBlur(img_ori, (3, 3), 0)
gaussian_blur_5 = cv2.GaussianBlur(img_ori, (5, 5), 0)
gaussian_blur_7 = cv2.GaussianBlur(img_ori, (7, 7), 0)

# Apply median blur with different kernel sizes
median_blur_3 = cv2.medianBlur(img_ori, 3)
median_blur_5 = cv2.medianBlur(img_ori, 5)
median_blur_7 = cv2.medianBlur(img_ori, 7)

# Plot the original and blurred images
plt.figure(figsize=(15, 10))

# Original image
plt.subplot(3, 4, 1)
plt.imshow(img_ori, cmap='gray')
plt.title('Original Image')
plt.axis('off')

# Mean blur
plt.subplot(3, 4, 2)
plt.imshow(mean_blur_3, cmap='gray')
plt.title('Mean Blur (3x3)')
plt.axis('off')

plt.subplot(3, 4, 3)
plt.imshow(mean_blur_5, cmap='gray')
plt.title('Mean Blur (5x5)')
plt.axis('off')

plt.subplot(3, 4, 4)
plt.imshow(mean_blur_7, cmap='gray')
plt.title('Mean Blur (7x7)')
plt.axis('off')

# Gaussian blur
plt.subplot(3, 4, 6)
plt.imshow(gaussian_blur_3, cmap='gray')
plt.title('Gaussian Blur (3x3)')
plt.axis('off')

plt.subplot(3, 4, 7)
plt.imshow(gaussian_blur_5, cmap='gray')
plt.title('Gaussian Blur (5x5)')
plt.axis('off')

plt.subplot(3, 4, 8)
plt.imshow(gaussian_blur_7, cmap='gray')
plt.title('Gaussian Blur (7x7)')
plt.axis('off')

# Median blur
plt.subplot(3, 4, 10)
plt.imshow(median_blur_3, cmap='gray')
plt.title('Median Blur (3x3)')
plt.axis('off')

plt.subplot(3, 4, 11)
plt.imshow(median_blur_5, cmap='gray')
plt.title('Median Blur (5x5)')
plt.axis('off')

plt.subplot(3, 4, 12)
plt.imshow(median_blur_7, cmap='gray')
plt.title('Median Blur (7x7)')
plt.axis('off')

plt.tight_layout()
plt.savefig('output/blur.png')
plt.show()

# Save results
# cv2.imwrite('output/img_ori.bmp', img_ori)
# cv2.imwrite('output/mean_blur_3.bmp', mean_blur_3)
# cv2.imwrite('output/mean_blur_5.bmp', mean_blur_5)
# cv2.imwrite('output/mean_blur_7.bmp', mean_blur_7)
# cv2.imwrite('output/gaussian_blur_3.bmp', gaussian_blur_3)
# cv2.imwrite('output/gaussian_blur_5.bmp', gaussian_blur_5)
# cv2.imwrite('output/gaussian_blur_7.bmp', gaussian_blur_7)
# cv2.imwrite('output/median_blur_3.bmp', median_blur_3)
# cv2.imwrite('output/median_blur_5.bmp', median_blur_5)
# cv2.imwrite('output/median_blur_7.bmp', median_blur_7)
