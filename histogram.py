import cv2
import numpy as np
import matplotlib.pyplot as plt

# img_ori = cv2.imread("input\lena.bmp",-1)#讀取影像檔案，第二個參數為影像的讀取方式
img_ori = cv2.imread("input\lena.bmp", cv2.IMREAD_GRAYSCALE)
# cv2.imshow("ex_test",img_ori)#建立視窗
# cv2.waitKey(0)#第6行WaitKey等待使用者的鍵盤輸入，單位為毫秒(ms)，例如1,000 代表等待1秒再關閉視窗，0則表示持續等待使用者輸入任意鑑後再關閉視窗。
# cv2.destroyAllWindows()#關閉所有視窗

# img_ori = cv2.imread("input\lena.bmp", cv2.IMREAD_GRAYSCALE) # Read image as gray.

img_equal = cv2.equalizeHist(img_ori)

histogram = cv2.calcHist([img_ori], [0], None, [256], [0, 256]) #image, channels, mask, histSize, ranges
hist_equalized = cv2.calcHist([img_equal], [0], None, [256], [0, 256])

# 繪製直方圖
plt.figure(figsize=(12, 6))

plt.subplot(1, 2, 1)
plt.title('Original Image Histogram')
plt.bar(range(256), histogram[:, 0], width=1.0, color='blue')
plt.xlim([0, 256])

plt.subplot(1, 2, 2)
plt.title('Equalized Image Histogram')
plt.bar(range(256), hist_equalized[:, 0], width=1.0, color='green')
plt.xlim([0, 256])

plt.savefig('output/histogram.png')
plt.show()

# save result

