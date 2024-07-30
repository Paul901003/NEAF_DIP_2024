import cv2
import numpy as np
import matplotlib.pyplot as plt

# img_ori = cv2.imread("input\lena.bmp",-1)#讀取影像檔案，第二個參數為影像的讀取方式
img_ori = cv2.imread("input\\binary_task1.bmp", cv2.IMREAD_GRAYSCALE)
# cv2.imshow("ex_test",img_ori)#建立視窗
# cv2.waitKey(0)#第6行WaitKey等待使用者的鍵盤輸入，單位為毫秒(ms)，例如1,000 代表等待1秒再關閉視窗，0則表示持續等待使用者輸入任意鑑後再關閉視窗。
# cv2.destroyAllWindows()#關閉所有視窗

# img_ori = cv2.imread("input\lena.bmp", cv2.IMREAD_GRAYSCALE) # Read image as gray.


plt.figure(figsize = (10, 5))

_,image_binary = cv2.threshold(img_ori, 155, 255, cv2.THRESH_BINARY)

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img_ori, cmap = 'gray')
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('Equalized Image')
plt.imshow(image_binary, cmap = 'gray')
plt.axis('off')

plt.savefig('output/binary.png')
plt.show()

# save result
# cv2.imwrite('output/img_ori.bmp', img_ori)
# cv2.imwrite('output/img_equa.bmp', img_equa)