import cv2
import numpy as np
import matplotlib.pyplot as plt

# img_ori = cv2.imread("input\lena.bmp",-1)#讀取影像檔案，第二個參數為影像的讀取方式
img_bgr = cv2.imread("input/yellow.jpg", -1)
# cv2.imshow("ex_test",img_ori)#建立視窗
# cv2.waitKey(0)#第6行WaitKey等待使用者的鍵盤輸入，單位為毫秒(ms)，例如1,000 代表等待1秒再關閉視窗，0則表示持續等待使用者輸入任意鑑後再關閉視窗。
# cv2.destroyAllWindows()#關閉所有視窗

# img_ori = cv2.imread("input\lena.bmp", cv2.IMREAD_GRAYSCALE) # Read image as gray.

lower = np.array([26,43,46])  # 轉換成 NumPy 陣列，範圍稍微變小 ( 55->30, 70->40, 252->200 )
upper = np.array([34,255,255]) # 轉換成 NumPy 陣列，範圍稍微加大 ( 70->90, 80->100, 252->255 )

image_hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
mask = cv2.inRange(image_hsv, lower, upper)

contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
if contours:
    # 找到最大的輪廓
    largest_contour = max(contours, key=cv2.contourArea)
    
    # 獲取包圍黃色區域的邊界框
    x, y, w, h = cv2.boundingRect(largest_contour)
    
    # # 裁切出黃色區域
    # cropped_image = image_hsv[y - 10: y + h + 10 , x - 10: x + w + 10]
    
result_img = cv2.bitwise_and(img_bgr, img_bgr, mask = mask)

img_rgb = cv2.cvtColor(result_img, cv2.COLOR_BGR2RGB)[y - 20: y + h + 20 , x - 20: x + w + 20]

img_show = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)[y - 20: y + h + 20 , x - 20: x + w + 20]

plt.figure(figsize = (10, 5))

plt.subplot(1, 2, 1)
plt.title('Original Image')
plt.imshow(img_bgr)
plt.axis('off')

plt.subplot(1, 2, 2)
plt.title('filter image')
plt.imshow(img_show)
plt.axis('off')

plt.savefig('output/hsv.png')
plt.show()


