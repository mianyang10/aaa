import matplotlib.pyplot as plt
from PIL import Image, ImageEnhance, ImageFilter
import numpy as np

# 打开图像
image = Image.open('D:/picture/9.jpg')
# 原始图像
plt.subplot(1, 4, 1)
plt.imshow(image)
plt.title('Original Image')

# 图像增强
enhancer = ImageEnhance.Contrast(image)
enhanced_image = enhancer.enhance(2.0)  # 增强对比度

# 增强后的图像
plt.subplot(1, 4, 2)
plt.imshow(enhanced_image)
plt.title('Enhanced Image')

# 去除背景噪点
filtered_image = enhanced_image.filter(ImageFilter.MedianFilter(5))

# 去噪后的图像
plt.subplot(1, 4, 3)
plt.imshow(filtered_image)
plt.title('Filtered Image')

# 伽马变换和二值化处理
gamma = 0.5  # 伽马值
threshold = 128  # 阈值

# 转换为灰度图像
gray_image = filtered_image.convert('L')

# 将图像转换为NumPy数组
image_array = np.array(gray_image)

# 应用伽马变换
gamma_corrected_array = 255 * (image_array / 255) ** gamma

# 将NumPy数组转换回图像
gamma_corrected_image = Image.fromarray(gamma_corrected_array.astype(np.uint8))

# 应用二值化处理
binary_image = gamma_corrected_image.point(lambda x: 255 if x > threshold else 0, mode='1')

# 伽马变换和二值化处理后的图像
plt.subplot(1, 4, 4)
plt.imshow(binary_image, cmap='gray')
plt.title('Gamma and Binary Image')

# 显示图像
plt.tight_layout()
plt.show()

# 保存伽马变换和二值化处理后的图像
binary_image.save("gamma_binary_image.jpg")

import cv2  
import numpy as np  
import matplotlib.pyplot as plt 

img = cv2.imread(r'gamma_binary_image.jpg')  
  
# 定义一个3x3的卷积核  
kernel = np.ones((5,5),np.uint8)  
  
# 膨胀和腐蚀函数：  
img_erosion = cv2.erode(img, kernel, iterations=1)  
plt.imshow(img_erosion)  
plt.show()  
  
img_dilation = cv2.dilate(img, kernel, iterations=1)  
plt.imshow(img_dilation)  
plt.show()  
  
er2 = cv2.erode(img_dilation, kernel, iterations=1)  
plt.imshow(er2)  
plt.show()  
  

dila = cv2.dilate(img_erosion, kernel, iterations=1)  
plt.imshow(dila)  
plt.show()


import pytesseract
from PIL import Image

def OCR_demo():
    # 导入OCR安装路径，如果设置了系统环境，就可以不用设置了
    # pytesseract.pytesseract.tesseract_cmd = r"D:\Program Files\Tesseract-OCR\tesseract.exe"
    # 打开要识别的图片

#     image = Image.open('D:/picture/b.jpg')
    image =img_erosion
    # 使用pytesseract调用image_to_string方法进行识别，传入要识别的图片，lang='chi_sim'是设置为中文识别，
    text = pytesseract.image_to_string(image, lang='chi_sim')

    print(text)


if __name__ == '__main__':
    OCR_demo()
