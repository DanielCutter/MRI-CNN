import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

# imagePath = "C:/Users/dancu/PycharmProjects/firstCNN/data/ad-vs-cn/valid/ad/"
imagePath = "C:/Users/dancu/PycharmProjects/firstCNN/data/ad-vs-cn/train/cn/not_preprocessed/ADNI_018_S_0043_MR_MPR____N3__Scaled_2_Br_20081002103555879_S25956_I118996_z080.png"
directory = os.fsencode(imagePath)

# i = 0
#
# for file in os.listdir(directory):
#     i = i+1
#     filename = os.fsdecode(file)
#     print(filename)
#     cvImg = cv2.imread(imagePath+filename)
#     medianBlurImg = cv2.medianBlur(cvImg, 3)
#     edges = cv2.Canny(cvImg, 100, 200)
#     os.chdir("C:/Users/dancu/PycharmProjects/firstCNN/data/ad-vs-cn/valid/ad_proc/")
#     cv2.imwrite("newBatch-" + str(i) + ".jpg", edges)


# Open the image
cvImg = cv2.imread(imagePath)
print(cvImg)
# Show image
cv2.imshow("pre-processing", cvImg)
cv2.waitKey(0)
# Apply noise reduction and save image
medianBlurImg = cv2.medianBlur(cvImg, 3)
# Show processed image
cv2.imshow("post-medianBlur", medianBlurImg)
cv2.waitKey(0)
# Edge detection
edges = cv2.Canny(cvImg, 0, 300)

plt.subplot(121),plt.imshow(cvImg, cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(122),plt.imshow(edges, cmap='gray')
plt.title('Edge Image'), plt.xticks([]), plt.yticks([])

plt.show()
