import numpy as np
import cv2
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
import math


# Görüntüyü yükle
img = cv2.imread('d.jpg')

# Görüntüyü gray formata donustur
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)


# gray goruntu histogrami
hist,bins = np.histogram(gray.flatten(),256,[0,256])
# Histogram grafiğini çizdir
plt.hist(gray.flatten(),256,[0,256])
plt.xlim([0,256])
plt.title("Orijinal histogram")
plt.show()




# =============== gri seviye logaritmik donusumu ( gerek gorulurse )===============
# c = 255 / np.log(1 + np.max(gray))
# log_transformed = c * np.log(1 + gray)
# hist,bins = np.histogram(log_transformed.flatten(),256,[0,256])
# # Histogram grafiğini çizdir
# plt.hist(log_transformed.flatten(),256,[0,256])
# plt.xlim([0,256])
# plt.title("Orijinal histogram")
# plt.show()
# # Tamsayı değerlere dönüştürün
# log_transformed = np.uint8(log_transformed)

# # Görüntüyü gösterin
# cv2.imshow('Logaritmik Dönüşüm', log_transformed)
# =================================================================================





# Goruntu uzerinde canny ile kenar belirle
# thresh degerleri, goruntu uzerinde yapilan denemeler sonucu bulunmustur.
canny = cv2.Canny(gray, threshold1=125, threshold2=130)
cv2.imshow("canny", canny)
cv2.imwrite("canny.jpg", canny)


# Görüntüyü RGB formatına dönüştür
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Resmi yeniden boyutlandır
pixel_values = img_rgb.reshape((-1, 3))

# K-Means kümeleme işlemini gerçekleştir
# 7 kume olarak belirlendi ancak bu deger islenen goruntuye gore degistirilebilir.
kmeans = KMeans(n_clusters=7, random_state=0).fit(pixel_values)

# Her pikselin küme etiketini al
labels = kmeans.labels_

# Her pikselin küme merkezini al
centers = np.uint8(kmeans.cluster_centers_)

# Her pikselin rengini küme merkezi ile değiştir
# bu asama, goruntu uzerindeki siniflarin ayristrilmasi ve etkili bir gorsel olsuturulabilmesi icin yapilmistir.
res = centers[labels.flatten()]

# Resmi yeniden boyutlandır
segmented_image = res.reshape((img_rgb.shape))


# Resmi göster
cv2.imshow('Segmented Image', segmented_image)
cv2.imwrite("segmented_image.jpg", segmented_image)

# canny filtre sonucu ile k-means ile olsuturulan goruntuyu birlestir
# burada canny filtresi bir sketch etkisi olusturabilmek icin kullanilmistir, nesnelerin cevresini belirtir.
# k-means ise nesnelerin genel alanlari uzerinde bir renklendirme yaparak goruntuyu daha digital hale getirmektedir.
segmented_image[..., 0] = cv2.bitwise_or(segmented_image[...,1]-canny , canny)
cv2.imshow("result", segmented_image)
cv2.imwrite("result.jpg", segmented_image)

hist,bins = np.histogram(segmented_image.flatten(),256,[0,256])
# Histogram grafiğini çizdir
plt.hist(segmented_image.flatten(),256,[0,256])
plt.xlim([0,256])
plt.title("Dönüştürülmüş görüntü histogramı")
plt.show()




cv2.waitKey(0)
cv2.destroyAllWindows()