import get_degree
import rectangleExtraction
import cv2
from scipy import ndimage
import lengthMeasure as lm
import os

""""--- 画像の回転 ---
    画像が単個体であれば有用"""
"""
# imagesディレクトリを探索 #
for dirname, _, filenames in os.walk('images/'):
    for filename in filenames:
        if filename != ".DS_Store":
            
            image_file = dirname + filename
            img = cv2.imread(image_file)
            deg = get_degree.get_degree(img) # 角度の算出
            rotate_img = ndimage.rotate(img, deg) # 回転
            
            # 書き出し #
            cv2.imwrite(f'rotate_images/{filename}', rotate_img)
"""

"""--- 画像の輪郭抽出 ---"""
for dirname, _, filenames in os.walk('images/'):
#for dirname, _, filenames in os.walk('rotate_images/'): # 画像の回転を行った場合
    for filename in filenames:
        if filename != ".DS_Store":
            
            image_file = dirname + filename
            img = cv2.imread(image_file)
            # 画像の回転を行った時(単個体) #
            #(w_list, h_list, aspect_list) = contour_extraction.contour_extraction(img, filename) # 輪郭抽出と書き出し
            # 画像を回転していない時 #
            #rectangleExtraction.rectangle_extraction2(img, filename)
            lm.lengthMeasure(img, filename)
            
            