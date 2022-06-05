import cv2
import rotateImage as rimg
import extraction
import numpy as np
import os
from PIL import Image, ImageSequence
from imageProcessing import imageProcessing

# tif画像に変換する
name = "test"# 大元のファイル名
#os.mkdir(f"images/{name}")
for dirname, _, filenames in os.walk('image/'):
    for filename in filenames:
        if filename != ".DS_Store":
            image_file = dirname + filename
            img = Image.open(image_file)
            itr = ImageSequence.Iterator(img)
            length = sum(1 for _ in itr)
            for count in range(length):
                img = itr[count] # 2ページ目
                img = np.array(img)
                cv2.imwrite(f'images/{name}/{count}.png', img)

# 前処理用 #
ip = imageProcessing(name, thres1=145, thres2=255, alpha=1.3, beta=0)
# 実行 #
# ディレクトリ作成
#os.mkdir(f"extraction/{name}")
for dirname, _, filenames in os.walk(f'images/{name}/'):
    for filename in filenames:
        if filename != ".DS_Store":
            
            image_file = dirname + filename
            img = cv2.imread(image_file)
            new_img = ip.doProcessing(img, filename)
            # 画像の回転を行った時(単個体) #
            extraction.extract(new_img, img, filename, name)


            