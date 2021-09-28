import cv2
import numpy as np
import math

# 回転させた場合 #
def rectangle_extraction(img, filename):
    # 輪郭の縦横比を計算する用 #
    w_list = []
    h_list = []
    aspect_list = []
    
    l_img = img.copy()
    # グレースケールに変換する。
    gray = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)

    # 2値化する
    ret, bin_img = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)
    
    # 輪郭抽出 #
    contours, hierarchy = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # 輪郭を１つずつ書き込んで出力
    for cnt in contours:

        x, y, w, h = cv2.boundingRect(cnt)
        w_list.append(w)
        h_list.append(h)
        aspect_list.append(h/w)
        cv2.rectangle(l_img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    cv2.imwrite(f'extraction_images/{filename}', l_img)
    
    return(w_list,h_list,aspect_list)

# 回転処理を指定していない場合 #
def rectangle_extraction2(img, filename):
    # 輪郭の縦横比を計算する用 #
    w_list = []
    h_list = []
    aspect_list = []
    
    l_img = img.copy()
    # グレースケールに変換する。
    gray = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
    cv2.imwrite(f"gray_images/{filename}", gray)

    # 2値化する
    ret, bin_img = cv2.threshold(gray, 200, 225, cv2.THRESH_BINARY)
    cv2.imwrite(f"bin_images/{filename}", bin_img)
    
    # 輪郭抽出 #
    contours, hierarchy = cv2.findContours(
        bin_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE
    )
    
    # 輪郭を１つずつ書き込んで出力
    for cnt in contours:

        rect = cv2.minAreaRect(cnt)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        w = math.sqrt((box[0][0]  - box[1][0])**2 + (box[0][1]  - box[1][1])**2)
        h = math.sqrt((box[1][0]  - box[2][0])**2 + (box[1][1]  - box[2][1])**2)
        area = w * h
        if  area >1000:
            # アスペクト比計算(最大1)
            if h > w: 
                aspect = w / h
            elif w >= h:
                aspect = h/ w
            l_img = cv2.drawContours(l_img,[box],0,(0,0,255),2)
            cv2.drawContours(l_img,[box],0,(0,0,255),2)
            font = cv2.FONT_HERSHEY_SIMPLEX # 文字font
            cv2.putText(l_img, str(aspect),(box[0][0],box[0][1]), font, 0.5,(0,0,255),2 )

    cv2.imwrite(f'extraction_images/{filename}', l_img)
    