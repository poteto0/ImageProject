import cv2

# 二値化 #
def binarization(img, filename, thres1=0, thres2=255):
    #--- 画像の二値化 ---#
    # 画像のコピー
    l_img = img.copy()
    # グレースケールに変換する。
    gray = cv2.cvtColor(l_img, cv2.COLOR_BGR2GRAY)
    # 2値化する
    ret, bin_img = cv2.threshold(gray, thres1, thres2, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    bin_img = cv2.bitwise_not(bin_img)
    
    #--- 保存 ---#
    cv2.imwrite(f"gray_images/{filename}", gray) # グレースケール
    cv2.imwrite(f"bin_images/{filename}", bin_img)
    
    return bin_img #二値化を返却

# モルフォロジー変換 #
def morphology(img, filename):
    #--- モルフォロジー変換 ---#
    kernel5x5 = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
    mor_img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel5x5, iterations=2)
    
    #--- 保存 ---#
    cv2.imwrite(f"mono/{filename}", mor_img)
    
    return mor_img

    