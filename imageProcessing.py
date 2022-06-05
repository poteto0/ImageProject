import cv2
import numpy as np


class imageProcessing:
    def __init__(self, dir_name, thres1=0, thres2=225, alpha=1.0, beta=0.0):
        self.thres1 = thres1
        self.thres2 = thres2
        self.alpha = alpha
        self.beta = beta
        self.dir_name = dir_name
    
    # 前処理の実行 #
    def doProcessing(self, img, filename="null"):
        img = self.adjust(img, filename)
        img = self.gaussian(img) # ガウシアンフィルタ
        img = self.binarization(img, filename) # 二値化
        img = self.morphology(img, filename) # モルフォロジー変換
        
        return img
    
    # 二値化 #
    def binarization(self, img, filename="null"):
        #--- 画像の二値化 ---#
        # グレースケールに変換する。
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # 2値化する
        ret, img = cv2.threshold(img,self.thres1,self.thres2,cv2.THRESH_BINARY_INV)
        #img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,\
            #cv2.THRESH_BINARY,11,2)
        #ret, img = cv2.threshold(img, self.thres1, self.thres2, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
        img = cv2.bitwise_not(img)
        
        #--- 保存 ---#
        #cv2.imwrite(f"gray/{filename}", img) # グレースケール
        #cv2.imwrite(f"bin/{self.dir_name}/{filename}", img)
        
        return img #二値化を返却
    
    # モルフォロジー変換 #
    def morphology(self, img, filename="null"):
        #--- モルフォロジー変換 ---#
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3)) # カーネルサイズ
        #img = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel, iterations=1)
        img = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel, iterations=2)
        img = cv2.dilate(img,kernel,iterations = 5)
        
        #--- 保存 ---#
        cv2.imwrite(f"morphology/{self.dir_name}/{filename}", img)
        
        return img
    
    # コントラスト調整 #
    def adjust(self, img, filename="null"):
        
        # 明るさβを足してコントラストαをつける
        img = self.alpha * img + self.beta
        
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        #--- 保存 ---#
        cv2.imwrite(f"adjust/{self.dir_name}/{filename}", img)
        
        return img
    
    # ガウシアンフィルタ #
    def gaussian(self, img):
        
        kernel = np.ones((5,5),np.float32)/25
        img = cv2.filter2D(img,-1,kernel)
        
        return img
    