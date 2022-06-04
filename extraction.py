import cv2
import numpy as np
import math

def extract(new_img, img, filename):
    # maskの作成
    msk_img = cv2.merge((new_img, new_img, new_img)) # マスクのチャンネル数を3に
    res = cv2.bitwise_and(img, msk_img)
    
    """　全てのオブジェクトを抽出 """
    # ラベリング #
    # 画像内の連結領域を抽出する(ラベリング)
    nLabels, labelImages, data, center = cv2.connectedComponentsWithStats(new_img) 
    # 連結部分を1つずつ抽出 #
    for i in range(1,nLabels,1): # ラベル数を取得 ＊背景である0を除く
        target_lb_id = i # ターゲットID
        tobjx = data[target_lb_id, 0] # (対象の)x座標
        tobjy = data[target_lb_id, 1] # (対象の)y座標
        tobjw = data[target_lb_id, 2] # (対象の)幅
        tobjh = data[target_lb_id, 3] # (対象の)長さ(高さ)
        h, w = img.shape[:2] # (画像の)大きさ
        btarget = np.zeros((h, w), np.uint8) # ターゲット描画用に空画像を生成
        btarget[labelImages == target_lb_id] = 255 # ターゲットを取得する
    
        tot_len = 0
        by = 0
        ypts, xpts = np.where(btarget == 255) # オブジェクトのある座標を取得
        rv = np.polyfit(xpts ,ypts, 3) # 3次回帰
        expr = np.poly1d(rv)
        for x in range(tobjx, tobjx+tobjw):
            v = expr(x)
            if(v > len(labelImages)-1): # 最大値処理
                v = len(labelImages)-1
            # 長さの測定 #
            if x == tobjx:
                by = v
                bx = x
            elif  x != tobjx:
                sdist = math.sqrt((bx-x)**2 + (by-v)**2)
                tot_len += sdist
                by = v
                bx = x
                
            #res[int(v),x,:] = (255,255,0)   # 曲線描画
        
        cv2.putText(res, f"tot_len:{tot_len}",(int(tobjx), int(tobjy)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1,cv2.LINE_AA)
        
            
    # 画像保存 #
    cv2.imwrite(f'extraction_images/{filename}', res)
    
    return res