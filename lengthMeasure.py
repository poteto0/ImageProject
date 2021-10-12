import cv2
import imageProcessing as ip
import numpy as np
import math
import widthScanner
import rotateImage as rimg

def lengthMeasure(img, filename):
    """--- 画像の前処理 ---"""
    bin_img = ip.binarization(img, filename) # 二値化
    mor_img = ip.morphology(bin_img, filename) # モルフォロジー変換
    msk_img = cv2.merge((mor_img, mor_img, mor_img)) # マスクのチャンネル数を3に
    res = cv2.bitwise_and(img,msk_img)
    
    """　全てのオブジェクトを抽出 """
    # ラベリング #
    # 画像内の連結領域を抽出する(ラベリング)
    nLabels, labelImages, data, center = cv2.connectedComponentsWithStats(mor_img) 
    # 連結部分を1つずつ抽出 #
    for i in range(1,nLabels,1): # ラベル数を取得 ＊背景である0を除く
        target_lb_id = i # ターゲットID
        tobjx = data[target_lb_id, 0] # (対象の)x座標
        tobjy = data[target_lb_id, 1] # (対象の)y座標
        tobjw = data[target_lb_id, 2] # (対象の)幅
        tobjh = data[target_lb_id, 3] # (対象の)長さ(高さ)
        h, w = mor_img.shape[:2] # (画像の)大きさ
        btarget = np.zeros((h, w), np.uint8) # ターゲット描画用に空画像を生成
        btarget[labelImages == target_lb_id] = 255 # ターゲットを取得する
    
        
        """ 回帰計算 + 長さの測定 """
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
                
            res[int(v),x,:] = (255,255,0)   # 曲線描画
            # 線の描画 #   
            cv2.line(res, (int(x), int(v)), (int(bx), int(by)), (255,255,255), 2)
            
        # 長さを左隅に記載 #
        cv2.putText(res, str('{:.1f}'.format(tot_len)), (int(tobjx), int(tobjy)), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255,255,255), 1,cv2.LINE_AA)
        
        # ターゲットオブジェクト認識領域描画 #
        cv2.rectangle(res,(tobjx,tobjy),(tobjx+tobjw,tobjy+tobjh),(0,255,255),1)
        
        """ 幅の測定 """
        # resをグレースケールに(チャンネル数を減らす) #
        bres = cv2.cvtColor(res, cv2.COLOR_BGR2GRAY)
        # 幅を測る
        offsetx = 10
        incx = 16 # 幅を測るステップ幅
        cary = np.zeros((incx,), np.float32)
        carx = np.zeros((incx,), np.float32)
        lbno = 1 # 幅を測った場所のID
        for sx in range(tobjx+incx, tobjx+tobjw-incx, incx):
            cary = np.zeros((incx,), np.uint8)
            for x in range(sx, sx+incx): # 横方向にオブジェクトの座標を取得
                vy = expr(x)
                carx[sx-x] = x
                cary[sx-x] = vy
            rv1 = np.polyfit(carx ,cary,1) # 1次回帰
            expr1 = np.poly1d(rv1)
    
            mx = sx + incx // 2 # x座標の平均
            my = expr(mx) # その時のy座標(1次回帰から)
    
            # 直行を求める #
            ex = widthScanner.WidthScanner(rv1, mx, my)
            spt, ept, dist = ex.getPoints(bres) # 座標と長さを取得
            # 幅の描画 #
            cv2.line(res, (int(spt[0]), int(spt[1])), (int(ept[0]), int(ept[1])), (255,0,0), 1, lineType=cv2.LINE_AA)
            cv2.putText(res, str('{:.1f}'.format(dist)), (int(spt[0]-6), int(spt[1])-8), cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0,255,255), 1,cv2.LINE_AA)
            
            lbno += 1
        
    # 画像保存 #
    cv2.imwrite(f'test/{filename}', res)
