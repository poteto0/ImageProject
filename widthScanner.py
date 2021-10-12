import math

""" 幅を測る """
class WidthScanner:
    def __init__(self, coef, mx, my):
        self._imx = int(mx)
        self._imy = int(my)
        self.delta = -1 / coef[0]
        self.b = my - mx * self.delta

    def getX(self, y):
        x = (y - self.b) / self.delta
        return x

    # 座標と長さを取得 #
    def getPoints(self, binimg):
        h,w = binimg.shape
        # 中央線より上をスキャン #
        x1 = 0
        y1 = 0
        for y in range(self._imy, 0, -1):
            x = int(self.getX(y))
            if binimg[y,x] == 0:
                break
            else:
                x1 = x
                y1 = y

        # 中央線より下をスキャン #
        x2 = 0
        y2 = 0
        for y in range(self._imy, h, 1):
            x = int(self.getX(y))
            if binimg[y,x] == 0:
                break
            else:
                x2 = x
                y2 = y
        dist = math.sqrt((x1-x2)**2+(y1-y2)**2) # 長さ計算
        return (x1,y1), (x2, y2), dist