# ImageProject

## 前処理

### モロフォジー変換
Kernel x*x枠でiteration飛びごとに画像を操作する。Kernel内に画素値が1以上のものがあればKernel内の画素値を1とする。これによって収縮(Erosion)、膨張(Dilation)またはノイズ除去であるオープニング(Opening)、クロージング(Closing)を行う。

参考
・http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_imgproc/py_morphological_ops/py_morphological_ops.html
