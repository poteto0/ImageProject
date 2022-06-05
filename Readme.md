# これは何？シアノバクテリアの長さを測るためのプロジェクト## ダウンロード/使い方* ダウンロードターミナルで作りたい場所に移動して<br>`$ git clone https://github.com/poteto0/ImageProject.git`<br>を実行<br>* 必要なライブラリを揃える<br>`$ pip install -r requirements.txt`<br>or<br>`$ brew install -r requirements.txt`<br>など<br>(注)不十分な可能性があります<br>* 最初の実行<br>mkdir.pyを最初に実行してください ## 現状* 単個体<br>=> 確認中* 集団<br>=> TODO細線化?## 長さ測定シアノバクテリアは曲線構造を持つので、OpenCVで輪郭抽出した後に三次曲線回帰をして長さを測定しています。## 参考* OpenCVチュートリアル<br><http://labs.eecs.tottori-u.ac.jp/sd/Member/oyamada/OpenCV/html/py_tutorials/py_tutorials.html>