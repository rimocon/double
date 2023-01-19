# dp_sympy
2リンクマニピュレータの分岐計算プログラム用に使えそうなパッケージを色々と作っています.

個人用に作っているので非常にコードが汚いですがご了承ください

- dp.py(未完成):分岐計算プログラム
- pp.py(ほぼ完成):位相図プログラム
- animation.py(ほぼ完成):シミュレーションプログラム
- dynamical_system.py:システム構造
- input.json:入力ファイル
***

## Requirements
* python(3.9~)
* sympy
* numpy
* scipy
* matplotlib

## Usage

```python3 *.py in.json ```

```in.json:``` input files



## アニメーションプログラム使い方

このプログラムでは平衡点を初期値とし，軌道をアニメーション化しています．


- キーボードアクション
    - `p`: 変更するパラメータを変える
    - '&uarr;/&darr;': パラメータを増やす,また減らす
    - `&rarr;/&larr;': パラメータの変更量を変化
    - `space`: もう一度最初から描画，この時パラメータセットはそのままです
    - `x`: 平衡点を変更します．この系では4つの平衡点があります
    - `q`: quit

- マウスアクション
    - `left click`: &theta;1,&omega;1 をクリックした値にセット
    - `right click`: &theta;2,&omega;2 をクリックした値にセット

初期パラメータを変更したい場合は`in.json`の`params`を変更してください．

上から順にk1,k2,&tau;1,&tau;2で，グラフ上のp0~p3も同様です．

x00~x30は平衡点の値です．

