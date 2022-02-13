# ハイブリッドロケットエンジンの燃焼シミュレーション

## はじめに
このリポジトリは，ハイブリッドロケットエンジンの燃焼シミュレーション作成用リポジトリです．

## 必要なライブラリ
- numpy
- pandas
- matplotlib
- Scipy

## プログラムの説明
- pe_iteration.py

燃焼室圧およびノズル出口圧を最適化します．
バージョン1をリリースしました．
計算時間は1.3秒です(M1 mac OSX, メモリ16 GB)．
Excel版との整合性を確認済みです．

- pulptest.py

PuLPを用いた線形最適化のテストです．

- scipyoptimizationtest.py

Scipyを用いた非線形最適化のテストです．