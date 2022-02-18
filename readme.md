# Combustion simulations of hybrid rocket engines
## Introduction

This is a repository for creating combustion simulations of hybrid rocket engines.

## Required libraries

- numpy
- pandas
- matplotlib
- Scipy
- Jupter notebook

## Program description

-pe_iteration.py

Optimizes the combustion chamber pressure and nozzle exit pressure. Version 1 has been released. The calculation time is 1.3 seconds (M1 mac OSX, 16 GB memory). It has been checked for consistency with the Excel version.

- tb_iteration.py

Optimizes the burn time. Version 1 has been released. Calculation time is 14 seconds (M1 mac OSX, 16 GB memory). Checking the consistency with Excel version.

- enginesim.ipynb

A Jupyter notebook that can run pe_iteration.py and tb_iteration.py simultaneously.
<!-- ## はじめに
このリポジトリは，ハイブリッドロケットエンジンの燃焼シミュレーション作成用リポジトリです．

## 必要なライブラリ
- numpy
- pandas
- matplotlib
- Scipy
- Jupter notebook

## プログラムの説明
- pe_iteration.py

燃焼室圧およびノズル出口圧を最適化します．
バージョン1をリリースしました．
計算時間は1.3秒です(M1 mac OSX, メモリ16 GB)．
Excel版との整合性を確認済みです．

- tb_iteration.py

燃焼時間を最適化します．
バージョン1をリリースしました．
計算時間は14秒です(M1 mac OSX, メモリ16 GB)．
Excel版との整合性を確認中です．

- enginesim.ipynb

pe_iteration.pyとtb_iteration.pyを同時に実行できるJupyter notebookです． -->