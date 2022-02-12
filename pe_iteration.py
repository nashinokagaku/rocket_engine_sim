#! /usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from pulp import *

# 入力値
P_t_i = 3.3 		# 初期タンク圧 (MPa)
P_t_f = 2.0 		# 最終タンク圧 (MPa)
t_b_d = 1.36 		# 燃焼予定時間 (s)
V_ox_i = 300 		# 酸化剤充填量 (cc)
rho_ox = 895.67 	# 酸化剤密度 (kg/m^3)
gamma_ox = 0 		# 酸化剤ガス比熱比 (-)
D_o = 3.4 			# オリフィス径 (mm)
C_d = 0.665 		# 流量計数 (-)
rho_f = 910 		# 燃料密度 (kg/m^3)
L_f = 250 			# 燃料軸長 (mm)
D_f_i = 36 			# 初期ポート径 (mm)
D_f_out = 38 		# 燃料外径 (mm)
N_port = 1 			# ポート数 (個)
a = 1.04e-4 		# 酸化剤流速係数 (m^3/kg)
n = 0.352 			# 酸化剤流速指数 (-)
eta_c_star = 0.85 	# C*効率 (-)
P_o = 0.1013 		# 背圧 (MPa)
D_t_i = 13.0 		# 初期スロート径 (mm)
D_e = 22.2 			# ノズル出口径 (mm)
alpha = 15 			# ノズル開口半長角 (deg)
r_dot_n = 0			# エロージョン速度 (mm/s)

# シミュレーション設定
simulation_time = 5 						# シミュレーション時間 (s)
Ts = 10e-3									# サンプリング周期 (s)
t = np.linspace(0, simulation_time, int(simulation_time / Ts))
m_ox = np.zeros([len(t), 1])				# 酸化剤消費量 (Kg)
P_t = np.zeros([len(t), 1])					# タンク圧 (MPa)
P_c = np.zeros([len(t), 1])					# 燃焼室圧 (MPa)
m_dot_ox = np.zeros([len(t), 1])			# 酸化剤流量 (kg/s)
m_f = np.zeros([len(t), 1])					# 燃料消費量 (kg)
D_f = np.zeros([len(t), 1])					# ポート径 (mm)
G_ox = np.zeros([len(t), 1])				# 酸化剤流速 (kg/m^2s)
r_dot = np.zeros([len(t), 1])				# 燃料後退速度 (mm/s)
m_dot_f = np.zeros([len(t), 1])				# 燃料流量 (kg/s)
o_f = np.zeros([len(t), 1])					# OF比 (-)
eta_c_star_c_star = np.zeros([len(t), 1])	# 特性排気速度 (m/s)
gamma = np.zeros([len(t), 1])				# 比熱比 (-)
D_t = np.zeros([len(t), 1])					# スロート径 (mm)
P_e = np.zeros([len(t), 1])					# ノズル出口圧 (MPa)
C_f = np.zeros([len(t), 1])					# 推力係数 (-)
F_t = np.zeros([len(t), 1])					# 推力 (N)
I_t = np.zeros([len(t), 1])					# 力積 (Ns)
epsilon_d = np.zeros([len(t), 1])			# 開口比 (-)
diff_epsilon_d = np.zeros([len(t), 1])		# 開口比誤差 (%)
P_c_d = np.zeros([len(t), 1])				# 燃焼室圧 (MPa)
diff_P_c_d = np.zeros([len(t), 1])			# 燃焼室圧誤差 (%)

# 初期値設定


# C*csvと比熱比csvを読み込み
df_c_star = pd.read_csv('cstar.csv', index_col=0)
df_gamma = pd.read_csv('gamma.csv', index_col=0)

# シミュレーション
for k in range(len(t)):
	m = LpProblem() # 数理モデル
	x = LpVariable('P_c', lowBound=0) # 変数
	m += (x - (4 * eta_c_star_c_star[k] * (m_dot_ox[k] + m_dot_f[k]) / (math.pi * D_t[k]^2))) / (4 * eta_c_star_c_star[k] * (m_dot_ox[k] + m_dot_f[k]) / (math.pi * D_t[k]^2)) * 100 # 目的関数
	m +=  ==  # 制約条件
	m.solve() # ソルバーの実行
	P_c[k] = value(x)

=$C$19* # C*効率
(INDEX(OF!$C$4:$AV$199,MATCH(K28,OF!$B$4:$B$199,1),MATCH(D28,OF!$C$3:$AV$3,1)) # C*csv中で現在のOFと燃焼室圧にマッチするC*の値
+ (K28 - INDEX(OF!$B$4:$B$199,MATCH(K28,OF!$B$4:$B$199,1),1)) # 現在のOF - C*csvのOFリスト中で現在のOFに最も近いOFの値
/(INDEX(OF!$B$4:$B$199,MATCH(K28,OF!$B$4:$B$199,1)+1,1) - INDEX(OF!$B$4:$B$199,MATCH(K28,OF!$B$4:$B$199,1),1)) # C*csvのOFリスト中で現在のOFに最も近いOFから1行下の値 - C*csvのOFリスト中で現在のOFに最も近いOFの値 = 0.1
*(INDEX(OF!$C$4:$AV$199,MATCH(K28,OF!$B$4:$B$199,1)+1,MATCH(D28,OF!$C$3:$AV$3,1))  - INDEX(OF!$C$4:$AV$199,MATCH(K28,OF!$B$4:$B$199,1),MATCH(D28,OF!$C$3:$AV$3,1))) # C*csv中で現在のOFに最も近いOFから1行下のOFと燃焼室圧にマッチするC*の値 - C*csv中で現在のOFと燃焼室圧にマッチするC*の値
+ (D28 - INDEX(OF!$C$3:$AV$3,1,MATCH(D28,OF!$C$3:$AV$3))) # 現在の燃焼室圧 - C*csv燃焼室圧リスト中で現在の燃焼室圧に最も近い燃焼室圧の値
/(INDEX(OF!$C$3:$AV$3,1,MATCH(D28,OF!$C$3:$AV$3)+1) - INDEX(OF!$C$3:$AV$3,1,MATCH(D28,OF!$C$3:$AV$3))) # C*csvの燃焼室圧リスト中で現在の燃焼室圧に最も近い燃焼室圧から1列右の値 - C*csvの燃焼室圧リスト中で現在の燃焼室圧に最も近い燃焼室圧の値 = 0.1
*(INDEX(OF!$C$4:$AV$199,MATCH(K28,OF!$B$4:$B$199,1),MATCH(D28,OF!$C$3:$AV$3,1)+1)  - INDEX(OF!$C$4:$AV$199,MATCH(K28,OF!$B$4:$B$199,1),MATCH(D28,OF!$C$3:$AV$3,1)))) # # C*csv中で現在の燃焼室圧に最も近い燃焼室圧から1列右の燃焼室圧とOFにマッチするC*の値 - C*csv中で現在の燃焼室圧とOFにマッチするC*の値