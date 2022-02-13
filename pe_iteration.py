#! /usr/bin/env python
# coding: utf-8

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import math
from pulp import *
from scipy.optimize import minimize_scalar, shgo, minimize

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
m_ox = np.ones([len(t), 1])					# 酸化剤消費量 (Kg)
P_t = np.ones([len(t), 1])					# タンク圧 (MPa)
P_c = np.ones([len(t), 1])*0.5				# 燃焼室圧 (MPa)
m_dot_ox = np.ones([len(t), 1])				# 酸化剤流量 (kg/s)
m_f = np.ones([len(t), 1])					# 燃料消費量 (kg)
D_f = np.ones([len(t), 1])					# ポート径 (mm)
G_ox = np.ones([len(t), 1])					# 酸化剤流速 (kg/m^2s)
r_dot = np.ones([len(t), 1])				# 燃料後退速度 (mm/s)
m_dot_f = np.ones([len(t), 1])				# 燃料流量 (kg/s)
o_f = np.ones([len(t), 1])*0.5				# OF比 (-)
eta_c_star_c_star = np.ones([len(t), 1])	# 特性排気速度 (m/s)
gamma = np.ones([len(t), 1])				# 比熱比 (-)
D_t = np.ones([len(t), 1])					# スロート径 (mm)
P_e = np.ones([len(t), 1])					# ノズル出口圧 (MPa)
C_f = np.ones([len(t), 1])					# 推力係数 (-)
F_t = np.ones([len(t), 1])					# 推力 (N)
I_t = np.ones([len(t), 1])					# 力積 (Ns)
# epsilon_d = np.ones([len(t), 1])			# 開口比 (-)
# diff_epsilon_d = np.ones([len(t), 1])		# 開口比誤差 (%)
# P_c_d = np.ones([len(t), 1])				# 燃焼室圧 (MPa)
# diff_P_c_d = np.ones([len(t), 1])			# 燃焼室圧誤差 (%)

# C*csvと比熱比csvを読み込み
df_c_star = pd.read_csv('cstar.csv', header=0, index_col=0)
df_gamma = pd.read_csv('gamma.csv', header=0, index_col=0)

# シミュレーション (1ステップ目)
m_ox[0, 0] = 0
P_t[0, 0] = (P_t_f - P_t_i) * 0 / t_b_d + P_t_i
P_c[0, 0] = 0.5
if P_t[0, 0] > P_c[0, 0]:
	diff_P = P_t[0, 0] > P_c[0, 0]
else:
	diff_P = 0
if m_ox[0, 0] <= V_ox_i * rho_ox * 1e-6:
	m_dot_ox[0, 0] = C_d * (math.pi / 4 * (D_o * 1e-3)**2) * math.sqrt(2 * rho_ox * diff_P)
else:
	m_dot_ox[0, 0] = 0
m_f[0, 0] = 0
D_f[0, 0] = math.sqrt(4 * m_f[0, 0] / (N_port * math.pi * rho_f * L_f * 1e-3) + (D_f_i * 1e-3)**2 * 1e3)
G_ox[0, 0] = 4 * m_dot_ox[0, 0] / (N_port * math.pi *(D_f[0, 0] * 1e-3)**2)
r_dot[0, 0] = alpha * G_ox[0, 0]**n * 1e3
m_dot_f[0, 0] = L_f * 1e-3 * math.pi * D_f[0, 0] * 1e-3 * rho_f * r_dot[0, 0] * 1e-3 * N_port
o_f[0, 0] = m_dot_ox[0, 0] / m_dot_f[0, 0]
if np.round(o_f[0, 0]) < 0.5:
	index_o_f = 0.5
elif np.round(o_f[0, 0]) > 20:
	index_o_f = 20
if np.round(P_c[0, 0]) < 0.5:
	index_P_c = 0.5
elif np.round(P_c[0, 0]) > 5:
	index_P_c = 5
eta_c_star_c_star[0, 0] = eta_c_star * (df_c_star.at[index_o_f, str(index_P_c)] \
	+ (o_f[0, 0] - index_o_f) / 0.1 * (df_c_star.at[(index_o_f + 0.1), str(index_P_c)] - df_c_star.at[index_o_f, str(index_P_c)]) \
	+ (P_c[0, 0] - index_P_c) / 0.1 * (df_c_star.at[index_o_f, str(index_P_c + 0.1)] - df_c_star.at[index_o_f, str(index_P_c)]))
gamma[0, 0] = df_gamma.at[index_o_f, str(index_P_c)] \
	+ (o_f[0, 0] - index_o_f) / 0.1 * (df_gamma.at[(index_o_f + 0.1), str(index_P_c)] - df_gamma.at[index_o_f, str(index_P_c)]) \
	+ (P_c[0, 0] - index_P_c) / 0.1 * (df_gamma.at[index_o_f, str(index_P_c + 0.1)] - df_gamma.at[index_o_f, str(index_P_c)])
D_t[0, 0] = D_t_i
P_e[0, 0] = 0.1
C_f[0, 0] = math.sqrt(2 * gamma[0, 0]**2 / (gamma[0, 0] - 1) * ((2 / (gamma[0, 0] + 1))**((gamma[0, 0] + 1) / (gamma[0, 0] - 1))) * (1 - (P_e[0, 0] / P_c[0, 0])**((gamma[0, 0] - 1) / gamma[0, 0]))) \
	+ ((P_e[0, 0] - P_o) / P_c[0, 0]) * ((D_e**2) / D_t_i**2)
F_t[0, 0] = ((1 * math.cos(math.radians(alpha))) / 2) * C_f[0, 0] * P_c[0, 0] * (math.pi * D_t_i**2 / 4)
I_t[0, 0] = 0

m1 = LpProblem() # 数理モデル: 燃焼室圧最適化
x1 = LpVariable('P_c', lowBound=0) # 変数: 燃焼室圧
m1 += (x1 - (4 * eta_c_star_c_star[0, 0] * (m_dot_ox[0, 0] + m_dot_f[0, 0]) / (math.pi * D_t[0, 0]**2))) / (4 * eta_c_star_c_star[0, 0] * (m_dot_ox[0, 0] + m_dot_f[0, 0]) / (math.pi * D_t[0, 0]**2)) * 100 # 目的関数
m1.solve() # ソルバーの実行
P_c[0, 0] = value(x1)

m2 = LpProblem() # 数理モデル: ノズル出口圧最適化
x2 = LpVariable('P_e', lowBound=0) # 変数: ノズル出口圧
# m2 += ((((2 / (gamma[0, 0] + 1))**(1 / (gamma[0, 0] - 1))) * ((P_c[0, 0] / x2)**(1 / gamma[0, 0])) / math.sqrt((gamma[0, 0] + 1) / (gamma[0, 0] - 1) * (1 - (P_e[0, 0] / P_c[0, 0])**((gamma[0, 0] - 1) / gamma[0, 0])))) - (D_e**2 / D_t_i**2)) / (D_e**2 / D_t_i**2) * 100 # 目的関数

m2.solve() # ソルバーの実行
P_e[0, 0] = value(x2)

# # シミュレーション (2ステップ目)
# m_ox[1, 0] = 0
# m = LpProblem() # 数理モデル
# x = LpVariable('P_c', lowBound=0) # 変数
# m += (x - (4 * eta_c_star_c_star[1, 0] * (m_dot_ox[1, 0] + m_dot_f[1, 0]) / (math.pi * D_t[1, 0]**2))) / (4 * eta_c_star_c_star[1, 0] * (m_dot_ox[1, 0] + m_dot_f[1, 0]) / (math.pi * D_t[1, 0]**2)) * 100 # 目的関数
# m.solve() # ソルバーの実行
# P_c[1, 0] = value(x)

# # シミュレーション (3ステップ目以降)
# for k in range(2, len(t)):
	
# 	eta_c_star_c_star[k, 0] = eta_c_star * df_c_star.at[(np.round(o_f[k, 0], 1)), str(np.round(P_c[k, 0], 1))] \
# 		+ (o_f[k, 0] - np.round(o_f[k, 0], 1)) / 0.1 * (df_c_star.at[(np.round(o_f[k, 0], 1) + 0.1), str(np.round(P_c[k, 0], 1))] - df_c_star.at[(np.round(o_f[k, 0], 1)), str(np.round(P_c[k, 0], 1))]) \
# 		+ (P_c[k, 0] - np.round(P_c[k, 0], 1)) / 0.1 * (df_c_star.at[(np.round(o_f[k, 0], 1)), str(np.round(P_c[k, 0], 1) + 0.1)] - df_c_star.at[(np.round(o_f[k, 0], 1)), str(np.round(P_c[k, 0], 1))])
	
# 	m = LpProblem() # 数理モデル
# 	x = LpVariable('P_c', lowBound=0) # 変数
# 	m += (x - (4 * eta_c_star_c_star[k, 0] * (m_dot_ox[k, 0] + m_dot_f[k, 0]) / (math.pi * D_t[k, 0]**2))) / (4 * eta_c_star_c_star[k, 0] * (m_dot_ox[k, 0] + m_dot_f[k, 0]) / (math.pi * D_t[k, 0]**2)) * 100 # 目的関数
# 	m.solve() # ソルバーの実行
# 	P_c[k, 0] = value(x)
# print(P_c)

