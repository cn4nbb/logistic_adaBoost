import csv
import numpy as np
import random

def read_csv(csv_name):
	'''
	读取数据
	:param csv_name:文件路径
	:return: data_x:特征列表 res_y:标签列表
	'''

	with open(csv_name) as f:
		reader = csv.reader(f)
		rows = [row for row in reader]

	data_x = [] # 特征列表 150*4
	res_y = [] # 标签列表 150*1
	for item in rows:
		if item[-1] == 'label': # 去掉第一行
			continue
		jtem = [float(t) for t in item[:-2]]
		data_x.append(jtem) # 录入列特征值
		res_y.append(int(item[-1])) # 录入标签
	# for item in data_x:
	# 	print(item)
	# print(res_y)
	return data_x, res_y

# read_csv("iris_data.csv")

def train(x, y, w):
	'''
	弱分类器训练
	:param x: 特征列表 150*4
	:param y:  标签列表 150*1
	:param w: 样本权值
	:return: min_loss:最小误差e res_k:表现最好的弱分类器参数值
	'''
	min_loss = 1 # 定义一个最小损失
	m = x.shape[1] # 取特征列表的列数 也就是特征数
	res_k = None
	nums = 10 # 自定义循环次数 用于选取一个最好的弱分类器参数
	for t in range(nums):
		k_random = [] # 弱分类器的参数
		for i in range(m): # 随机取特征数个参数
			k_random.append(random.uniform(0, 1))
		k_random = np.array(k_random) # np化 方便计算
		mis = (k_random*x).sum(axis = 1) # k1*x1+k2*x2+k3*x3+k4*x4 mis:150*1
		#归一化处理
		mis = mis - min(mis) # 都减掉最小值 那么最小的那一行就是从0开始
		mis = mis / max(mis) # 都除以最大值 那么所有值都在[0,1]
		res = []
		# 进行分类
		for item in mis:
			if item <= 0.33:
				res.append(0)
			elif item <= 0.66:
				res.append(1)
			else:
				res.append(2)
		res = np.array(res)
		miss = sum((res != y)*w) #计算error 也就是e
		if miss < min_loss: # 保存最好的弱分类器参数
			min_loss = miss
			res_k = k_random
	return min_loss, res_k

def predict(x, res_k):
	'''
	:param x: 特征列表
	:param res_k: 传入弱分类器的参数
	:return: res:预测的分类结果
	'''
	mis = (res_k*x).sum(axis = 1)
	mis = mis - min(mis)
	mis = mis / max(mis)
	res = []
	for item in mis:
		if item <= 0.33:
			res.append(0)
		elif item <= 0.66:
			res.append(1)
		else:
			res.append(2)
	res = np.array(res)
	return res

def adaboost(csv_name):
	'''
	AdaBoost算法
	:param csv_name: 文件路径
	'''
	x, y = read_csv(csv_name) # 读取数据
	x = np.array(x)
	y = np.array(y)
	n = x.shape[0] # n为特征数
	M = 4 # 设置AdaBoost算法的迭代次数
	w_m = np.array([1/n]*n) # 初始化样本权重
	res = np.zeros(n) # 初始化预测分类结果
	for m in range(M): # 开始AdaBoost算法的迭代
		e_m, res_k = train(x, y, w_m) # 返回最好的弱分类器的e 以及参数
		a_m = 1/2 * np.log((1 - e_m)/e_m) # 根据公式计算当前弱分类器的权值alpha
		y_m = predict(x, res_k) # 得到预测的结果
		w_m = w_m * np.exp(-a_m*y*y_m) #先计算一部分新的w
		z_m = np.sum(w_m) # 根据公式计算z
		w_m = w_m/z_m # 最后得出新的w
		res += a_m*y_m # 得出最终的分类结果 就是各弱分类器的权值乘上其预测结果的和
	res = res - min(res) # 同样的归一会处理 为的是最后的分类
	res = res / max(res)
	result = []
	for item in res:
		if item <= 0.33:
			result.append(0)
		elif item <= 0.66:
			result.append(1)
		else:
			result.append(2)
	result = np.array(result)
	# print(result)
	percent = sum(result == y)/n
	print("正确率为{}%".format(percent*100))

if __name__ == '__main__':
	csv_name = "./data/iris_data.csv"
	adaboost(csv_name)