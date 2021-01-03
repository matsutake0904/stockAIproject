from my_util.util import movingAve
import numpy as np
from myAI.stock_DNN02_multidatum import DNN02_multi_model
from my_util.util import loadHist, saveHist, saveModel
import datetime

modName="DNN02"
num=6502
date_begin=datetime.date(2018, 1, 1)
date_end=datetime.date(2020, 12, 30)
path="/Users/ryo/Documents/StockData/"+str(num)+"_"+str(date_begin)+"to"+str(date_end)+".csv"
path_hist = "/Users/ryo/Work/python_work/myAI_data/"+ modName+ "_" + datetime.datetime.now().strftime("%Y-%m-%d")+".json"
path_model = "/Users/ryo/Work/python_work/myAI_data/"+ modName+ "_" + datetime.datetime.now().strftime("%Y-%m-%d")
path_predict=""
data = np.loadtxt(path, delimiter=",", dtype="int64, U20, float, float, float, float,int64")
data_np=np.zeros([len(data),len(data[0])-2])
for i in range(len(data)):
    for j in range(len(data[1])-2):
        data_np[i,j] = data[i][j+2]
print("data " + str(data_np))

data2=movingAve(data_np, 3)
print("data2 " + str(data2))
data3 = np.hstack([data_np,data2])
print(data3)
