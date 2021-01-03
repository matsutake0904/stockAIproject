# This is a sample Python script.

# Press ⌃R to execute it or replace it with your code.
# Press Double ⇧ to search everywhere for classes, files, tool windows, actions, and settings.
from myStock_handler.stock_day_handler import regist_day, get_stock_day
import datetime
from myAI.stock_litm import LSTM_model
from myAI.stock_DNN01 import DNN01_model
# from myAI.stock_DNN02 import DNN02_model
# from myAI.stock_DNN03 import DNN03_model
from myAI.stock_DNN02_multidatum import DNN02_multi_model
from myAI.stock_DNN04 import DNN04_model
from my_util.util import loadHist, saveHist, saveModel
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    modName="DNN02"
    num=6502
    interval = 5
    date_begin=datetime.date(2018, 1, 1)
    date_end=datetime.date(2020, 12, 30)
    path="/Users/ryo/Documents/StockData/"+str(num)+"_"+str(date_begin)+"to"+str(date_end)+".csv"
    path_minute="/Users/ryo/Documents/StockData_min"+ str(interval) +"/"+str(num)+"_"+str(date_begin)+"to"+str(date_end)+".csv"

    path_hist = "/Users/ryo/Work/python_work/myAI_data/"+ modName+ "_" + datetime.datetime.now().strftime("%Y-%m-%d")+".json"
    path_model = "/Users/ryo/Work/python_work/myAI_data/"+ modName+ "_" + datetime.datetime.now().strftime("%Y-%m-%d")
    path_predict=""
    # is_success, data = get_stock_day(num, date_begin, date_end)
    is_success = True
    jj=100

    ## day date
    data = np.loadtxt(path, delimiter=",", dtype="int64, U20, float, float, float, float,int64")

    ##Loading statistics data
    path_ma200="/Users/ryo/Documents/StockData/"+str(num)+"_"+str(date_begin)+"to"+str(date_end)+"_ma200.csv"
    path_ma75="/Users/ryo/Documents/StockData/"+str(num)+"_"+str(date_begin)+"to"+str(date_end)+"_ma75.csv"
    path_ma50="/Users/ryo/Documents/StockData/"+str(num)+"_"+str(date_begin)+"to"+str(date_end)+"_ma50.csv"
    path_ma25="/Users/ryo/Documents/StockData/"+str(num)+"_"+str(date_begin)+"to"+str(date_end)+"_ma25.csv"
    data_ma200 = np.loadtxt(path_ma200, delimiter=",", dtype="int64, U20, float, float, float, float,int64, U20")
    data_ma75 = np.loadtxt(path_ma75, delimiter=",", dtype="int64, U20, float, float, float, float,int64, U20")
    data_ma50 = np.loadtxt(path_ma50, delimiter=",", dtype="int64, U20, float, float, float, float,int64, U20")
    data_ma25 = np.loadtxt(path_ma25, delimiter=",", dtype="int64, U20, float, float, float, float,int64, U20")


    ## minute date
    # data_minute = np.loadtxt(path_minute, delimiter=",", dtype="int64, U30, float, float, float, float, int64, int64")
    # data = np.loadtxt(path, delimiter=",", dtype=["int64", "str", "float", "float", "float", "float", "int64"])

    print('File loading is ' + str(is_success))
    if is_success:
       print('Date range is ' + str(data[1][0]) + " to " + str(data[1][-1]))
       print('Data size is ' + str(len(data[:][0])))
       # data_np=np.zeros([len(data_minute),len(data_minute[0])-3])
       data_np=np.zeros([len(data),len(data[0])-2])
       data_np_ma200=np.zeros([len(data_ma200),len(data_ma200[0])-2])
       data_np_ma75=np.zeros([len(data_ma75),len(data_ma75[0])-2])
       data_np_ma50=np.zeros([len(data_ma50),len(data_ma50[0])-2])
       data_np_ma25=np.zeros([len(data_ma25),len(data_ma25[0])-2])
       # print('Date range is ' + str(data_minute[1][0]) + " to " + str(data_minute[1][-1]))
       # print('Data size is ' + str(len(data_minute[:][0])))
       # # data_np=np.zeros([len(data_minute),len(data_minute[0])-3])

       for i in range(len(data)):
           for j in range(len(data[1])-2):
               data_np[i,j] = data[i][j+2]
               data_np_ma200[i,j] = data_ma200[i][j+2]
               data_np_ma75[i,j] = data_ma75[i][j+2]
               data_np_ma50[i,j] = data_ma50[i][j+2]
               data_np_ma25[i,j] = data_ma25[i][j+2]


       print(data_np[:int(len(data_np)/2),0])
       # model = LSTM_model()
       # model = DNN01_model()
       # model = DNN02_model()
       # model = DNN03_model()
       model = DNN04_model()
       # model = DNN02_multi_model()
       boo, history = model.train_model(data_np[:-150])
       saveHist(path_hist, model.history)
       saveModel(path_model, model.model)
       success_ai, predicted = model.predict(data_np[-150:], path_predict)
       print("predicted", predicted)
    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
