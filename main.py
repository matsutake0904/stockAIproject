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
from my_util.util import loadHist, saveHist, saveModel
import numpy as np

# Press the green button in the gutter to run the script.
if __name__ == '__main__':
    modName="DNN02"
    num=6502
    date_begin=datetime.date(2018, 1, 1)
    date_end=datetime.date(2020, 12, 30)
    path="/Users/ryo/Documents/StockData/"+str(num)+"_"+str(date_begin)+"to"+str(date_end)+".csv"
    path_hist = "/Users/ryo/Work/python_work/myAI_data/"+ modName+ "_" + datetime.datetime.now().strftime("%Y-%m-%d")+".json"
    path_model = "/Users/ryo/Work/python_work/myAI_data/"+ modName+ "_" + datetime.datetime.now().strftime("%Y-%m-%d")
    path_predict=""
    # is_success, data = get_stock_day(num, date_begin, date_end)
    is_success = True
    jj=100
    data = np.loadtxt(path, delimiter=",", dtype="int64, U20, float, float, float, float,int64")
    # data = np.loadtxt(path, delimiter=",", dtype=["int64", "str", "float", "float", "float", "float", "int64"])

    print('File loading is ' + str(is_success))
    print('Loaded data is ' + str(data))
    if is_success:
       print('Date range is ' + str(data[1][0]) + " to " + str(data[1][-1]) )
       print('Data size is ' + str(len(data[:][0])))
       data_np=np.zeros([len(data),len(data[0])-2])
       for i in range(len(data)):
           for j in range(len(data[1])-2):
               data_np[i,j] = data[i][j+2]

       print(data_np[:int(len(data_np)/2),0])
       # model = LSTM_model()
       # model = DNN01_model()
       # model = DNN02_model()
       # model = DNN03_model()
       model = DNN02_multi_model()
       boo, history = model.train_model(data_np[:-150])
       saveHist(path_hist, model.history)
       saveModel(path_model, model.model)
       success_ai, predicted = model.predict(data_np[-150:], path_predict)
       print("predicted", predicted)
    # See PyCharm help at https://www.jetbrains.com/help/pycharm/
