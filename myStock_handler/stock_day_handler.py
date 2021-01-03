import psycopg2
import datetime
import logging
# STOCK MODEL
### stock_num: stock number and company name
# stock_num integer (primary key), comp_name varchar(50);

### stock_day: stock data of date
#  (num integer, date date, open float, high float, low float, close float, volumes integer);


def regist_day(num, date, open, high, low, close, volumes):
    get_data_query = 'SELECT count(*) FROM stock_day WHERE num=%(num)s AND date=%(date)s '
    # insert_query ='INSERT INTO stock_day (num, date, open, high, low, close, volumes) VALUES (%(num)s, %(date)s, %(open)s, $(high)s, %(low)s, %(close)s, $(volumes)s)'
    insert_query ='INSERT INTO stock_day (num, date, open, high, low, close, volumes) ' \
                  'VALUES (%(num)s, %(date)s, %(open)s, %(high)s, %(low)s, %(close)s, %(volumes)s)'
    DATABASE_DIR = 'postgres://ryo:maryo1994@localhost/stockmodel'
    conn = psycopg2.connect(DATABASE_DIR)
    cur = conn.cursor()
    try:
        data_exist = cur.execute(get_data_query, {'num':num, 'date':date})
        if cur.fetchone()[0] == 0:
            logging.info("Registration is begin")
            print("Registration is begin")
            cur.execute(insert_query, {'num':num, 'date':date, 'open':open, 'high':high, 'low':low, 'close':close, 'volumes':volumes})
            conn.commit()
            logging.info("Registration is success")
        else:
            print('The Data has already registered  ' + str(date))
    except Exception as e:
        logging.critical('critical error {}'.format(e))
        cur.execute('ROLLBACK')
        conn.close()
        return False
    conn.close()
    return True

def get_stock_day(num, date_begin, date_end):
    get_data_query = 'SELECT * FROM stock_day WHERE num = %(num)s AND date BETWEEN date %(begin)s AND date %(end)s ORDER BY date '
    DATABASE_DIR = 'postgres://ryo:maryo1994@localhost/stockmodel'
    conn = psycopg2.connect(DATABASE_DIR)
    cur = conn.cursor()
    try:
        cur.execute(get_data_query, {'num':num ,'begin':date_begin, 'end': date_end})
        print('query is success')
        counter = 0
        datalist = cur.fetchall()
        length = len(datalist)
        # data = [length, length]
        # for data in datalist:
        #     date[counter] = data[1]
        #     open[counter] = data[2]
        #     print(str(date[counter]) + " " + str(open[counter]))
        return True, datalist
    except Exception as e:
        logging.critical('CRITICAL {}'.format(e))






