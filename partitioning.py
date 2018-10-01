from datetime import date , datetime, timedelta, time
import pandas as pd
import csv
from scipy.stats import norm as normal
from statsmodels.robust.scale import mad
import numpy as np
import glob
import os
from multiprocessing import Pool

def red_and_combine_all_csv_file(input_location, output_location):    
    all_data = []
    for file_name in glob.glob(input_location+'*.csv'):
        df = pd.read_csv(file_name)
        all_data.append(df)
    final_data = pd.concat(all_data)
    final_data.to_csv(os.path.join(output_location, "final_data.csv"), index =None)

def get_distance(point_1, point_2, mapping):
    p1 = mapping[point_1]
    p2 = mapping[point_2]
    return np.sqrt(((p1["latitude"]-p2["latitude"])* LATITUDE_INTERVAL_LEN)**2 + 
                   ((p1["longitude"]-p2["longitude"])*  LONGITUDE_INTERVAL_LEN)**2)
def qcut_modified(x, q):
    try:
        return pd.qcut(x,q,labels=False)
    except ValueError as e:
        print "{0} group had an exception {1}".format(x.name, e)
        return [pd.np.NaN]*len(x)

def cut_modified(x,q, use_mad_for_std=True):
    try:
        quantiles_in_sigmas = np.asarray(map(normal.ppf, q))
        x_clean = x.dropna()
        mean = np.mean(x_clean)
        std = np.std(x_clean) if not use_mad_for_std else mad(x_clean)
        bins = mean + quantiles_in_sigmas*std
        bins = np.sort(np.append(bins, (x_clean.min()-1E-6, x_clean.max()+1E-6)))
        return pd.cut(x, bins, labels=range(len(bins)-1))
    except ValueError as e:
        #print len(x_clean)
        #print "{0} group had an exception {1}".format(x.name, e)
        return [pd.np.NaN]*len(x)


def add_digitized_columns_given_input_filter(df_orig, columns_list, cut_point_list, based_on_filter, quantile_cut_point_format= True,
                                             digitized_columns_names=[]):
    df = df_orig.copy()
    filter_name = '*'.join(['_Digital'] +based_on_filter)
    if not digitized_columns_names:
        digitized_columns_names = map(lambda x: x + filter_name, columns_list)
    print digitized_columns_names
    if not based_on_filter:
        if quantile_cut_point_format:
            for k,col in enumerate(columns_list):
                df[digitized_columns_names[k]] = cut_modified(df[col], cut_point_list )
        else:
            for k,col in enumerate(columns_list):
                df[digitized_columns_names[k]] = pd.cut(df[col], cut_point_list, labels =range(len(cut_point_list)-1))

    else:
        df_groups = df.groupby(based_on_filter)
        if quantile_cut_point_format:
            for k,col in enumerate(columns_list):
                #df[digitized_columns_names[k]] = df_groups[col].transform(lambda x: pd.qcut(x,cut_point_list,
                #labels =digital_vals))
                df[digitized_columns_names[k]] = df_groups[col].transform(lambda x: cut_modified(x,cut_point_list))
        else:
            for k,col in enumerate(columns_list):
                df[digitized_columns_names[k]] = df_groups[col].transform(lambda x: pd.cut(x,cut_point_list,
                                                                                           labels =range(len(cut_point_list)-1)))
    return df, digitized_columns_names


def use_digitize(df_org, **args):
    df = df_org.copy(deep=True)
    #mapping_dict = {key : {k: (eval(key + "_bins")[k] + eval(key + "_bins")[k+1])/2.0
    #                      for k in range(len(eval(key + "_bins"))-1)} for key in ["longitude","latitude"]}



    for col in args:
        df, added_col = add_digitized_columns_given_input_filter(df, [col[:-5]], args[col], [],False)
        #df[col +"_center"] = df[col].apply(lambda x: mapping_dict[col][x])
    return df.dropna()









def calc(obj):
    res = obj.calculate()
    return res

class SingleCalibration(object):
    def __init__(self, date):
        self.date= date
        
    def calculate(self):
        calibrate_for_one_date(self.date)
        print self.date



def calibrate_for_one_date(j):
    taxi_file_name = '2016-01-'+'{num:02d}'.format(num=j)
    grid=pd.read_csv("C:/Users/Admin/Desktop/proposal/nyc_grid_all.csv")
    #grid=pd.read_csv("C:/Users/Admin/Desktop/proposal/nyc_grid_all.csv")
    A=['00','01','02','03','04','05','06','07','08','09','10','11','12','13','14','15','16','17','18','19','20','21','22','23']
    top=grid['top'].unique()  #28
    left=grid['left'].unique()  #18
    right=grid['right'].unique()
    bottom=grid['bottom'].unique()
    
    latitude_bins = list(np.sort(list(set(top).union(set(bottom)))))
    longitude_bins = list(np.sort(list(set(left).union(set(right)))))
    G=pd.DataFrame()
    D=pd.DataFrame()
    Daily=pd.DataFrame()      
    LATITUDE_INTERVAL_LEN = 0.53
    LONGITUDE_INTERVAL_LEN = 0.42
    
    for i in A:
        taxi = pd.read_csv("C:/Users/Admin/Desktop/proposal/Data_2016/Output/{0}/{0}~{1}.csv".format(taxi_file_name,i))
        
        df_digitized = use_digitize(taxi,pickup_longitude_bins = longitude_bins,
                                    pickup_latitude_bins= latitude_bins,
                                    dropoff_longitude_bins = longitude_bins,
                                    dropoff_latitude_bins= latitude_bins)
        df_digitized["pickup_longitude_Digital"]=df_digitized["pickup_longitude_Digital"].astype(int)
        df_digitized["pickup_latitude_Digital"]=df_digitized["pickup_latitude_Digital"].astype(int)
        df_digitized["dropoff_longitude_Digital"]=df_digitized["dropoff_longitude_Digital"].astype(int)
        df_digitized["dropoff_latitude_Digital"]=df_digitized["dropoff_latitude_Digital"].astype(int) 
        
        for case in ["dropoff","pickup"]:
            df_digitized[case + "_bin"] = df_digitized[case + "_longitude_Digital"].astype(int) *(len(latitude_bins)-1)+ df_digitized[case + "_latitude_Digital"].astype(int)
        df_digitized.to_csv("C:/Users/Admin/Desktop/proposal/Data_2016/Output/{0}/{0}~{1}_digital.csv".format(taxi_file_name,i) , index=None)
        G=G.append(df_digitized)
       
        #a=df_digitized.pivot_table(columns='pickup_bin',aggfunc='count',index='dropoff_bin',values='passenger_count')
        b=df_digitized.groupby(by=['pickup_bin','dropoff_bin']).count()
        Pickup_to_dropoff_count=b['Unnamed: 0']
        Pickup_to_dropoff_count.to_csv("C:/Users/Admin/Desktop/proposal/Data_2016/Count/pick up to dropoff trips - 2016/P_to_d_count_{0}~{1}.csv".format(taxi_file_name,i),header=True)
        D=D.append(Pickup_to_dropoff_count)
        
        a=df_digitized.groupby(['pickup_bin']).count().rename(columns = {"Unnamed: 0" : i})
        a = a.reindex(pd.Series(pd.np.arange(4000)))
        Pickup_sum=a[i]
        Pickup_sum.to_csv("C:/Users/Admin/Desktop/proposal/Data_2016/Count/pickup count - 2016/Pickup_sum_"+taxi_file_name+'~'+i+'.csv',header=True)
        Daily=Daily.append(Pickup_sum)
    #H=D.groupby(by=['pickup_bin'])
    #H.to_csv("C:/Users/Admin/Desktop/proposal/Data_2016/Count/daily pickup to dropoff/Daily_Distribution"+taxi_file_name+'.csv'
                 #,header=True, index= None)        
    date_time_cols = ["date","time"]
    date_time_cols +=list(Daily.columns)
        
    Daily = Daily.reset_index().rename(columns ={"index" : "time"})
    Daily["date"] = taxi_file_name    
    Daily[date_time_cols].to_csv("C:/Users/Admin/Desktop/proposal/Data_2016/Count/Daily_count - 2016/Daily_"+taxi_file_name+'count.csv'
                                 ,header=True, index= None)     
    print 'DONE'
if __name__ == '__main__':
 
    p = list(np.arange(1,32))
    #date = p[7]
    #sc = SingleCalibration(date)
    #sc.calculate()
    
    bb = [SingleCalibration(date) for date in p]
    pool= Pool(processes=8)
    results = pool.map(calc, bb)
                       
    red_and_combine_all_csv_file("C:/Users/Admin/Desktop/proposal/Data_2016/Count/Daily_count - 2016/",
                                 "C:/Users/Admin/Desktop/proposal/Data_2016/Count/Entire_Data/")                          

''''
#distance calculation
#mapping = {key : {"longitude": key/len(top), "latitude":key%len(top)} for key in range(len(top)* len(left)-1)}

#distance_df = pd.DataFrame(index = mapping.keys(),columns = mapping.keys())
#distance_df = distance_df.reset_index()
#columns = list(distance_df.columns)[1:]
#for col in columns:
    #distance_df[col] = distance_df["index"].apply(lambda point : get_distance(point, col, mapping))
'''''
