import numpy as np
import os
import pickle

load_dir = os.path.join(os.getcwd(), 'stock_data')
save_dir = os.path.join(os.getcwd(), 'pkl_data')


for filename in os.listdir(load_dir):
   if filename.endswith(".csv"): 
      path_to_load = os.path.join(load_dir, filename)
      print(path_to_load)
      path_to_save = os.path.join(save_dir, filename[:-4]+'.pkl')
      temp =  np.loadtxt(path_to_load, skiprows=1, usecols=[1,2,3,4,6], delimiter=',')
      for column in range(temp.shape[1]):
         temp[:,column] = temp[:,column] - temp[:,column].mean()
         temp[:,column] = temp[:,column] / np.absolute(temp[:,column]).max()
      pickle.dump(temp, open(path_to_save,"wb"))
