import csv
import numpy as np
from numpy import *
import os



file_exists = os.path.isfile('time.csv')
x = arange(0, 240, 0.03)
x = np.ndarray.tolist(x)
x = np.reshape(x, (len(x), 1))


with open('animation2.csv', 'a') as csvfile1:
    with open('learn.csv', 'rb') as csvfile:
        learn = csv.reader(csvfile, delimiter=',', quotechar='|')

        wr = csv.writer(csvfile1, delimiter='\t', quotechar=' ', quoting=csv.QUOTE_MINIMAL)
        for row in learn:
            wr.writerow(row)
