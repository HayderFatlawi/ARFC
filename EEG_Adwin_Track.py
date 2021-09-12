import pandas as pd 

import numpy as np
from skmultiflow.drift_detection.adwin import ADWIN
adwin = ADWIN()

data_stream = pd.read_csv("/home/ubuntu/class.csv").to_numpy()

m=110640
count=0
count2=0
for i in range(m):
  adwin.add_element(np.float64(data_stream[i]))

  if adwin.detected_change():
      print('Change detected in data at index: ' + str(i)+' ; window size : '  + str(adwin.width))
      count+=1
      count2+=adwin.width

print(count)
if (count==0):
    print(adwin.width)
else:
    count2=count2/count

    print(count2)
