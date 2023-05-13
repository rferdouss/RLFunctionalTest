import matplotlib.pyplot as plt

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np


x1 =[1,2,3,4,5,6,7,8,9,10] # [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
y1 =[77,130,86,333,45, 150,102,41,16,55]# [176,62,38,4,21,12,7,19,4,3,5,15,9,3,4]
#penalty1 =[63,53,113,321,13,121,78,21,0,27]
# plotting the line 1 points
plt.plot(x1, y1, label = "Curiosity-RL")
# line 2 points
x2 =[1,2,3,4,5,6,7,8,9,10]#[1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
y2 = [116,31,56,35,165,80,58,53,51,30]#[81,13,23,17,34,38,63,33,89,53,53,52,28,19,17]
#penalty2 =[63,5,28,4,156,66,26,13,16,1]
# plotting the line 2 points
plt.plot(x2, y2, label = "Random")
x3 =[1,2,3,4,5,6,7,8,9,10]# [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15]
y3 = [71,147,225,45,39,71,150,126,133,192]#[41,25,19,26,37,58,5,8,11,33,109,11,7,8,6]
#penalty3=[63,5,28,4,156,66,26,13,16,1]
plt.plot(x3, y3, label = "Sparse reward RL")
plt.xlabel('Episodes')
# Set the y axis label of the current axis.
plt.ylabel('Steps')
# Set a title of the current axes.
plt.title('Training')
# show a legend on the plot
plt.legend()
# Display a figure.
plt.show()