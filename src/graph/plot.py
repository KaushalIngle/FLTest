# importing the required module
import matplotlib.pyplot as plt
import pyqtgraph as pg
import numpy as np

# # x axis values
# x = [1,2,3]
# # corresponding y axis values
# y = [2,4,1]
  
# # plotting the points 
# plt.plot(x, y)
  
# # naming the x axis
# plt.xlabel('x - axis')
# # naming the y axis
# plt.ylabel('y - axis')
  
# # giving a title to my graph
# plt.title('My first graph!')
  
# # function to show the plot
# plt.show()

def graph_results(plot_name, y_label):
    x_label = [item for item in range(1, len(y_label)+1)]
    plt.plot(x_label, y_label)
    # x = np.random.normal(size=1000)
    # y = np.random.normal(size=1000)
    pg.plot()
    pg.plot(x_label, y_label, pen=None, symbol='o')  
    # naming the x axis
    plt.xlabel('epochs')
    # naming the y axis
    plt.ylabel(plot_name)
    
    # giving a title to my graph
    plt.title(plot_name)
    
    # function to show the plot
    plt.show()