#  import pandas as pd 
#  import matplotlib.pyplot as plt 
#  
#  data = pd.read_csv('results.csv')
#  
#  plt.figure(figsize=(10, 6))
#  plt.plot(data['t'], data['time'], marker='o')
#  plt.xlabel('threads')
#  plt.ylabel('Time (milli seconds)')
#  plt.title('Threading Analysis of task3')
#  plt.grid(True)

import matplotlib.pyplot as plt
import pandas as pd

# Load data from CSV files
data1 = pd.read_csv('results.csv', header=None)
data2 = pd.read_csv('results1.csv', header=None)

# Extract y-values from the first column of each dataset
y_values1 = data1[0].values
y_values2 = data2[0].values

# Define x-values (2^10 to 2^29)
x_values = [2 ** i for i in range(10, 30)]

# Plot data from both files
plt.figure(figsize=(10, 6))
plt.plot(x_values, y_values1, marker='o', color='b', linestyle='-',label='512 threads')
plt.plot(x_values, y_values2, marker='s', color='r', linestyle='--',label='16 threads')

# Set log scale for x-axis
plt.xscale('log', base=2)

# Label axes and add title
plt.xlabel('Number of elements (2^n)', fontsize=12)
plt.ylabel('Time taken by vscale function', fontsize=12)
plt.title('Plot of time taken by vscale by using  CUDA threads (512 and 16 threads per block)', fontsize=14)

# Show legend and grid
plt.legend()
plt.grid(True, which='both', linestyle='--', linewidth=0.5)

plt.savefig('/filespace/k/kundu8/Desktop/repo759/HW05/task3.pdf')
