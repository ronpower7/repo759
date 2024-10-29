import pandas as pd 
import matplotlib.pyplot as plt 

data = pd.read_csv('results.csv')

plt.figure(figsize=(10, 6))
plt.plot(data['t'], data['time'], marker='o')
plt.xlabel('threads')
plt.ylabel('Time (milli seconds)')
plt.title('Threading Analysis of task3')
plt.grid(True)
plt.savefig('/filespace/k/kundu8/Desktop/repo759/HW03/task3_t.pdf')
