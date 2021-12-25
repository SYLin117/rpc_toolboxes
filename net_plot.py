import matplotlib.pyplot as plt
import os
import pandas as pd

DATA_ROOT = f'C:\\Users\\ian\\Desktop\\ns3_test'
types = ['cwnd', 'flight', 'rto', 'rtt', 'rx', 'ssthress', 'tx']
type = 'ssthress'
tcp_versions = ['Tahoe', 'Reno', 'NewReno', 'Westwood']
# symbols = ['+', 'x', '_', '1']
# colors = ['blue', 'red', 'green', 'gray']
colors = ['#000099', '#FF0000', '#33CC33', '#CC33FF']

idx = 0
idx_x = 0
idx_y = 0
nrows = 4
ncols = 1
fig, axes = plt.subplots(nrows=nrows, ncols=ncols, figsize=(15, 10))
for y in range(nrows):
    for x in range(ncols):
        tcp = tcp_versions[idx]
        data = pd.read_csv(os.path.join(DATA_ROOT, 'Tcp{}_{}.data'.format(tcp, type)), sep=" ", names=['time', type])
        axes[idx].set_title(tcp, fontsize=20)
        axes[idx].set_ylim(0, 50000)
        axes[idx].set_xlim(0, 120)
        axes[idx].plot(data['time'], data[type], c=colors[idx], label=tcp, linewidth=3, alpha=0.5)
        # axes[y, x].title(tcp)
        axes[idx].grid()
        # fig.savefig('{}.png'.format(type))
        idx += 1
plt.show()
fig.savefig('{}.png'.format(type), dpi=200)
