import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('Statistics/structural_tokens_count.csv')
data['cumulative'] = data['count'].cumsum().values / data['count'].sum()
plt.plot(data['cumulative'])
plt.title('structural tokens')
plt.xlabel('token index')
plt.ylabel('cumulative fraction')
plt.show()

with open('Statistics/cell_content_tokens_count.tsv', 'r') as f:
    f.readline()
    data = f.readlines()
tokens = [x.strip('\n').split('\t')[0] for x in data]
counts = [int(x.strip('\n').split('\t')[1]) for x in data]
data = pd.DataFrame({'count': counts})
data['cumulative'] = data['count'].cumsum().values / data['count'].sum()
plt.plot(data['cumulative'])
plt.title('cell content tokens')
plt.xlabel('token index')
plt.ylabel('cumulative fraction')
plt.show()

data = pd.read_csv('Statistics/num_structural_tokens_count.csv')
data['cumulative'] = data['count'].cumsum().values / data['count'].sum()
x = []
cumulative = []
for n_rows in range(1, len(data)+1):
    subset = data.head(n_rows)
    x.append(subset['num structural tokens'].max())
    cumulative.append(subset['cumulative'].max())
plt.plot(x, cumulative)
plt.title('num structural tokens')
plt.xlabel('num structural tokens')
plt.ylabel('cumulative fraction of examples')
plt.show()

data = pd.read_csv('Statistics/num_cells_count.csv')
data['cumulative'] = data['count'].cumsum().values / data['count'].sum()
x = []
cumulative = []
for n_rows in range(1, len(data)+1):
    subset = data.head(n_rows)
    x.append(subset['num cells'].max())
    cumulative.append(subset['cumulative'].max())
plt.plot(x, cumulative)
plt.title('num cells')
plt.xlabel('num cells')
plt.ylabel('cumulative fraction of examples')
plt.show()

data = pd.read_csv('Statistics/max_num_cell_content_tokens_count.csv')
data['cumulative'] = data['count'].cumsum().values / data['count'].sum()
x = []
cumulative = []
for n_rows in range(1, len(data)+1):
    subset = data.head(n_rows)
    x.append(subset['max num tokens'].max())
    cumulative.append(subset['cumulative'].max())
plt.plot(x, cumulative)
plt.title('max num cell content tokens')
plt.xlabel('max num cell content tokens')
plt.ylabel('cumulative fraction of examples')
plt.show()
