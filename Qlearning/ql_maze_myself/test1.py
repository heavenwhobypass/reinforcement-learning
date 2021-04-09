import numpy as np
import pandas as pd

a = np.zeros((4,4))
a[1,2] = a[2,1] = -1
a[2,2] = 1
print(a)
print(a.shape[0])
data = pd.DataFrame(np.arange(16).reshape(4,4), columns=list("ABCD"))
tmp = data.loc[1,:]
xx = tmp[tmp==tmp.max()].index
print(np.random.choice(xx))
action_list = ['up','down', 'right', 'left']

b = ['A','B','C']
print(b)

countlist = []
for i in range(10):
    countlist.append(i)
    print(countlist)
#for i in range(100):
#    print(''.join(str(i) for i in b))

