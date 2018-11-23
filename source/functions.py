from numpy.core.fromnumeric import transpose

import numpy as np
x=[[2,-2,3,2],[3,7,1,5],[3,0,-5,6]];
def normalizeDataNN(array):
    m=transpose(array);
    res=np.zeros((len(array[0]),len(array)),dtype=float);
    for i in range (len(m)):
        maximum=max(m[i]);
        minimum=min(m[i]);
        for j in range(len(m[i])):
            res[i][j]=float(((m[i][j]-minimum)/float(maximum-minimum))-0.5);
    return transpose(res);

