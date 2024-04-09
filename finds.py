import pandas as pd
import numpy as np
data=pd.read_csv("ML_exam\enjoysport.csv")
hypothesis=['%' for _ in range(len(data.columns)-1)] 
positive_examples=data[data['EnjoySport']=='Yes'].iloc[:,:-1].values.tolist()
for example in positive_examples:
    for i in range(len(example)):
        if(hypothesis[i]!='%' and hypothesis[i]!=example[i]):
            hypothesis[i]='?'
        else:
            hypothesis[i]=example[i]
print("final conclusion of find_s algorithm is :")
print(hypothesis)