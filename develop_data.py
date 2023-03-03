import pandas as pd

dir_name = r'C:/Dinesh/SEM-VI/RBL/Mental_Health_Portal/data/'
    
df_train = pd.read_csv(dir_name+'raw/labels/train_split_Depression_AVEC2017.csv')

df_test = pd.read_csv(dir_name+'raw/labels/dev_split_Depression_AVEC2017.csv')

df_dev = pd.concat([df_train, df_test], axis=0)