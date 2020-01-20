import pandas as pd

test_dataframe = pd.read_csv(r"C:\Users\muron\Desktop\Coding\VIsual Studio Code\Titanic Predictive Model\train.csv") # use r for raw string data
train_df = pd.read_csv(r"C:\Users\muron\Desktop\Coding\VIsual Studio Code\Titanic Predictive Model\train.csv")  # use r for raw string data
train_df.info()  # 891 n. of passengers
train_df.describe()  # shows that 38% fo the set survived!
