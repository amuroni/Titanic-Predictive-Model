import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import style
import seaborn as sns

test_dataframe = pd.read_csv(r"C:\Users\muron\Desktop\Coding\VIsual Studio Code\Titanic Predictive Model\train.csv") # use r for raw string data
train_df = pd.read_csv(r"C:\Users\muron\Desktop\Coding\VIsual Studio Code\Titanic Predictive Model\train.csv")  # use r for raw string data
train_df.info()  # 891 n. of passengers
train_df.describe()  # shows that 38% fo the set survived!

train_df.head(10)  # analyze the first n rows of the df

# analyze missing data that could impact the ML

total = train_df.isnull().sum().sort_values(ascending=False)
percent_1 = train_df.isnull().sum()/train_df.isnull().count()*100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head

# age has 177 missing values
# Cabin has over 600 missing values..
# embarked has only 2 missing values, the rest has no missing values

# 1. analyzing by age and sex

survived = "survived"
not_survived = "not survived"

fig, axes = plt.subplots(nrows=1, ncols=2,figsize=(10, 4))

women = train_df[train_df["Sex"]=="female"]
men = train_df[train_df["Sex"] == "male"]

ax = sns.distplot(women[women['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[0], kde =False)
ax = sns.distplot(women[women['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[0], kde =False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived']==1].Age.dropna(), bins=18, label = survived, ax = axes[1], kde = False)
ax = sns.distplot(men[men['Survived']==0].Age.dropna(), bins=40, label = not_survived, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')
plt.show()
