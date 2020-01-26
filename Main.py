import pandas as pd
from matplotlib import pyplot as plt
from matplotlib import style
import seaborn as sns
import re
import numpy as np

test_df = pd.read_csv(r"C:\Users\muron\PycharmProjects\Titanic-Predictive-Model\test.csv")  # use r for raw string data
train_df = pd.read_csv(r"C:\Users\muron\PycharmProjects\Titanic-Predictive-Model\train.csv")
train_df.info()  # 891 n. of passengers
train_df.describe()  # shows that 38% fo the set survived!

train_df.head(10)  # analyze the first n rows of the df

# analyze missing data that could impact the ML

total = train_df.isnull().sum().sort_values(ascending=False)
percent_1 = train_df.isnull().sum() / train_df.isnull().count() * 100
percent_2 = (round(percent_1, 1)).sort_values(ascending=False)
missing_data = pd.concat([total, percent_2], axis=1, keys=['Total', '%'])
missing_data.head()

# age has 177 missing values
# Cabin has over 600 missing values..
# embarked has only 2 missing values, the rest has no missing values

# 1. analyzing by age and sex

survived = "survived"
not_survived = "not survived"

fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(10, 4))

women = train_df[train_df["Sex"] == "female"]
men = train_df[train_df["Sex"] == "male"]

ax = sns.distplot(women[women['Survived'] == 1].Age.dropna(), bins=18, label=survived, ax=axes[0], kde=False)
ax = sns.distplot(women[women['Survived'] == 0].Age.dropna(), bins=40, label=not_survived, ax=axes[0], kde=False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['Survived'] == 1].Age.dropna(), bins=18, label=survived, ax=axes[1], kde=False)
ax = sns.distplot(men[men['Survived'] == 0].Age.dropna(), bins=40, label=not_survived, ax=axes[1], kde=False)
ax.legend()
ax.set_title('Male')
plt.show()

#  2. analyze correlation between Embarked and Pass class, survives and sex
#  by using a seaborn pointplot chart
FacetGrid = sns.FacetGrid(train_df, row="Embarked", height=5, aspect=2)
FacetGrid.map(sns.pointplot, "Pclass", "Survived", "Sex", palette=None, order=None, hue_order=None)
FacetGrid.add_legend()
plt.show()

# Embarked is correlated with survival dependind on gender
# Passenger class is also correlated with survival

# 3. Analyzing by Passenger class
sns.barplot(x="Pclass", y="Survived", data=train_df)
plt.show()

# higher survival chance for passengers in class 1!
# another plot to analyze this by correlating with passenger Age

grid = sns.FacetGrid(train_df, col="Survived", row="Pclass", height=2.2, aspect=1.6)
grid.map(plt.hist, "Age", alpha=.5, bins=20)
grid.add_legend()
plt.show()

# by correlating with age too, seems like Pclass 3 has high chance of not surviving

# 4. Data Processing

train_df = train_df.drop(["PassengerId"], axis=1)  # dropping pass ID, no use

# 4.1 dealing with missing data and Cabin, Embarked, Age data
# assigning deck number to each passenger, where decks go A-G

deck = {"A": 1, "B": 2, "C": 3, "D": 4, "E": 5, "F": 6, "G": 7, "U": 8, }

# where U stands for missing data
data = [train_df, test_df]

for dataset in data:
    dataset["Cabin"] = dataset["Cabin"].fillna("U0")
    dataset["Deck"] = dataset["Cabin"].map(lambda x: re.compile('([a-zA-Z]+)').search(x).group())
    dataset['Deck'] = dataset['Deck'].map(deck)
    dataset['Deck'] = dataset['Deck'].fillna(0)
    dataset['Deck'] = dataset['Deck'].astype(int)

# since we created a deck attribute, we can drop cabin
train_df = train_df.drop(["Cabin"], axis=1)
test_df = test_df.drop(["Cabin"], axis=1)
# train_df.head(10)
# test_df.head(10)

# then: randomize NaN age values based on mean age value

data = [train_df, test_df]

for dataset in data:
    mean = train_df["Age"].mean()
    std = test_df["Age"].std()
    is_null = dataset["Age"].isnull().sum()
    rand_age = np.random.randint(mean - std, mean + std, size=is_null)
    age_slice = dataset["Age"].copy()
    age_slice[np.isnan(age_slice)] = rand_age
    dataset["Age"] = age_slice
    dataset["Age"] = train_df["Age"].astype(int)

# if we check now the total of NaN values is 0

if print(train_df["Age"].isnull().sum()) == 0:
    print("No NaN values")
else:
    print(str(train_df["Age"].isnull().sum()) + " values")

# 4.2 fill embarked missing values with the most common one, only 2 to fill
common_value = train_df["Embarked"].describe()["top"]

data = [train_df, test_df]
for dataset in data:
    dataset["Embarked"] = dataset["Embarked"].fillna(common_value)

train_df.info()  # now has 891 non-null attributes

# 4.3 convert Fare in int format

data = [train_df, test_df]

for dataset in data:
    dataset["Fare"] = dataset["Fare"].fillna(0)  # fill NaN with 0
    dataset["Fare"] = dataset["Fare"].astype(int)

# 4.4 extract Titles from Name and convert into numbers

data = [train_df, test_df]
titles = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Rare": 5}

for dataset in data:
    dataset["Title"] = dataset.Name.str.extract('([A-Za-z]+)\.', expand=False)
    dataset["Title"] = dataset["Title"].replace(['Lady', 'Countess', 'Capt', 'Col', 'Don', 'Dr',
                                                 'Major', 'Rev', 'Sir', 'Jonkheer', 'Dona'], 'Rare')
    dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
    dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')
    dataset['Title'] = dataset['Title'].map(titles)
    dataset['Title'] = dataset['Title'].fillna(0)

# dropping Name since it's replaced with Title now
train_df = train_df.drop(['Name'], axis=1)
test_df = test_df.drop(['Name'], axis=1)

# 4.5 convert Sex into numeric data

genders = {"male": 0, "female": 1}
data = [train_df, test_df]

for dataset in data:
    dataset["Sex"] = dataset["Sex"].map(genders)

# 4.6 drop Ticket data since it has too many unique values
train_df = train_df.drop(["Ticket"], axis=1)
test_df = test_df.drop(["Ticket"], axis=1)

# 4.7 convert Embarked in numeric

ports = {"S": 0, "C": 1, "Q": 2}
data = [train_df, test_df]

for dataset in data:
    dataset['Embarked'] = dataset['Embarked'].map(ports)

# 4.7 create age groups paying attention to group distribution

data = [train_df, test_df]

for dataset in data:
    dataset['Age'] = dataset['Age'].astype(int)
    dataset.loc[dataset['Age'] <= 11, 'Age'] = 0
    dataset.loc[(dataset['Age'] > 11) & (dataset['Age'] <= 18), 'Age'] = 1
    dataset.loc[(dataset['Age'] > 18) & (dataset['Age'] <= 22), 'Age'] = 2
    dataset.loc[(dataset['Age'] > 22) & (dataset['Age'] <= 27), 'Age'] = 3
    dataset.loc[(dataset['Age'] > 27) & (dataset['Age'] <= 33), 'Age'] = 4
    dataset.loc[(dataset['Age'] > 33) & (dataset['Age'] <= 40), 'Age'] = 5
    dataset.loc[(dataset['Age'] > 40) & (dataset['Age'] <= 66), 'Age'] = 6
    dataset.loc[ dataset['Age'] > 66, 'Age'] = 6

# 4.8 create Fare groups same as we did with Age

data = [train_df, test_df]

for dataset in data:
    dataset.loc[ dataset['Fare'] <= 7.91, 'Fare'] = 0
    dataset.loc[(dataset['Fare'] > 7.91) & (dataset['Fare'] <= 14.454), 'Fare'] = 1
    dataset.loc[(dataset['Fare'] > 14.454) & (dataset['Fare'] <= 31), 'Fare']   = 2
    dataset.loc[(dataset['Fare'] > 31) & (dataset['Fare'] <= 99), 'Fare']   = 3
    dataset.loc[(dataset['Fare'] > 99) & (dataset['Fare'] <= 250), 'Fare']   = 4
    dataset.loc[ dataset['Fare'] > 250, 'Fare'] = 5
    dataset['Fare'] = dataset['Fare'].astype(int)

# 5 adding new features

# age * class
data = [train_df, test_df]
for dataset in data:
    dataset['Age_Class']= dataset['Age']* dataset['Pclass']

# fare per person
data = [train_df, test_df]
for dataset in data:
    dataset['relatives'] = dataset['SibSp'] + dataset['Parch']
    dataset.loc[dataset['relatives'] > 0, 'not_alone'] = 0
    dataset.loc[dataset['relatives'] == 0, 'not_alone'] = 1
    dataset['not_alone'] = dataset['not_alone'].astype(int)
    dataset['Fare_Per_Person'] = dataset['Fare']/(dataset['relatives']+1)
    dataset['Fare_Per_Person'] = dataset['Fare_Per_Person'].astype(int)
