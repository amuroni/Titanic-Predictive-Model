import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier

from Machine_Learning_Models import X_train, Y_train

# applying the K-Fold analysis to Random Forest and Decision Tree
rf = RandomForestClassifier(n_estimators=100)
rf_score = cross_val_score(rf, X_train, Y_train, cv=10, scoring="accuracy")  # using K=10 folds
print("RF_Scores:", np.round(rf_score, 5))
print("RF_Mean:", np.round(rf_score.mean(), 5))
print("RF_standard deviation:", np.round(rf_score.std(), 5) * 100, "%")  # 4% standard_deviation

dt = DecisionTreeClassifier()
dt_score = cross_val_score(dt, X_train, Y_train, cv=10, scoring="accuracy")  # still using K=10 folds
print("DT Scores:", np.round(dt_score, 5))
print("DT Mean:", np.round(dt_score.mean(), 5))
print("DT standard deviation:", np.round(dt_score.std(), 5) * 100, "%")  # almost 6% standard deviation
