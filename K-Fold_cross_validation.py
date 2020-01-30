import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
import pandas as pd
from matplotlib import pyplot as plt

from Machine_Learning_Models import X_train, Y_train, X_test, rand_forest
from Main import train_df, test_df

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

# Random forest makes the better choice
# we can analyze the importance of each feature with sklearn

importance = pd.DataFrame({"feature": X_train.columns,
                           "importance": np.round(rand_forest.feature_importances_, 3)})
importance_sorted = importance.sort_values("importance", ascending=False).set_index("feature")
print(importance_sorted)
plot = importance_sorted.plot.bar()
plt.show()

# drop Parch and not_alone since they have no significance
train_df = train_df.drop("not_alone", axis=1)
test_df = test_df.drop("not_alone", axis=1)

train_df = train_df.drop("Parch", axis=1)
test_df = test_df.drop("Parch", axis=1)

# rerun the random forest model to check accuracy
rand_forest2 = RandomForestClassifier(n_estimators=100, oob_score=True)
rand_forest2.fit(X_train, Y_train)
Y_pred= rand_forest2.predict(X_test)
rand_forest2_acc = round(rand_forest2.score(X_train, Y_train) * 100, 2)
print("Random Forest 2 accuracy: " + str(round(rand_forest2_acc, 2)), "%")
print("OOB Score of Random Forest 2: " + str(round(rand_forest2.oob_score, 4)*100), "%")
# almost 93% accuracy and 82% OOB score, good
