import pandas as pd
from sklearn import linear_model
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import LinearSVC
from sklearn.tree import DecisionTreeClassifier

from Main import test_df
from Main import train_df

# set X and Y train and X test datasets
X_train = train_df.drop("Survived", axis=1)
Y_train = train_df["Survived"]
X_test = test_df.drop("PassengerId", axis=1).copy()

# STOCHASTIC GRADIENT DESCENT - SGD
sgd = linear_model.SGDClassifier(max_iter=5, tol=None)
sgd.fit(X_train, Y_train)
sgd_pred = sgd.predict(X_test)
sgd_score = sgd.score(X_train, Y_train)
sgd_acc = round(sgd_score * 100, 2)
print("SGD Score = ", round(sgd_acc), "%")  # approximately 75%

# RANDOM FOREST
rand_forest = RandomForestClassifier(n_estimators=100)
rand_forest.fit(X_train, Y_train)
rf_pred = rand_forest.predict(X_test)
rf_score = rand_forest.score(X_train, Y_train)
rf_acc = round(rf_score * 100, 2)
print("RF Score = ", round(rf_acc), "%")  # approximately 93%

# LOGISTIC REGRESSION
logreg = LogisticRegression(max_iter=4000)
logreg.fit(X_train, Y_train)
lr_pred = logreg.predict(X_test)
lr_score = logreg.score(X_train, Y_train)
lr_acc = round(lr_score * 100, 2)
print("LR Score = ", round(lr_acc), "%")  # approximately 81%

# KNN
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, Y_train)
knn_pred = knn.predict(X_test)
knn_score = knn.score(X_train, Y_train)
knn_acc = round(knn_score * 100, 2)
print("KNN Score = ", round(knn_acc), "%")  # approximately 88%

# GAUSSIAN NAIVE BAYES
gaussian = GaussianNB()
gaussian.fit(X_train, Y_train)
gaussian_pred = gaussian.predict(X_test)
gaussian_score = gaussian.score(X_train, Y_train)
gaussian_acc = round(gaussian_score * 100, 2)
print("Gaussian Score = ", round(gaussian_acc), "%")  # approximately 79%

# PERCEPTRON
perceptron = Perceptron(max_iter=1000)
perceptron.fit(X_train, Y_train)
perceptron_pred = perceptron.predict(X_test)
perceptron_score = perceptron.score(X_train, Y_train)
perceptron_acc = round(perceptron_score * 100, 2)
print("Perceptron Score = ", round(perceptron_acc), "%")  # approximately 79%

# LINEAR SVC
linear_svc = LinearSVC(max_iter=10000)
linear_svc.fit(X_train, Y_train)
lsvc_pred = linear_svc.predict(X_test)
lsvc_score = linear_svc.score(X_train, Y_train)
lsvc_acc = round(lsvc_score * 100, 2)
print("LSVC Score = ", round(lsvc_acc), "%")  # approximately 82% - might increase n. of iterations

# DECISION TREE
decision_tree = DecisionTreeClassifier()
decision_tree.fit(X_train, Y_train)
dt_pred = decision_tree.predict(X_test)
dt_score = decision_tree.score(X_train, Y_train)
dt_acc = round(dt_score * 100, 2)
print("Decision Tree Score = ", round(dt_acc), "%") # approximately 93%

# ML RESULTS

results = pd.DataFrame({
    "Model": ["Stochastic Gradient Descent", "Random Forest", "Logistic Regression", "KNN", "Gaussian Naive Bayes",
              "Perceptron", "Linear SVC", "Decision Tree"],
    "Score": [sgd_acc, rf_acc, lr_acc, knn_acc, gaussian_acc, perceptron_acc, lsvc_acc, dt_acc]
    })

results_df = results.sort_values(by="Score", ascending=False)
results_df = results_df.set_index("Score")
results_df.head()
