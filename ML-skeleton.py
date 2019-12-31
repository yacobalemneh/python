import pandas as pd 
import numpy as np
import csv 
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.svm import SVC
from sklearn.svm import LinearSVC
from sklearn import tree
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score


df = pd.read_csv("test15.csv", header=None)
# You might not need this next line if you do not care about losing information about flow_id etc. All you actually need to
# feed your machine learning model are features and output label.
columns_list = ['Conn','len','min_len','max_len','std_len','time_diff','std_time','label']
df.columns = columns_list
features = ['Conn','len','min_len','max_len','std_len','time_diff','std_time']

X = df[features]
y = df['label']
f1_list = [[] for i in range(3)] # store the f1 scores for the three ML algorithims
acc_list = [[] for i in range(3)] # store the accuracy scores for the three ML algorithims
prec_list = [[] for i in range(3)] # store the precision scores for the three ML algorithims
recall_list = [[] for i in range(3)] # store the recall scores for the three ML algorithims

acc_scores = 0
for i in range(0, 10):
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.25)

    #Decision Trees
    clf = tree.DecisionTreeClassifier()
    clf.fit(X_train, y_train)
    y_pred = clf.predict(X_test)
    # f1 score
    f1_score_tree = f1_score(y_test,y_pred,average='macro')
    p_score_tree = precision_score(y_test, y_pred, average="macro")
    r_score_tree = recall_score(y_test, y_pred, average="macro")
    
    # Neural network (MultiPerceptron Classifier)
    dtf = MLPClassifier()
    dtf.fit(X_train, y_train)
    y_pred2 = dtf.predict(X_test)
    f1_score_MLP = f1_score(y_test,y_pred2,average='macro')
    p_score_MLP = precision_score(y_test, y_pred2, average="macro")
    r_score_MLP = recall_score(y_test, y_pred2, average="macro")

    #SVM's
    svf = SVC(gamma='auto')     #SVC USE THIS
    svf = LinearSVC()  #Linear SVC
    svf.fit(X_train, y_train) 
    y_pred3 = svf.predict(X_test)
    f1_score_SVM = f1_score(y_test,y_pred3,average='macro')
    p_score_SVM = precision_score(y_test, y_pred3, average="macro")
    r_score_SVM = recall_score(y_test, y_pred3, average="macro")

    #here you are supposed to calculate the evaluation measures indicated in the project proposal (accuracy, F-score etc)
    result_tree = clf.score(X_test, y_test)  #accuracy score for decision tree
    result_MLP = dtf.score(X_test,y_test)
    result_SVM = svf.score(X_test,y_test)

    f1_list[0].extend([f1_score_tree])# extend to f1 score list
    f1_list[1].extend([f1_score_MLP])
    f1_list[2].extend([f1_score_SVM])

    prec_list[0].extend([p_score_tree])# extend to precision list
    prec_list[1].extend([p_score_MLP])
    prec_list[2].extend([p_score_SVM])

    recall_list[0].extend([r_score_tree])#extend to  recall list
    recall_list[1].extend([r_score_MLP])
    recall_list[2].extend([r_score_SVM])

    acc_list[0].extend([result_tree])# extend to accuracy list
    acc_list[1].extend([result_MLP])
    acc_list[2].extend([result_SVM])

final_list = []
for i in range(3):# out put the means of the lists
   print(np.mean(f1_list[i]))
   print(np.mean(prec_list[i]))
   print(np.mean(recall_list[i]))
   print(np.mean(acc_list[i]))
   final_list.append([np.mean(f1_list[1]),np.mean(prec_list[i]),np.mean(recall_list[i]),np.mean(acc_list[i])])

print(*final_list)
