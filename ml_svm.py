"""SVMでirisの分類器をつくる
http://scikit-learn.org/stable/tutorial/basic/tutorial.html
"""

from sklearn import svm, datasets
from sklearn.cross_validation import train_test_split
from sklearn.metrics import confusion_matrix, accuracy_score
import numpy as np
from matplotlib import pyplot as plt

iris = datasets.load_iris()
iris_data_train, iris_data_test, iris_target_train, iris_target_test = train_test_split(iris.data, iris.target, test_size=0.5)

iris_predict = svm.SVC().fit(iris_data_train, iris_target_train).predict(iris_data_test)

cm = confusion_matrix(iris_target_test, iris_predict)
cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
ascore = accuracy_score(iris_target_test, iris_predict)
print (cm)
print ("accuracy score: %.5f" % ascore)

def plot_confusion_matrix(cm, title='Confusion matrix', cmap=plt.cm.Blues):
    ''' confusion_matrixをheatmap表示する関数
    Args:
        cm -- confusion_matrix
        title -- 図の表題
        cmap -- 使用するカラーマップ
 
    '''
    plt.imshow(cm, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    tick_marks = np.arange(len(iris.target_names))
    plt.xticks(tick_marks, iris.target_names, rotation=45)
    plt.yticks(tick_marks, iris.target_names)
    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')

print('Normalized confusion matrix')
print(cm_normalized)
plt.figure()
plot_confusion_matrix(cm_normalized)
plt.show()