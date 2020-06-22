from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score


def classification_report(y, y_pred):
    accuracy = accuracy_score(y, y_pred)
    print('Accuracy: %f' % accuracy)
    precision = precision_score(y, y_pred, average="macro")
    print('Precision: %f' % precision)
    recall = recall_score(y, y_pred, average="macro")
    print('Recall: %f' % recall)
    f1 = f1_score(y, y_pred, average="macro")
    print('F1 score: %f' % f1)

    return accuracy, precision, recall, f1
