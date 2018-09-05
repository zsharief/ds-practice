## calculate roc score
# false_pos_rate, true_pos_rate, thresholds = roc_curve(true_labels, predicted_probs)

## calculate AUC
# auc = roc_auc_score(y, predicted_probs)

from sklearn.datasets import make_classification
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from matplotlib import pyplot
