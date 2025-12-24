import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report,accuracy_score,confusion_matrix
from sklearn.metrics import roc_auc_score,roc_curve,auc
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV,RandomizedSearchCV
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import label_binarize
import sys
import warnings
warnings.filterwarnings("ignore")

class IMAGE_PREDICTION():
  def __init__(self,path):
    try:
      self.path = path
      self.df = pd.read_csv(self.path)
      self.df = self.df.fillna(0)
      self.X = self.df.iloc[: , 1 : ]
      self.y = self.df.iloc[: , 0]
      self.classes = sorted(self.y.unique())
      self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)
      self.y_test_bin = label_binarize(self.y_test, classes=self.classes)
    except Exception as e:
      ex_type,ex_msg,ex_line = sys.exc_info()
      print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

  def knn(self):

     self.knn_reg = KNeighborsClassifier(n_neighbors=5)
     self.knn_reg.fit(self.X_train,self.y_train)


  def nb(self):

     self.nb_reg = GaussianNB()
     self.nb_reg.fit(self.X_train,self.y_train)


  def lr(self):

     self.lr_reg = LogisticRegression(max_iter= 20)
     self.lr_reg.fit(self.X_train,self.y_train)

  def dt(self):

     self.dt_reg = DecisionTreeClassifier(criterion='entropy')
     self.dt_reg.fit(self.X_train,self.y_train)


  def rf(self):

     self.rf_reg = RandomForestClassifier(n_estimators=20)
     self.rf_reg.fit(self.X_train,self.y_train)

  def ada(self):

     self.ada_reg = AdaBoostClassifier(n_estimators=20)
     self.ada_reg.fit(self.X_train,self.y_train)


  def gb(self):

    self.gb_reg = GradientBoostingClassifier(n_estimators=20)
    self.gb_reg.fit(self.X_train,self.y_train)

  def xgb(self):

    self.xgb_reg = XGBClassifier(n_estimators = 20)
    self.xgb_reg.fit(self.X_train,self.y_train)

  def svm(self):

    self.svm_reg = SVC(kernel='rbf',probability = True)
    self.svm_reg.fit(self.X_train,self.y_train)

  def model_data(self, model, model_name):
    # Train Accuracy
    train_pred = model.predict(self.X_train)
    train_acc = accuracy_score(self.y_train, train_pred)

    # Test Accuracy
    test_pred = model.predict(self.X_test)
    test_acc = accuracy_score(self.y_test, test_pred)

    print("\n==============================")
    print(f" MODEL : {model_name}")
    print("==============================")
    print(f"Train Accuracy : {train_acc:.4f}")
    print(f"Test  Accuracy : {test_acc:.4f}")

    print("\nClassification Report:")
    print(classification_report(self.y_test, test_pred))

    print("\nConfusion Matrix:")
    print(confusion_matrix(self.y_test, test_pred))
    print("==============================\n")



  def common(self):
    try:
     self.knn()
     self.model_data(self.knn_reg,'knn')
     self.nb()
     self.model_data(self.nb_reg,'nb')
     self.lr()
     self.model_data(self.lr_reg,'lr')
     self.dt()
     self.model_data(self.dt_reg,'dt')
     self.rf()
     self.model_data(self.rf_reg,'rf')
     self.ada()
     self.model_data(self.ada_reg,'ada')
     self.gb()
     self.model_data(self.gb_reg,'gb')
     self.xgb()
     self.model_data(self.xgb_reg,'xg')
     self.svm()
     self.model_data(self.svm_reg,'svm')
    except Exception as e:
      ex_type,ex_msg,ex_line = sys.exc_info()
      print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')
      
  def predictions(self):
    self.knn_predictions = self.knn_reg.predict_proba(self.X_test)
    self.nb_predictions = self.nb_reg.predict_proba(self.X_test)
    self.lr_predictions = self.lr_reg.predict_proba(self.X_test)
    self.dt_predictions = self.dt_reg.predict_proba(self.X_test)
    self.rf_predictions = self.rf_reg.predict_proba(self.X_test)
    self.ada_predictions = self.ada_reg.predict_proba(self.X_test)
    self.gb_predictions = self.gb_reg.predict_proba(self.X_test)
    self.xgb_predictions = self.xgb_reg.predict_proba(self.X_test)
    self.svm_predictions = self.svm_reg.predict_proba(self.X_test)

  def curve(self):
    # Micro-average ROC: flatten y and predictions
    y_true = self.y_test_bin.ravel()

    # Compute FPR, TPR, AUC for each model
    def roc_values(proba):
        fpr, tpr, _ = roc_curve(y_true, proba.ravel())
        auc_val = auc(fpr, tpr)
        return fpr, tpr, auc_val

    self.knn_fpr, self.knn_tpr, self.knn_auc = roc_values(self.knn_predictions)
    self.nb_fpr,  self.nb_tpr,  self.nb_auc  = roc_values(self.nb_predictions)
    self.lr_fpr,  self.lr_tpr,  self.lr_auc  = roc_values(self.lr_predictions)
    self.dt_fpr,  self.dt_tpr,  self.dt_auc  = roc_values(self.dt_predictions)
    self.rf_fpr,  self.rf_tpr,  self.rf_auc  = roc_values(self.rf_predictions)
    self.ada_fpr, self.ada_tpr, self.ada_auc = roc_values(self.ada_predictions)
    self.gb_fpr,  self.gb_tpr,  self.gb_auc  = roc_values(self.gb_predictions)
    self.xgb_fpr, self.xgb_tpr, self.xgb_auc = roc_values(self.xgb_predictions)
    self.svm_fpr, self.svm_tpr, self.svm_auc = roc_values(self.svm_predictions)

  def figure(self):
    plt.figure(figsize=(7,5))
    plt.plot([0, 1], [0, 1], "k--")

    plt.plot(self.knn_fpr, self.knn_tpr, label=f"KNN ({self.knn_auc:.3f})")
    plt.plot(self.nb_fpr,  self.nb_tpr,  label=f"NB ({self.nb_auc:.3f})")
    plt.plot(self.lr_fpr,  self.lr_tpr,  label=f"LR ({self.lr_auc:.3f})")
    plt.plot(self.dt_fpr,  self.dt_tpr,  label=f"DT ({self.dt_auc:.3f})")
    plt.plot(self.rf_fpr,  self.rf_tpr,  label=f"RF ({self.rf_auc:.3f})")
    plt.plot(self.ada_fpr, self.ada_tpr, label=f"ADA ({self.ada_auc:.3f})")
    plt.plot(self.gb_fpr,  self.gb_tpr,  label=f"GB ({self.gb_auc:.3f})")
    plt.plot(self.xgb_fpr, self.xgb_tpr, label=f"XGB ({self.xgb_auc:.3f})")
    plt.plot(self.svm_fpr, self.svm_tpr, label=f"SVM ({self.svm_auc:.3f})")

    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC CURVE")
    plt.legend(loc="lower right")
    plt.show()


  def best_model(self):
    # Dictionary of model name → AUC score
    auc_scores = {
        "KNN": self.knn_auc,
        "NB": self.nb_auc,
        "LR": self.lr_auc,
        "DT": self.dt_auc,
        "RF": self.rf_auc,
        "ADA": self.ada_auc,
        "GB": self.gb_auc,
        "XGB": self.xgb_auc,
        "SVM": self.svm_auc
    }

    # Find best model
    best_name = max(auc_scores, key=auc_scores.get)
    best_auc = auc_scores[best_name]

    print("\n BEST MODEL RESULT ")
    print("==============================")
    print(f"Best Model : {best_name}")
    print(f"AUC Score  : {best_auc:.4f}")
    print("==============================\n")

    # Return both
    return best_name, best_auc

if __name__ == '__main__':
  try:
    dataset_path = "C:\\Users\\hp\\Downloads\\Mnist_project\\mnist_train.csv"
    obj = IMAGE_PREDICTION(dataset_path)
    obj.common()
    obj.predictions()
    obj.curve()
    obj.figure()
    obj.best_model()


  except Exception as e:
    ex_type,ex_msg,ex_line = sys.exc_info()
    print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

df = pd.read_csv("C:\\Users\\hp\\Downloads\\Mnist_project\\mnist_train.csv")
df = df.dropna()
df=df.head(5000)
X = df.iloc[: , 1:] # independent
y = df.iloc[: , 0] # dependent
X_train,X_test,y_train,y_test = train_test_split(X,y,test_size=0.2,random_state=42)
reg = SVC()
parameters_from_develop = {'kernel':['rbf','poly','linear'],
                           'C': [1.0,2.0,3.0],
                           'class_weight':[None,'balanced'],
                           'decision_function_shape': ['ovr', 'ovo']
                           }



grid_model = GridSearchCV(estimator = reg,
             param_grid = parameters_from_develop,
             cv=10,
             scoring = 'accuracy')

result = grid_model.fit(X_train,y_train)
result.best_params_

class MNIST_PREDICTION():
  def __init__(self,path):
    try:
      self.path = path
      self.df = pd.read_csv(self.path)
      self.df = self.df.fillna(0)
      self.X = self.df.iloc[: , 1 : ]
      self.y = self.df.iloc[: , 0]
      self.X_train,self.X_test,self.y_train,self.y_test = train_test_split(self.X,self.y,test_size=0.2,random_state=42)


    except Exception as e:
      ex_type,ex_msg,ex_line = sys.exc_info()
      print(f'Issue is from {ex_line.tb_lineno} : due to : {ex_msg}')

  def training(self):
    try:
      self.svm=SVC(C=3.0,class_weight = None,decision_function_shape='ovr',kernel='rbf')
      self.svm.fit(self.X_train,self.y_train)
      print(f'Train Accuracy:{accuracy_score(self.y_train,self.svm.predict(self.X_train))}')
      print(f'Classification Report:{classification_report(self.y_train,self.svm.predict(self.X_train))}')
      print(f'Confusion Matrix:{confusion_matrix(self.y_train,self.svm.predict(self.X_train))}')

    except Exception as e:
      ex_type,ex_tb,ex_msg=sys.exc_info()
      print(f'issue is from {ex_tb.tb_lineno} due to {ex_msg}')

  def testing(self):
    try:
      print(f'Test Accuracy:{accuracy_score(self.y_test,self.svm.predict(self.X_test))}')
      print(f'Classification Report:{classification_report(self.y_test,self.svm.predict(self.X_test))}')
      print(f'Confusion Matrix:{confusion_matrix(self.y_test,self.svm.predict(self.X_test))}')

    except Exception as e:
      ex_type,ex_tb,ex_msg=sys.exc_info()
      print(f'issue is from {ex_tb.tb_lineno} due to {ex_msg}')

if __name__ == '__main__':
  try:
    dataset_path=("C:\\Users\\hp\\Downloads\\Mnist_project\\mnist_train.csv")
    obj=MNIST_PREDICTION(dataset_path)
    obj.training()
    obj.testing()
  except Exception as e:
       ex_type,ex_tb,ex_msg=sys.exc_info()
       print(f'issue is from {ex_tb.tb_lineno} due to {ex_msg}')