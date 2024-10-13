import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler, LabelEncoder, StandardScaler, RobustScaler
from sklearn.impute import KNNImputer

pd.set_option('display.max_columns', None)
pd.set_option('display.width', 500)

#Dataset loading
def load_dataset():
    return pd.read_csv('Telco_Customer_Churn/dataset/Telco-Customer-Churn.csv')

df = load_dataset()


def check_df(dataframe, head=5):
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print(dataframe.head(head))
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print('##################### Unique Values #####################')
    print(dataframe.nunique())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

check_df(df)


def grab_col_names(dataframe, cat_th=10, car_th=20):

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if dataframe[col].nunique() < cat_th and
                   dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if dataframe[col].nunique() > car_th and
                   dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')
    return cat_cols, num_cols, cat_but_car

cat_cols, num_cols, cat_but_car = grab_col_names(df)

df['TotalCharges'].head()

df['TotalCharges'] = df['TotalCharges'].replace(' ', np.nan)

df['TotalCharges'] = df['TotalCharges'].astype('float')

cat_cols, num_cols, cat_but_car = grab_col_names(df)

check_df(df)

#VARIABLE SUMMARIES
def cat_summary(dataframe, col_name, plot=False):
    print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                        "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))

    if plot:
        sns.countplot(x=dataframe[col_name], data=dataframe)
        plt.show()

def num_summary(dataframe, numerical_col, plot=False):
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        sns.histplot(data = dataframe, x = numerical_col)
        #dataframe[numerical_col].hist(bins=50)
        #plt.xlabel(numerical_col)
        #plt.title(numerical_col)
        plt.show()

for col in cat_cols:
    cat_summary(df, col, True)

for col in num_cols:
    num_summary(df, col, True)

#Target encoding for EDA
df['Churn'] = df['Churn'].apply(lambda x: 1 if x == 'Yes' else 0)

#TARGET SUMMARY
def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")

for col in cat_cols:
    target_summary_with_cat(df, "Churn", col)


def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


for col in num_cols:
    target_summary_with_num(df, "Churn", col)


#OUTLIER DETECTION
def outlier_thresholds(dataframe, col_name, q1=0.05, q3=0.95):
    quartile1 = dataframe[col_name].quantile(q1)
    quartile3 = dataframe[col_name].quantile(q3)
    interquantile_range = quartile3 - quartile1
    up_limit = quartile3 + 1.5 * interquantile_range
    low_limit = quartile1 - 1.5 * interquantile_range
    return low_limit, up_limit

def check_outlier(dataframe, col_name):
    low_limit, up_limit = outlier_thresholds(dataframe, col_name)
    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True
    else:
        return False
    
for col in num_cols:
    print(col, check_outlier(df, col))


scaler = StandardScaler()
df[num_cols] = pd.DataFrame(scaler.fit_transform(df[num_cols]))
df[num_cols].head()

imputer = KNNImputer(n_neighbors=5)
df[num_cols] = pd.DataFrame(imputer.fit_transform(df[num_cols]))


def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    dataframe = pd.get_dummies(dataframe, columns=categorical_cols, drop_first=drop_first)
    return dataframe

cat_cols_without_target = [col for col in cat_cols if col not in 'Churn']

df = one_hot_encoder(df, cat_cols_without_target)


check_df(df)


import joblib
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier, VotingClassifier, AdaBoostClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_validate, GridSearchCV, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from xgboost import XGBClassifier
import warnings
from sklearn.exceptions import ConvergenceWarning
from catboost import CatBoostClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score, roc_auc_score

warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter("ignore", category=ConvergenceWarning)

def base_models(X, y, scoring="accuracy"):
    print("Base Models....")
    classifiers = [('LR', LogisticRegression()),
                   ('KNN', KNeighborsClassifier()),
                   ("SVC", SVC()),
                   ("CART", DecisionTreeClassifier()),
                   ("RF", RandomForestClassifier()),
                   ('Adaboost', AdaBoostClassifier()),
                   ('GBM', GradientBoostingClassifier()),
                   ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss')),
                   ('LightGBM', LGBMClassifier()),
                   ('CatBoost', CatBoostClassifier(verbose=False))
                   ]

    for name, classifier in classifiers:
        cv_results = cross_validate(classifier, X, y, cv=3, scoring=scoring)
        print(f"{scoring}: {round(cv_results['test_score'].mean(), 4)} ({name}) ")

# Hyperparameter Optimization

# config.py

# ElasticNet parameters
elasticnet_params = {"alpha": [0.1, 1.0, 10.0],
                     "l1_ratio": [0.1, 0.5, 0.9]}

# K-Nearest Neighbors parameters
knn_params = {"n_neighbors": range(2, 50)}

# Decision Tree parameters
cart_params = {'max_depth': range(1, 20),
               "min_samples_split": range(2, 30)}

# Random Forest parameters
rf_params = {"max_depth": [8, 15, None],
             "max_features": [5, 7, "sqrt"],
             "min_samples_split": [15, 20],
             "n_estimators": [200, 300]}

# Support Vector parameters
svc_params = {'kernel': ['linear', 'rbf'],
              'C': [0.1, 1, 10],
              'gamma': ['scale', 'auto']}

# Gradient Boosting parameters
gbm_params = {"learning_rate": [0.01, 0.1],
              "n_estimators": [100, 200],
              "max_depth": [3, 5, 7]}

# XGBoost parameters
xgboost_params = {"learning_rate": [0.1, 0.01],
                  "max_depth": [5, 8],
                  "n_estimators": [100, 200]}

# LightGBM parameters
lightgbm_params = {"learning_rate": [0.01, 0.1],
                   "n_estimators": [300, 500]}

# CatBoost parameters
catboost_params = {"depth": [4, 6, 10],
                   "learning_rate": [0.01, 0.1],
                   "iterations": [100, 200]}

classifiers = [
    ('KNN', KNeighborsClassifier(), knn_params),
    ("CART", DecisionTreeClassifier(), cart_params),
    ("RF", RandomForestClassifier(), rf_params),
    ('SVC', SVC(), svc_params),
    ('GBM', GradientBoostingClassifier(), gbm_params),
    ('XGBoost', XGBClassifier(use_label_encoder=False, eval_metric='logloss'), xgboost_params),
    ('LightGBM', LGBMClassifier(), lightgbm_params),
    ("CatBoost", CatBoostClassifier(verbose=False), catboost_params)
]

def hyperparameter_optimization(X, y, cv=3, scoring="accuracy"):
    print("Hyperparameter Optimization....")
    best_models = {}
    for name, classifier, params in classifiers:
        print(f"########## {name} ##########")
        cv_results = cross_validate(classifier, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (Before): {round(cv_results['test_score'].mean(), 4)}")

        gs_best = GridSearchCV(classifier, params, cv=cv, n_jobs=-1, verbose=False).fit(X, y)
        final_model = classifier.set_params(**gs_best.best_params_)

        cv_results = cross_validate(final_model, X, y, cv=cv, scoring=scoring)
        print(f"{scoring} (After): {round(cv_results['test_score'].mean(), 4)}")
        print(f"{name} best params: {gs_best.best_params_}", end="\n\n")
        best_models[name] = final_model
    return best_models

# Stacking & Ensemble Learning
def voting_classifier(best_models, X, y):
    print("Voting Classifier...")
    voting_clf = VotingClassifier(estimators=[('CatBoost', best_models["CatBoost"]),
                                              ('GBM', best_models["GBM"]),
                                              ('XGBoost', best_models["XGBoost"]),
                                              ('RF', best_models["RF"]),
                                              ('LightGBM', best_models["LightGBM"]),
                                              ],
                                  voting='soft').fit(X, y)
    cv_results = cross_validate(voting_clf, X, y, cv=3, scoring=["accuracy", "f1", "roc_auc"])
    print(f"Accuracy: {cv_results['test_accuracy'].mean()}")
    print(f"F1Score: {cv_results['test_f1'].mean()}")
    print(f"ROC_AUC: {cv_results['test_roc_auc'].mean()}")
    return voting_clf

X = df.drop(['Churn', 'customerID'], axis=1)
y = df['Churn']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=9)

base_models(X_train, y_train)
best_models = hyperparameter_optimization(X_train, y_train)
voting_clf = voting_classifier(best_models, X_train, y_train)
predictions = voting_clf.predict(X_test)

accuracy = accuracy_score(y_test, predictions)
conf_matrix = confusion_matrix(y_test, predictions)
f1 = f1_score(y_test, predictions, average='weighted')
precision = precision_score(y_test, predictions, average='weighted')
recall = recall_score(y_test, predictions, average='weighted')

print(f"Accuracy: {accuracy}")
print(f"Confusion Matrix:\n{conf_matrix}")
print(f"F1 Score: {f1}")
print(f"Precision: {precision}")
print(f"Recall: {recall}")

# ROC AUC (sadece ikili sınıflandırma için geçerli)
if hasattr(voting_clf, "predict_proba"):
    probabilities = voting_clf.predict_proba(X_test)
    roc_auc = roc_auc_score(y_test, probabilities[:, 1])
    print(f"ROC AUC Score: {roc_auc}")

#dictionary = {"Id":test_prep.index, "Degerlendirme Puani":predictions}
#dfSubmission = pd.DataFrame(dictionary)

#dfSubmission['SalePrice'] = pd.DataFrame(scaler.inverse_transform(dfSubmission['SalePrice']))
#dfSubmission['SalePrice'] = np.exp(dfSubmission['SalePrice'])
    
#dfSubmission.to_csv("predictions.csv", index=False)

joblib.dump(voting_clf, "voting_clf.pkl")