import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import roc_curve, auc, precision_recall_curve, f1_score, confusion_matrix
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn import svm
import lightgbm as lgb
import shap
import warnings
warnings.filterwarnings('ignore')

def shap_plot(model, data, center, modelname):
    if modelname == 'RF' or modelname == 'LGBM':
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(data, approximate=False, check_additivity=False)
    elif modelname == 'LR':
        masker = shap.maskers.Independent(data=data)
        explainer = shap.LinearExplainer(model, masker=masker)
        shap_values = explainer.shap_values(data)
    elif modelname == 'SVM' or modelname == 'MLP':
        explainer = shap.KernelExplainer(model.predict_proba, data)
        shap_values = explainer.shap_values(data)

    if isinstance(shap_values, list):
        shap.summary_plot(shap_values[1], data, max_display=10, show=False, plot_size=(12, 8))
    elif isinstance(shap_values, np.ndarray):
        shap.summary_plot(shap_values, data, max_display=10, show=False, plot_size=(12, 8))

    plt.savefig('plots/SHAP_{}_{}.png'.format(center, modelname))
    plt.close()

def plot_aucs(label, score, name):
    sns.set_style('whitegrid')
    fig, axs = plt.subplots(1, 2, figsize=(10.5, 4.5))
    axs = axs.ravel()

    precision, recall, _ = precision_recall_curve(label, score, pos_label=1)
    auprc = auc(recall, precision)
    fpr, tpr, _ = roc_curve(label, score, pos_label=1)
    auroc = auc(fpr, tpr)

    axs[0].plot(fpr, tpr, c='b', lw=2, label='AUROC: {:.3f}'.format(auroc))
    axs[0].set_xlabel('fpr', fontsize=15)
    axs[0].set_ylabel('tpr', fontsize=15)
    axs[0].plot([0, 1], [0, 1], linestyle='dashed')
    axs[0].set_xlim([0, 1.05])
    axs[0].set_ylim([0, 1.05])
    axs[0].legend()

    axs[1].plot(recall, precision, c='b', lw=2,
                label='AUPRC: {:.3f}'.format(auprc))
    axs[1].plot([0, 1], [np.mean(label), np.mean(label)], linestyle='dashed')
    axs[1].set_xlim([0, 1.05])
    axs[1].set_ylim([0, 1.05])
    axs[1].legend()
    axs[1].set_xlabel('recall', fontsize=15)
    axs[1].set_ylabel('precision', fontsize=15)

    plt.tight_layout()
    plt.savefig('plots/{}.png'.format(name), dpi=300)
    plt.close()

def plot_aucs_withUncertainty(data, name):
    labels = []
    scores = []
    aucs = []
    precisions, recalls, fprs, tprs, auprcs, aurocs = [], [], [], [], [], []
    mean_fpr = np.linspace(0, 1, 100)
    mean_recall = np.linspace(0, 1, 100)
    for j in range(len(data[0])):
        label = data[0][j].tolist()
        score = data[1][j]
        labels.extend(label)
        scores.extend(score)
        precision, recall, _ = precision_recall_curve(label, score, pos_label=1)
        precision, recall = precision[::-1], recall[::-1]
        auprc = auc(recall, precision)
        interp_precision = np.interp(mean_recall, recall, precision)
        precisions.append(interp_precision)
        auprcs.append(auprc)
        fpr, tpr, _ = roc_curve(label, score, pos_label=1)
        auroc = auc(fpr, tpr)
        interp_tpr = np.interp(mean_fpr, fpr, tpr)
        interp_tpr[0] = 0.0
        tprs.append(interp_tpr)
        aucs.append(auroc)
    mean_tpr = np.mean(tprs, axis=0)
    std_tpr = np.std(tprs, axis=0)
    tprs_upper = np.minimum(mean_tpr + std_tpr, 1)
    tprs_lower = np.maximum(mean_tpr - std_tpr, 0)
    precisions, recalls, _ = precision_recall_curve(labels, scores)
    auprc = auc(recalls, precisions)
    std_auprc = np.std(auprcs)
    std_precision = np.std(precisions, axis=0)
    fprs, tprs, _ = roc_curve(labels, scores, pos_label=1)
    auroc = auc(fprs, tprs)
    std_auroc = np.std(aucs)

    plt.figure(figsize=(10.5, 4.5))
    plt.subplot(121)
    plt.plot(mean_fpr, mean_tpr, color='r',
             label='AUROC {:.3f}'.format(auroc),
             lw=2, alpha=.8)
    plt.fill_between(mean_fpr, tprs_lower, tprs_upper, color='r', alpha=0.3, )
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.legend()
    plt.ylabel('tpr', fontsize=15)
    plt.xlabel('fpr', fontsize=15)
    plt.title('ROC', fontsize=15)
    plt.subplot(122)
    plt.plot(recalls, precisions, lw=2, c='r',
             label='AUPRC {:.3f}'.format(auprc))
    plt.fill_between(recalls, precisions + std_precision,
                     precisions - std_precision, alpha=0.3, linewidth=0, color='r')
    plt.plot([0, 1], [np.mean(labels), np.mean(labels)], color='navy', lw=2, linestyle='--')
    plt.xlabel('recall', fontsize=15)
    plt.ylabel('precision', fontsize=15)
    plt.xlim([0, 1])
    plt.ylim([0, 1.05])
    plt.legend()
    plt.title('PRC', fontsize=15)
    plt.savefig('plots/{}.png'.format(name), dpi=300)
    plt.close()
    return auprc, auroc, std_auprc, std_auroc


def get_label(data_day, daytime):
    data_day['MODS'] = data_day[['cons05.resp', 'cons05.cvs', 'cons05.cns',
          'cons05.ren', 'cons05.hep', 'cons05.hem']].sum(1)
    data_day['MODS'] = (data_day['MODS']>=2).astype(int)
    patients = []
    nb_pid = 0
    for pid, patient in data_day.groupby('sample.id'):
        nb_pid += 1
        if daytime in patient['daytime'].tolist() and patient[patient['daytime'] == daytime]['MODS'].values[0] == 1:
                patient['label_MODS_day{}'.format(daytime)] = 1
        if patient['death.delay'].values[0] <= daytime:
            patient.loc[:, 'label_death_day{}'.format(daytime)] = 1
        patients.append(patient)

    patients = pd.concat(patients)
    patients_bc = patients[patients['daytime'] == 0]
    nb_pid_bc = len(patients_bc)
    patients_bc['label'] = (patients_bc[['label_MODS_day{}'.format(daytime),
                                         'label_death_day{}'.format(daytime)]].sum(1)==0).astype(int)

    patients_bc_MODS = patients_bc[patients_bc['MODS'] == 1]
    nb_pid_bc_MODS = len(patients_bc_MODS)
    return patients_bc_MODS, nb_pid, nb_pid_bc, nb_pid_bc_MODS

def metrics_80recall(model, test, label, score):
    precisions, recalls, thresholds = precision_recall_curve(label, score)
    threshold = thresholds[np.abs(recalls - 0.8).argmin()]
    pred = (model.predict_proba(test)[:, 1] >= threshold).astype(int).flatten()
    test_table = pd.DataFrame(columns=['true', 'pred'])
    test_table['true'] = list(label)
    test_table['pred'] = list(pred)
    tn, fp, fn, tp = confusion_matrix(test_table['true'], test_table['pred']).ravel()
    precision = tp / (tp + fp)
    specificity = tn / (tn + fp)
    f1 = f1_score(test_table['true'], test_table['pred'])
    npv = tn / (tn + fn)
    accuracy = (tp + tn) / (tp + fp + tn + fn)
    recall = tp / (tp + fn)
    return [precision,  specificity, f1, npv, accuracy, recall]

def RF_grid(data, label, scoring):
    grid = {"n_estimators": np.arange(100, 2000, 200),
           'max_depth':[10,20,30,40,50,60],
           'max_leaf_nodes': [50, 100, 200],
           'min_samples_leaf':[1,2,4]}
    rf = RandomForestClassifier()
    rf_grid = RandomizedSearchCV(rf, grid, n_iter=50, cv=5, n_jobs=-1, scoring=scoring)
    rf_grid.fit(data, label)
    return rf_grid

def MLP_grid(data, label, scoring):
    grid = {
        'hidden_layer_sizes': [(25, 50, 25), (25, 50),
                               (25, 25), (50, 50),
                               (50, 100), (50, 50, 50),
                              (50, 100, 50), ],
        'activation': ['tanh', 'relu'],
        'solver': ['adam'],
        'alpha': [0.0001, 0.00001, 0.001],
        'learning_rate': ['constant', 'adaptive'],
        'learning_rate_init':[0.001, 0.0001],
        'max_iter':[500, 1000, 5000, 10000]
    }
    mlp = MLPClassifier()
    mlp_grid = RandomizedSearchCV(mlp, grid, n_iter=50, cv=5, n_jobs=-1, scoring=scoring)
    mlp_grid.fit(data, label)
    return mlp_grid

def SVM_grid(data, label, scoring):
    grid = {
        {"kernel": ["rbf"], "gamma": [1e-2, 1e-3, 1e-4], 'C':np.logspace(-4,4,15)},
        {"kernel": ["linear"], 'C':np.logspace(-4,4,15)},
        }
    model = svm.SVC(probability=True)
    svm_grid = RandomizedSearchCV(model, grid, n_iter=50, cv=5, n_jobs=-1, scoring=scoring)
    svm_grid.fit(data, label)
    return svm_grid

def LR_grid(data, label, scoring):
    clf = LogisticRegression(max_iter=15000)
    grid ={'penalty': ["l1",  'l2'],
           'C':np.logspace(-4,4,20),
           'solver':['lbfgs’', 'liblinear']}
    lr_grid = GridSearchCV(clf, grid, n_jobs=-1, cv=5, scoring=scoring)
    lr_grid.fit(data, label)
    return lr_grid

def LGBM_grid(data, label, scoring):
    clf = lgb.LGBMClassifier(
        random_state=42,
        objective='binary',
        is_unbalance=True
    )
    grid = {
        'learning_rate':[0.01, 0.005, 0.001],
        'boosting_type': ["gbdt", "dart"],
        'n_estimators': np.arange(100, 2000, 200),
        'max_depth': [10, 20, 30, 40, 50, 60],
        'subsample': [0.8, 0.9, 1]
    }
    gbm_grid = algorithm_pipeline(data, label, clf, grid, scoring)
    return gbm_grid

def algorithm_pipeline(train_x, train_y,
                model, param_grid, scoring):
    rs = RandomizedSearchCV(
        estimator=model,
        param_distributions=param_grid,
        n_iter=50,
        cv=5,
        n_jobs=-1,
        scoring=scoring
    )
    fitted_CV = rs.fit(train_x, train_y)
    return fitted_CV

def cross_validation(train_list, val_list, test_list, features_selected, features_to_scale, model):
    predictions_val = []
    labels_val = []
    predictions_test = []
    labels_test = []
    models = []
    scoring = 'roc_auc'
    scaler = StandardScaler()
    for i in range(len(train_list)):

        X_train = train_list[i][features_selected]
        X_train[features_to_scale] = scaler.fit_transform(X_train[features_to_scale])
        X_train[features_to_scale] = X_train[features_to_scale].fillna(0)
        y_train = train_list[i]['label']

        X_val = val_list[i][features_selected]
        X_val[features_to_scale] = scaler.transform(X_val[features_to_scale])
        X_val[features_to_scale] = X_val[features_to_scale].fillna(0)
        y_val = val_list[i]['label']

        X_test = test_list[i][features_selected]
        X_test[features_to_scale] = scaler.transform(X_test[features_to_scale])
        X_test[features_to_scale] = X_test[features_to_scale].fillna(0)
        y_test = test_list[i]['label']

        fitted_CV = model(X_train, y_train, scoring)
        pred_val = fitted_CV.best_estimator_.predict_proba(X_val)[:, 1]
        pred_test = fitted_CV.best_estimator_.predict_proba(X_test)[:, 1]

        predictions_val.extend(list(pred_val))
        labels_val.extend(list(y_val))
        predictions_test.extend(list(pred_test))
        labels_test.extend(list(y_test))
        models.append(fitted_CV)

    auprc_val = average_precision_score(labels_val, predictions_val)
    auroc_val = roc_auc_score(labels_val, predictions_val)
    auprc_test = average_precision_score(labels_test, predictions_test)
    auroc_test = roc_auc_score(labels_test, predictions_test)

    return auprc_val, auroc_val, auprc_test, auroc_test, models, predictions_test, labels_test

def enrichment_test(From, To, test_col, alpha=0.05):

    # test for enrichment/depletion in From set compared to To set.
    counts = {}
    draw_num = To.shape[0]
    # 100 random samplings
    instances = From[test_col].unique()
    num_test = instances.shape[0] * 100
    correction_num = From[test_col].unique().shape[0]

    for instance in instances:
        counts[instance] = []

    for k in range(num_test):
        draw = From.sample(draw_num)[test_col].value_counts() / draw_num
        for instance in instances:
            if instance in draw.index:
                counts[instance].append((draw)[instance])
            else:
                counts[instance].append(0)

    ratios_to = To[test_col].value_counts() / len(To)
    ratios_from = From[test_col].value_counts() / len(From)

    base = {}
    report = pd.DataFrame(columns=['Group', 'Instance', 'From', 'To', 'Type'])
    i = 1

    for instance in instances:
        if instance in ratios_to.index:
            base[instance] = 1 if (ratios_to)[instance] >= (ratios_from)[instance] else 0
        else:
            ratios_to[instance] = 0
            base[instance] = 0

        if base[instance] == 1 and np.sum(np.array(counts[instance]) >= (ratios_to)[instance]) <= (
                num_test * alpha / correction_num):
            print('Enrichment of {} in FP from {:.3f} to {:.3f}.'.format(
                instance, (ratios_from)[instance], (ratios_to)[instance]))
            report.loc[i] = [test_col, instance, (ratios_from)[instance],
                             (ratios_to)[instance], 'Enrichment']
            i += 1

        if base[instance] == 0 and np.sum(np.array(counts[instance]) <= (ratios_to)[instance]) <= (
                num_test * alpha / correction_num):
            print('Depletion of {} in FP from {:.3f} to {:.3f}.'.format(
                instance, (ratios_from)[instance], (ratios_to)[instance]))
            report.loc[i] = [test_col, instance, (ratios_from)[instance],
                             (ratios_to)[instance], 'Depletion']
            i += 1

    return report




