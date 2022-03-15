import argparse
import warnings
warnings.filterwarnings('ignore')
from utils import *

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--center',
                        required=True,
                        type=str,
                        choices=['SPSS','LCHC']
                        )
    parser.add_argument('--modelname',
                        required=True,
                        type=str,
                        choices=['RF','LR','LGBM','MLP','SVM']
                        )
    parser.add_argument('--daytime',
                        required=True,
                        type=int,
                        choices=[1,2,3,4,5,6]
                        )
    args = parser.parse_args()
    train_list, val_list, test_list, _ = pd.read_pickle('data/data_{}_10Folds_day{}.csv'.format(
            args.center, args.daytime))
    feature_dict = pd.read_pickle('data/selected_features.pkl')
    selected_features = feature_dict['features']

    models = pd.read_pickle('models/{}_{}_10FoldsCV_day{}.pkl'.format(
            args.center, args.modelname, args.daytime))

    # the analysis is done when recall is fixed at 80%
    pred_list = []
    label_list = []
    for i in range(len(test_list)):
        pred_list.extend(models[i].best_estimator_.predict_proba(test_list[i][selected_features])[:, 1])
        label_list.extend(test_list[i]['label'])
    precisions, recalls, thresholds = precision_recall_curve(label_list, pred_list)
    threshold = thresholds[np.abs(recalls - 0.8).argmin()]

    FP = []
    FN = []
    TP = []
    TN = []
    for i in range(len(test_list)):
        model = models[i]
        test = test_list[i].set_index('sample.id')[selected_features]
        label = test_list[i].set_index('sample.id')['label']
        pred = (model.predict_proba(test)[:, 1] >= threshold).astype(int)

        X_test = pd.DataFrame(pred, index=test.index, columns=['predict'])
        X_test['true'] = label.values
        X_test['cate'] = 'NAN'
        for j in range(len(X_test)):
            if X_test.iloc[j, X_test.columns.get_loc("true")] == 1 and X_test.iloc[
                j, X_test.columns.get_loc("predict")] == 1:
                X_test.iloc[j, X_test.columns.get_loc("cate")] = 'TP'
            elif X_test.iloc[j, X_test.columns.get_loc("true")] == 0 and X_test.iloc[
                j, X_test.columns.get_loc("predict")] == 1:
                X_test.iloc[j, X_test.columns.get_loc("cate")] = 'FP'
            elif X_test.iloc[j, X_test.columns.get_loc("true")] == 0 and X_test.iloc[
                j, X_test.columns.get_loc("predict")] == 0:
                X_test.iloc[j, X_test.columns.get_loc("cate")] = 'TN'
            elif X_test.iloc[j, X_test.columns.get_loc("true")] == 1 and X_test.iloc[
                j, X_test.columns.get_loc("predict")] == 0:
                X_test.iloc[j, X_test.columns.get_loc("cate")] = 'FN'
        cols_to_test = ['pathogen.grp', 'age.grp', 'ethnicity', 'sex',
                'focus.grp', 'category', 'hosp.los.bc', 'picu.los.bc']
        testSet = test_list[i].set_index('sample.id')[cols_to_test]
        if X_test[X_test['cate'] == 'FP'].shape[0] != 0:
            FP.append(testSet[X_test['cate'] == 'FP'])
        if X_test[X_test['cate'] == 'FN'].shape[0] != 0:
            FN.append(testSet[X_test['cate'] == 'FN'])
        if X_test[X_test['cate'] == 'TP'].shape[0] != 0:
            TP.append(testSet[X_test['cate'] == 'TP'])
        if X_test[X_test['cate'] == 'TN'].shape[0] != 0:
            TN.append(testSet[X_test['cate'] == 'TN'])

    FPs = pd.concat(FP)
    TPs = pd.concat(TP)
    FNs = pd.concat(FN)
    TNs = pd.concat(TN)

    GPs = pd.concat([TPs, FNs])
    GNs = pd.concat([TNs, FPs])

    for From, To, name in zip([GNs, GPs], [FPs, FNs], ['GNsToFPs', 'GPsToFNs']):
        for target in ['pathogen.grp', 'age.grp', 'focus.grp']:
            report = enrichment_test(From, To, target, 0.05)
            report.to_csv('reports/Enrichment_{}.csv'.format(name))




if __name__ == '__main__':
    main()


