import argparse
import warnings
warnings.filterwarnings('ignore')
import os
import pickle
from utils import *

model_dict = {}
model_dict['RF'] = RF_grid
model_dict['LR'] = LR_grid
model_dict['SVM'] = SVM_grid
model_dict['LGBM'] = LGBM_grid
model_dict['MLP'] = MLP_grid

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
    # loading features to use and model
    feature_dict = pd.read_pickle('data/selected_features.pkl')
    selected_features = feature_dict['features']
    features_to_scale = feature_dict['features_to_scale']
    model = model_dict[args.modelname]
    report = pd.DataFrame(columns=['auprc_val', 'auroc_val', 'auprc_test', 'auroc_test'])
    # loading data splits
    train_list, val_list, test_list, _ = pd.read_pickle('data/data_{}_10Folds_day{}.csv'.format(
        args.center, args.daytime))
    # crsoss validation
    auprc_val, auroc_val, auprc_test, auroc_test, models, predictions_test, labels_test = cross_validation(
        train_list, val_list, test_list, selected_features, features_to_scale, model)

    # saving results
    report.loc[args.modelname] = [auprc_val, auroc_val, auprc_test, auroc_test]

    with open('models/{}_{}_10FoldsCV_day{}.pkl'.format(
            args.center, args.modelname, args.daytime), 'wb') as fp:
        pickle.dump(models, fp)

    if not os.path.exists('reports/perf_{}_10FoldsCV.csv'.format(args.center)):
        report.to_csv('reports/perf_{}_10FoldsCV.csv'.format(args.center))
    else:
        report.to_csv('reports/perf_{}_10FoldsCV.csv'.format(args.center), mode='a', header=False)
    plotname = '{}_{}_10FoldsCV_day{}'.format(args.center, args.modelname, args.daytime)
    plot_aucs_withUncertainty([labels_test, predictions_test], plotname)

if __name__ == '__main__':
    main()


