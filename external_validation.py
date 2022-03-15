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
    # loading data and construct label
    if args.center == 'LCHC':
        val_center = 'SPSS'
    elif args.cetner == 'SPSS':
        val_center = 'LCHC'
    data = pd.read_csv('data/data_day_{}.csv'.format(args.center))
    data = data.set_index('sample.id')
    data, _, _, _ = get_label(data, args.daytime)
    perf_80recall = pd.DataFrame(columns=['precision', 'specificity',
                                          'f1', 'npv', 'accuracy', 'recall'])

    print('Evaluating {} model'.format(args.modelname))
    model_dict = pd.read_pickle('models/validation_{}_{}.pkl'.format(val_center,
                                                                     args.modelname))

    feature_dict = pd.read_pickle('data/selected_features.pkl')
    selected_features = feature_dict['features']
    features_to_scale = feature_dict['features_to_scale']

    model = model_dict['model']
    sc = model_dict['scaler']

    data[features_to_scale] = sc.transform(data[features_to_scale])
    data[features_to_scale] = data[features_to_scale].fillna(0)
    X = data[selected_features]
    # saving results
    label = data['label']
    pred = model.predict_proba(X)[:, 1]
    plot_name = '{}_{}'.format(val_center, args.modelname)
    plot_aucs(label, pred, plot_name)
    metrics = metrics_80recall(model, X, label, pred)
    perf_80recall.loc[args.modelname] = metrics
    shap_plot(model, X, val_center, args.modelname)
    perf_80recall.to_csv('reports/performance_80recall_{}.csv'.format(args.center))

if __name__ == '__main__':
    main()


