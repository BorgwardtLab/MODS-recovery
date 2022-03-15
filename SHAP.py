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

    train_list, val_list, test_list, sc_list = pd.read_pickle('data/data_{}_10Folds_day{}.csv'.format(
            args.center, args.daytime))
    feature_dict = pd.read_pickle('data/selected_features.pkl')
    selected_features = feature_dict['features']
    features_to_scale = feature_dict['features_to_scale']

    models = pd.read_pickle('models/{}_{}_10FoldsCV_day{}.pkl'.format(
            args.center, args.modelname, args.daytime))

    # calculate SHAP values across folds
    X = pd.DataFrame()
    posi_shap = []
    nega_shap = []

    posi_shap_df = pd.DataFrame(columns=selected_features)
    for i in range(len(test_list)):
        model = models[i]
        masker = shap.maskers.Independent(data=val_list[i][selected_features])
        explainer = shap.LinearExplainer(model.best_estimator_, masker=masker)
        val_list[i] = val_list[i].reset_index(drop=True)
        shap_values = explainer.shap_values(val_list[i][selected_features])
        posi_shap.append(shap_values[1])
        nega_shap.append(shap_values[0])
        posi_shap_df.loc[i] = np.abs(shap_values[1]).mean(0)
        X = pd.concat([X, val_list[i][selected_features]])
    top_ten = posi_shap_df.mean(0).sort_values(ascending=False).iloc[:10].index

    rc = {'figure.figsize': (10, 14),
          'axes.facecolor': 'white',
          'axes.grid': True,
          'grid.color': '.8',
          'font.family': 'Times New Roman',
          'font.size': 30,
          "axes.labelsize": 55}

    # save mean SHAP values plot for top 10 variables
    plt.rcParams.update(rc)
    sns.set(font_scale=1.1)
    sns.set_style('whitegrid')
    sns.barplot(data=posi_shap_df[top_ten], color="c", orient='h')
    sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
    plt.title('Mean absolute Shapley value', y=-0.1, fontsize=20)
    plt.tight_layout()
    plt.savefig('plots/SHAP_{}_{}_10FoldsCV_day{}_top10_mean.png'.format(
            args.center, args.modelname, args.daytime), dpi=300)
    plt.close()

    # save mean SHAP values plot for all variables
    sns.barplot(data=posi_shap_df[posi_shap_df.mean(0).sort_values(ascending=False).index],
                color="c", orient='h')
    sns.despine(fig=None, ax=None, top=True, right=True, left=False, bottom=False, offset=None, trim=False)
    plt.title('Mean absolute Shapley value', y=-0.1, fontsize=15)
    plt.tight_layout()
    plt.savefig('plots/SHAP_{}_{}_10FoldsCV_day{}_All_mean.png'.format(
        args.center, args.modelname, args.daytime), dpi=300)
    plt.close()

    # save SHAP scatter plot for top 10 variables
    plt.rcParams["axes.labelsize"] = 88
    sns.set(font_scale = 2)
    shap.summary_plot(np.vstack(posi_shap), X, max_display=10, plot_size=(12, 14), show=False)
    plt.savefig('plots/SHAP_{}_{}_10FoldsCV_day{}_top10.png'.format(
        args.center, args.modelname, args.daytime), dpi=300)
    plt.close()

    # save SHAP scatter plot for all variables
    shap.summary_plot(np.vstack(posi_shap), X)
    plt.savefig('plots/SHAP_{}_{}_10FoldsCV_day{}_All.png'.format(
        args.center, args.modelname, args.daytime), dpi=300)
    plt.close()

if __name__ == '__main__':
    main()


