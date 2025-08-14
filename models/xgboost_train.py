import pandas as pd
from pathlib import Path
from collections import defaultdict

from utils.data_utils import get_data_df

import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score

def search_best_params(data_path='data', data_type = 'train'):
    """
    Grid search over parameters to find the best set.
    """
    data_path = Path(data_path)
    embeddings_file = f'x_{data_type}_nlp_embeddings.csv'
    embeddings_file_path = data_path/data_type/embeddings_file

    # combine embeddings with other features for modeling
    embeddings_df = pd.read_csv(embeddings_file_path).drop(columns=['Unnamed: 0'])
    data_df = get_data_df(data_type='train')
    data_df = data_df.merge(embeddings_df, on='UniqueID', how='inner')
    data_df.set_index('UniqueID', inplace=True)
    data_df.drop(columns=['nlp_feature_vector'], inplace=True)

    x = data_df[data_df.columns[~data_df.columns.isin(['Target'])]]
    y = data_df['Target']
    
    x_train, x_test, y_train, y_test = train_test_split(
        x, y,
        test_size=0.2,
        random_state=200,
        shuffle=True,
        stratify=y
    )
    
    #store metrics and params then search over them to find best ones
    metrics = defaultdict(list)
    grid = {
        'subsamples': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'n_estimators': [10, 20, 30, 40, 50, 60, 70, 80, 90, 100],
        'learning_rates': [1e-1, 2e-1, 3e-1, 1e-2, 2e-2, 3e-2, 1e-3, 2e-3, 3e-3, 1e-4],
        'colsample_bytree': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0],
        'depth': [4, 6, 8, 10, 12, 14, 16, 18, 20, 22]
    }
    for subsample in grid['subsamples']:
        for n_estimators in grid['n_estimators']:
            for learning_rate in grid['learning_rates']:
                for colsample_bytree in grid['colsample_bytree']:
                    for depth in grid['depth']:
                        model = xgb.XGBClassifier(
                            objective='binary:logistic',
                            eval_metric='auc',
                            n_estimators=n_estimators,
                            max_depth=depth,
                            learning_rate=learning_rate,
                            subsample=subsample,
                            colsample_bytree=colsample_bytree,
                            tree_method='hist',   # faster training
                            random_state=42
                        )

                        #training
                        model.fit(x_train, y_train)

                        # predict from the trained models
                        y_pred_train = model.predict(x_train)
                        y_pred = model.predict(x_test)

                        # evaluation
                        metrics['n_estimators'].append(n_estimators)
                        metrics['subsample'].append(subsample)
                        metrics['learning_rate'].append(learning_rate)
                        metrics['colsample_bytree'].append(colsample_bytree)
                        metrics['depth'].append(depth)
                        metrics['train_accuracy'].append(accuracy_score(y_train, y_pred_train))
                        metrics['test_accuracy'].append(accuracy_score(y_test, y_pred))
                        metrics['train_roc_auc_score'].append(roc_auc_score(y_train, y_pred_train))
                        metrics['test_roc_auc_score'].append(roc_auc_score(y_test, y_pred))
                        print(f"train accuracy: {accuracy_score(y_train, y_pred_train)} | test accuracy: {accuracy_score(y_test, y_pred)} | test auc: {roc_auc_score(y_test, y_pred)} | depth: {depth} | n_estimators: {n_estimators} | subsamples: {subsample} | learning rate: {learning_rate} | colsample_bytree: {colsample_bytree}")
    
    # find the best parameters by auc_score
    metrics_df = pd.DataFrame(metrics)
    best_params = metrics_df.iloc[metrics_df[['test_roc_auc_score']].idxmax()]
    best_params
    return metrics, 
                        