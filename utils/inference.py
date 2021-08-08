from glob import glob
import numpy as np
from sklearn.model_selection import LeaveOneGroupOut
from mlflow.pyfunc import load_model

def load_models(run_folder):
    # extract feature and number of splits used in LeaveOneGroupOut() during the training
    with open(f'{run_folder}/params/xtrain_split_feature', 'r') as f:
        xtrain_split_feature = f.read()
    with open(f'{run_folder}/params/n_splits', 'r') as f:
        n_splits = f.read()
    if not (n_splits.isdigit() and int(n_splits)>=1):
        raise Exception(f'n_splits should be integer and >= 1, got {n_splits}')
    n_splits = int(n_splits)
    print(f'\n[INFO] Fetched xtrain_split_feature ({xtrain_split_feature}) and number of splits ({n_splits})')

    # check that there are as many models logged as needed for retrieved n_splits
    model_idx = {int(s.split('/')[-1].split('model_')[-1]) for s in glob(f'{run_folder}/artifacts/model_*')}
    if model_idx != set(range(n_splits)):
        raise Exception(f'Indices of models in {run_folder}/artifacts ({model_idx}) doesn\'t correspond to the indices of splits used during the training ({set(range(n_splits))})')

    # load mlflow logged models for all folds
    print(f'\n--> Loading models')
    models = [load_model(f'{run_folder}/artifacts/model_{i_fold}') for i_fold in range(n_splits)]
    return models, n_splits, xtrain_split_feature

def predict_folds(df, train_features, misc_features, fold_id_column, models):
    # init structure to be written into output file
    pred_dict = {
                 'pred_class': [],
                 'pred_class_proba': [],
                 **{misc_feature: [] for misc_feature in misc_features}
                 }
    if (n_groups:=len(set(df[fold_id_column]))) != (n_splits:=len(models)):
        raise Exception(f'Number of fold groups in the input DataFrame ({n_groups}) \
                                    is not equal to the number of splits infered from the number of models ({n_splits}).')
    if n_splits > 1: # perform cross-inference
        splitter = LeaveOneGroupOut()
        idx_yielder = splitter.split(df, groups=df[fold_id_column])
        # split into folds and get predictions for each with corresponding model
        for i_fold, (_, pred_idx) in enumerate(idx_yielder): # loop over splits
            df_fold = df.iloc[pred_idx]

            # check that `i_fold` is the same as fold ID corresponding to each fold split
            fold_idx = set(df_fold[fold_id_column])
            assert len(fold_idx)==1 and i_fold in fold_idx

            # make predictions
            print(f"        predicting fold {i_fold}")
            y_proba = models[i_fold].predict(df_fold[train_features])
            pred_dict['pred_class'].append(np.argmax(y_proba, axis=-1).astype(np.int32))
            pred_dict['pred_class_proba'].append(np.max(y_proba, axis=-1).astype(np.float32))
            [pred_dict[f].append(df_fold[f].to_numpy()) for f in misc_features]

        # concatenate folds together
        pred_dict = {k: np.concatenate(v) for k,v in pred_dict.items()}

    elif n_splits == 1: # simply make predictions for a given dataframe
        y_proba = models[0].predict(df[train_features])
        pred_dict['pred_class'] = np.argmax(y_proba, axis=-1).astype(np.int32)
        pred_dict['pred_class_proba'] = np.max(y_proba, axis=-1).astype(np.float32)
        for f in misc_features:
            pred_dict[f] = df[f].to_numpy()
    else:
        raise ValueError(f'n_splits should be positive integer, got {n_splits}')

    return pred_dict
