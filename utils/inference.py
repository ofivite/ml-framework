import numpy as np
from sklearn.model_selection import LeaveOneGroupOut

def predict_folds(df, train_features, misc_features, fold_id_column, models):
    # init structure to be written into output file
    pred_dict = {
                 'pred_class': [],
                 'pred_class_proba': [],
                 **{misc_feature: [] for misc_feature in misc_features}
                 }
    if (n_groups:=len(set(df[fold_id_column]))) != len(models):
        raise Exception(f'Number of groups in the input DataFrame ({n_groups}) is not equal to the number of models ({len(models)}).')

    # split into folds and get predictions for each with corresponding model
    logo = LeaveOneGroupOut()
    for i_fold, (_, pred_idx) in enumerate(logo.split(df, groups=df[fold_id_column])): # loop over splits
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
    return pred_dict
