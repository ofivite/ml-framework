import os
import yaml
import gc
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig

import ROOT as R
import uproot
from lumin.nn.data.fold_yielder import FoldYielder
import pandas as pd
import mlflow

from utils.processing import fill_placeholders
from utils.inference import load_models, predict_folds

@hydra.main(config_path="configs/predict")
def main(cfg: DictConfig) -> None:
    # fill placeholders in the cfg parameters
    input_path = to_absolute_path(cfg.input_path)
    run_folder = to_absolute_path(f'mlruns/{cfg.experiment_id}/{cfg.run_id}/')
    models, n_splits, xtrain_split_feature = load_models(run_folder)

    # extract names of training features from mlflow-stored model
    # note: not checking that the set of features is the same across models
    misc_features = OmegaConf.to_object(cfg.misc_features)
    train_features = []
    with open(to_absolute_path(f'{run_folder}/artifacts/model_0/MLmodel'), 'r') as f:
        model_cfg = yaml.safe_load(f)
    for s in model_cfg['signature']['inputs'].split('{')[1:]:
        train_features.append(s.split("\"")[3])
    fold_id_column = 'fold_id'

    mlflow.set_tracking_uri(f"file://{to_absolute_path('mlruns')}")
    with mlflow.start_run(experiment_id=cfg.experiment_id, run_id=cfg.run_id):
        # loop over input fold files
        for sample_name in cfg.sample_names:
            print(f'\n--> Predicting {sample_name}')
            print(f"        loading data set")
            input_filename = fill_placeholders(cfg.input_filename_template, {'{sample_name}': sample_name})

            # extract DataFrame from fold file
            fy = FoldYielder(f'{input_path}/{input_filename}')
            df = fy.get_df(inc_inputs=True, deprocess=False, nan_to_num=False, verbose=False, suppress_warn=True)
            for f in misc_features: # add misc features
                if f not in df.columns:
                    if f in fy.columns():
                        df[f] = fy.get_column(f)
                    else:
                        raise KeyError(f'Couldn\'t find {f} neither in DataFrame nor in FoldYielder columns')
            df[fold_id_column] = (df[xtrain_split_feature] % n_splits).astype('int32')

            # run cross-inference for folds
            pred_dict = predict_folds(df, train_features, misc_features, fold_id_column=fold_id_column, models=models)

            print(f"        storing to output file")
            output_filename = fill_placeholders(cfg.output_filename_template, {'{sample_name}': sample_name})
            if cfg.kind == 'for_datacards':
                output_path = to_absolute_path(cfg.output_path)
                os.makedirs(output_path, exist_ok=True)
                if os.path.exists(f'{output_path}/{output_filename}'):
                    os.system(f'rm {output_path}/{output_filename}')
                
                # extract original index
                orig_filename = fill_placeholders(to_absolute_path(f'{cfg.orig_path}/{cfg.orig_filename_template}'), {'{sample_name}': sample_name})
                with uproot.open(orig_filename) as f:
                    t = f['TauCheck']
                    orig_index = t.arrays(['evt', 'run'], library='pd')
                    orig_index = list(map(tuple, orig_index.values))
                
                # reorder entries to match original indices
                df_pred = pd.DataFrame(pred_dict).set_index(['evt', 'run'])
                df_pred = df_pred.loc[orig_index].reset_index()
                pred_dict = {c: df_pred[c].values for c in df_pred.columns}

                # store predictions in RDataFrame and snapshot it into output ROOT file
                R_df = R.RDF.MakeNumpyDataFrame(pred_dict)
                R_df.Snapshot(cfg.output_tree_name, f'{output_path}/{output_filename}')
                del(df, R_df); gc.collect()
            elif cfg.kind == 'for_evaluation':
                df_pred = pd.DataFrame(pred_dict)
                df_pred.to_csv(output_filename, index=False)
                mlflow.log_artifact(output_filename, artifact_path='pred')
                del(df_pred); os.remove(output_filename); gc.collect()
            else:
                raise Exception(f'Unknown kind for prediction: {cfg.kind}')
        print()
        
if __name__ == '__main__':
    main()
