import yaml
import hydra
from hydra.utils import to_absolute_path
from omegaconf import OmegaConf, DictConfig

import ROOT as R
import uproot
from lumin.nn.data.fold_yielder import FoldYielder

import numpy as np
from mlflow.pyfunc import load_model

def fill_placeholders(string, placeholder_to_value):
    for placeholder, value in placeholder_to_value.items():
        string = string.replace(placeholder, str(value))
    return string

@hydra.main(config_path="configs", config_name="predict_cfg")
def main(cfg: DictConfig) -> None:

    # file_name = fill_placeholders(cfg.file_name_template, {'{sample_name}': cfg.sample_name, '{year}': cfg.year})
    # f = uproot.open(f'{input_path}/{file_name}')
    # t = f[cfg.tree_name]
    # data = t.arrays(train_features + misc_features, library='pd')
    # f.close()
    # del(t)
    #
    # # some preprocessing
    # data.replace([np.inf, -np.inf], np.nan, inplace=True) # lumin handles nans automatically
    # data['njets'] = data.njets.clip(0, 5)
    #
    # # apply normalisation
    # with open(cfg.input_pipe, 'rb') as fin:
    #     input_pipe = pickle.load(fin)
    # data[cont_features] = input_pipe.transform(data[cont_features])
    # cat_maps, cat_szs = proc_cats(data, cat_features)
    #
    # # dummy target to be able to save into the fold
    # data[_target] = -1
    #
    # # store into a hdf5 fold file
    # df2foldfile(df=data,
    #             n_folds=10,
    #             cont_feats=cont_features,
    #             cat_feats=cat_features, cat_maps=cat_maps,
    #             targ_feats=_target, targ_type=int,
    #             misc_feats=misc_features,
    #             savename=f'data/2018/multi/pred/{sample_name}'
    #             )

    input_path = to_absolute_path(fill_placeholders(cfg.input_path, {'{year}': cfg.year}))
    output_path = to_absolute_path(fill_placeholders(cfg.output_path, {'{year}': cfg.year}))
    model_path = to_absolute_path(f'mlruns/{cfg.mlflow_experimentID}/{cfg.mlflow_runID}/artifacts/model')
    input_pipe = to_absolute_path(fill_placeholders(cfg.input_pipe, {'{year}': cfg.year}))

    # extract names of training features from mlflow-stored model
    misc_features = OmegaConf.to_object(cfg.misc_features)
    train_features = []
    with open(to_absolute_path(f'{model_path}/MLmodel'), 'r') as f:
        model_cfg = yaml.safe_load(f)
    for s in model_cfg['signature']['inputs'].split('{')[1:]:
        train_features.append(s.split("\"")[3])

    for sample_name in cfg.sample_names:
        # extract DataFrame from fold file
        fy = FoldYielder(f'{input_path}/{sample_name}.hdf5', input_pipe=input_pipe)
        df = fy.get_df(inc_inputs=True, deprocess=False)
        for f in misc_features:
            df[f] = fy.get_column(f)

        # load mlflow logged model
        model = load_model(model_path)

        # make predictions
        y_proba = model.predict(df[train_features])
        y_pred_class = np.argmax(y_proba, axis=1).astype(np.int32)
        y_pred_class_proba = np.max(y_proba, axis=1).astype(np.float32)
#         if 'evt' not in df:
#             raise Exception('"evt" (or other unique ID variable) should be present in the input file')
#         else:
#             df['evt'] = df['evt'].to_numpy().astype(np.int32)

        # write predictions into RDataFrame and snapshot it into output ROOT file
        output_filename = fill_placeholders(cfg.output_filename_template, {'{sample_name}': sample_name, '{year}': cfg.year})
        R_df = R.RDF.MakeNumpyDataFrame({'pred_class': y_pred_class,
                                         'pred_class_proba': y_pred_class_proba,
                                          **{misc_feature: df[misc_feature].to_numpy() for misc_feature in misc_features}
                                         })
        R_df.Snapshot(cfg.output_tree_name, f'{output_path}/{output_filename}')

if __name__ == '__main__':
    main()
