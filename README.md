# ML framework for HTT analyses
**Foreword:** this README contains practical instructions on how to setup & run the framework. For a conceptual overview of the workflow and a general idea behind its structure please have a look at [this presentation](https://drive.google.com/file/d/197bM--JW-mwuppMup6fDNFXf_wgbSvZB/view?usp=sharing). However, since the framework is being constantly developed, it is the README that will contain the most up-to-date information.

## Environment setup
It is recommended to run the code or install complementary packages from within a dedicated [conda](https://www.anaconda.com) environment, which contains all the necessary packages and can be set up from a `conda.yaml` file.

1) Firstly, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) following [these instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). For example, for Linux x86_64:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Following the instructions one will be asked to specify the directory where Miniconda will be installed. In case of using the framework at `naf`, **make sure to specify the path in nfs area**: e.g. `/nfs/dust/cms/user/${USERNAME}/miniconda3`. This is needed to ensure that sufficient space is available to download python packages. Also, type _yes_ once asked for initialisation of conda and if there is no need for the conda `base` environment to be activated on startup, execute `conda config --set auto_activate_base false`. Then, relaunch the shell/relogin for the installation to take effect.

2) After that, clone the repo and install the conda environment (note: this might take some time):
```bash
git clone https://github.com/yaourtpourtoi/ml-framework.git
cd ml-framework
conda env create -f conda.yaml
```

3) Once the conda environment is created, it should be activated/deactivated with:
```bash
ENV_NAME=htt-ml
conda activate ${ENV_NAME}
# conda deactivate
```

4) At this point, one might want to register this environment in Jupyter as a kernel. To do that, execute inside of the environment the following command:
```bash
ENV_NAME=htt-ml
ipython kernel install --user --name=${ENV_NAME}
```
The newly created kernel can now be activated in the Jupyter selection menu (top-right corner in the Jupyter tab)

**NB:** other initialisations of environment variables (especially those of $PATH) might negatively interfere with conda. One example would be the command `cmsenv` or sourcing of CMSSW-related initialisation scripts. If something doesn't work with packages inside the conda environment, check that there is no such conflict with other initialisation procedures (e.g., happening automatically in `~/.bashrc` or `~/.zshrc`).   

**NB:** For correct `plotly` rendering in JupyterLab, [check](https://plotly.com/python/troubleshooting/#jupyterlab-problems) that extensions are enabled in Extension Manager (located in the corresponding tab on the left panel of the JupyterLab window) and `jupyterlab-plotly` is displayed amongst them.

## Data preprocessing
As the very first step, input ROOT files are preprocessed and skimmed within the framework prior to training. This is done with `preprocess.py` script, which combines the input set of ROOT files (also referred to as _nodes_) into a single pandas dataframe, then performs necessary transformations/additions/filtering, then splits the dataframe into the output nodes and stores each into `hdf5` foldfile using `df2foldfile` function of [`lumin`](https://lumin.readthedocs.io/en/stable/) library. Conceptually, the preprocessing stage can be viewed as a _dataflow_, where input source files are firstly merged into a single stream which then flows through a series of transformations and splits towards its end, which would be a storage into `hdf5` foldfiles.

There are two paths in this preprocessing dataflow: the one to skim the data set for training and the other to skim it to make predictions for final statistical inference. This is done by the same `preprocessing.py` script and the flag `for_training` in the input cfg file indicates the path to be followed within the script. Therefore, there are two configs which define what and how needs to be skimmed, each corresponding to either training or prediction paths: `configs/preprocess/training_data.yaml` and  `configs/preprocess/prediction_data.yaml` respectively. These configs are passed to `preprocessing.py` with [`hydra`](https://hydra.cc/docs/intro), see its documentation for more details on the usage.

For example, in order to prepare the data for training on 2018 data one needs to execute (note that there is no need to specify in the command line prepath `configs/preprocess`):

```bash
python preprocess.py --config-name training_data.yaml year=2018
# python preprocess.py --config-name prediction_data.yaml year=2018 ### preprocessing of data for prediction
```

This will produce out of the input ROOT files in `input_path` folder `hdf5` skims in the `output_path`, which can be further passed to a model of one's choice.

**NB:** generally speaking, it is the user's responsibility to implement preprocessing which is appropriate for their analysis. Current implementation includes simple checks for NaN/inf values (without their preprocessing), [categorical feature treatment](https://lumin.readthedocs.io/en/stable/lumin.data_processing.html#lumin.data_processing.pre_proc.proc_cats), scaling with [`input_pipe`](https://lumin.readthedocs.io/en/stable/lumin.data_processing.html#lumin.data_processing.pre_proc.fit_input_pipe) and clipping of `njets`.

## Model training
To track the model training [`mlflow`](https://mlflow.org/docs/latest/index.html) project has been set up, see its description in `MLproject` file. There is currently one entry points _main_ (multiclass classification problem, default) which runs `python train.py` with necessary parameters from `configs/train/train.yaml` added/overriden. There is `hydra` also used under the hood to parse those parameters from the `yaml` cfg file.  

Internally, [`FoldYielder`](https://lumin.readthedocs.io/en/stable/core_concepts.html#reading-fold-files) class of `lumin` is used to extract pandas DataFrames from input foldfiles and pass it on to the model. At the moment, only gradient boosting on trees (aka BDT) with [`lightgbm`](https://lightgbm.readthedocs.io/en/latest/) is implemented to solve the classification problem. More models and architectures are to be interfaced with the framework in the nearest future.

To train and track the model create an experiment (unless already done) and run it with `mlflow` specifying:
*  a corresponding entry point (with `-e` option, defaults to `main`)
*  name of the experiment for the run to be assigned to (`--experiment-name $MY_EXPERIMENT_NAME`)
*  `--no-conda` to avoid creating new conda environment and at the runtime
*  mlflow params with their values (e.g. `-P num_iterations=5` or `-P n_splits=2`, optional, see `MLproject` for all of them and their default values)
*  project directory (`.` - current directory)

```bash
MY_EXPERIMENT_NAME=test
mlflow run --experiment-name $MY_EXPERIMENT_NAME -P year=2018 -P num_iterations=5 -P n_splits=2 --no-conda .
```

*Note*: Oppositely to the manual installation, running `mlflow run` without `--no-conda` flag automatically creates a conda environment from `conda.yaml` cfg file and runs the code from there.

`mlflow` takes care of logging and saving all the basic information about the training, including the model and optional metrics/artifacts (if specified in `train.py`) and **assigns each model to a unique mlflow `run_ID`**. This is by default logged into `mlruns/{experiment_ID}/{run_ID}` folder inside of the framework directory. Please **remember both `experiment_ID`** which was assigned internally to a new `$MY_EXPERIMENT_NAME` **and `run_ID`**. The former should be an integer number, which is needed as one of the input cfg parameters in the following modules of the framework. The latter is outputted in the terminal once the training is done, or alternatively, it is always possible to fetch it from `mlflow ui` (see Tracking section below).

It is important to note that the training is implemented in **N-fold manner** (also referred to as *cross-training*). The input dataset will be split into `n_splits` folds (as defined in the training cfg file) and `n_splits` models will be trained, where `model_{i}` uses `fold_{i}` only for metric validation during the training, not for the training itself. The folds are indexed based on the remainder of division of `xtrain_split_feature` column in the input data set by `n_splits`. In case when `n_splits=1` is set, only one model will be trained on `train_size` fraction of the input data set, while the rest of it will be used for validation of loss/metrics during the training.   

## Tracking results
Once the training is done, `mlflow` provides a [UI interface](https://www.mlflow.org/docs/latest/tracking.html#tracking-ui) to inspect and compare the logged results across experiments/runs. Firstly, in case of running the code on a remote machine, find out its hostname with:

```bash
echo ${USERNAME}@${HOSTNAME}
```

Then, to run `mlflow` UI execute **from the directory which contains `mlruns` folder** the command (`-p` specifies the port id, default is 5000):
```bash
REMOTE_PORT_ID=5000
mlflow ui -p ${REMOTE_PORT_ID}
```

In case of running on a remote machine and not being able to open a browser as a GUI there, one can listen to a remote server via `ssh` and interact with it on a local machine (e.g. personal laptop). For that, run the commands below on a local machine to make an ssh tunnel and forward remote port to a local one (insert the output of the `echo ${USERNAME}@${HOSTNAME}` command from remote in `SERVER` variable definition):
```bash
REMOTE_PORT_ID=5000
LOCAL_PORT_ID=5010
# SERVER=...
ssh -N -f -L localhost:${LOCAL_PORT_ID}:localhost:${REMOTE_PORT_ID} ${SERVER}
```

Then one can access `mlflow` UI locally by going to http://localhost:5010 in a browser (here, `5010` is a local port id taken from a code snippet example).

**NB:** it is important to **close all the ports** after their usage is finished, otherwise they will likely hang there forever. This implies executing `Ctrl+C` to close the server with `mlflow ui`, and then also manually closing the ssh tunnels locally (i.e. on one's laptop). Depending on the OS, it can be done either via system process manager, or in the terminal with firstly finding the corresponding processes with `ps aux | grep ssh`, and then killing it with `kill <process_id>`. If you see that on the remote machine you can't create a port because of the error "Connection in use" (likely because it wasn't closed before), you can find the corresponding process ID with `ps aux | grep mlflow` and kill it with `kill <process_ID>`.   

## Making predictions
Given the trained model, one can now produce predictions for further inference for the given set of `hdf5` files (skimmed by `preprocess.py`). This is performed with `predict.py` script which loads the model(s) with `mlflow` given the corresponding `experiment_ID` and `run_ID`, opens each of the input fold files with `FoldYielder` and passes the data to the model(s). 

Prediction workflow is also implemented to be in N-fold fashion, which should be transparent to the user similarly to the training step. The number of splits is infered from `mlflow` logs for the corresponding run ID, so that the strategy of the prediction split is automatically adapted to the strategy of the training split. That is, conceptually only `mlflow_experimentID`/`mlflow_runID` and path to input data is needed to produce predictions and store them to the output files. Variables stored are (see `utils/inference.py` for details) `pred_class_{i}_proba` (predicted probability of i-th class), `pred_class` (argmax of output nodes), `pred_class_proba` (probability of `pred_class`) and additionally `misc_features` as those specified in the cfg file.     

There are two possible outputs (each configured with its own cfg file) which can be created at the prediction step. One is of the kind `for_datacards` and the other is `for_evaluation`. For both of them the predictions are produced in the same way, but they are saved to different file formats. For example, in case of option `for_datacards`:

```bash
python predict.py --config-name for_datacards.yaml year=2018 mlflow_experimentID=None mlflow_runID=None # insert the corresponding experiment/run ID here
```

For a given model from a corresponding mlflow run it will produce in `output_path` ROOT files one per `sample_name` with predictions saved therein to a TTree named `output_tree_name`. To do that, [`RDataFrame`](https://root.cern/doc/master/classROOT_1_1RDataFrame.html) class is used to snapshot a python dictionary with prediction arrays into ROOT files. After that they can be used in the next steps of the analysis, e.g. in order to produce datacards. Using [`TTree` friends](https://root.cern.ch/root/htmldoc/guides/users-guide/Trees.html#example-3-adding-friends-to-trees) might be especially helpful in this case to augment the original input ROOT files with predictions added as a new branch (`evt` in the example below is used as a common index):  
```cpp
TFile *f = new TFile("file_pred.root","READ");
TFile *F = new TFile("file_main.root","READ");
TTree* t = (TTree*)f->Get("tree");
TTree* T = (TTree*)F->Get("tree");
t->BuildIndex("evt");
T->BuildIndex("evt");
T->AddFriend(t);
T->Scan("pred_class_proba:evt");
```

The second option `for_evaluation` is implemented in order to run the next step of estimating the model performance:
```bash
python predict.py --config-name for_evaluation.yaml year=2018 mlflow_experimentID=None mlflow_runID=None # insert the corresponding experiment/run ID here
```

Here, for a given input files (from `input_path`, as always preprocessed with `preprocess.py`) and a given training (from `mlflow_runID`) predictions will be logged into `.csv` files under exactly the same `mlflow_runID`. After that, simply referring to a single `mlflow_runID`, predictions will be fetched automatically from `mlflow` logs and a dashboard with various metrics and plots can be produced with `evaluate.py` script (WIP).

## Evaluating results

Given the trained model and corresponding predictions for train and test skims (produced with `for_evaluation`), one can be interested in evaluating the model's performance on these data sets. For that purpose, there is a dedicated `evaluate.py` script which is configured with a corresponding `configs/evaluate.yaml` cfg file. Basically, one needs to simply specify there usual `mlflow_experimentID`/`mlflow_runID` and the name of the `dataset` to be used for estimation ("train"/"test"). The latter will be fetched from corresponding `mlflow` run folder. That is, executing the following command:

```bash
python evaluate.py mlflow_experimentID=None mlflow_runID=None dataset=None # insert the values here
```

will compute, plot and finally log all the plots (both interactive `plotly` html and pdf files) and metrics under the corresponding `mlflow_runID`. Currently, these include plots of model's output distribution for true classes in each predicted category (as probability density), confusion matrix (weighted with class weights, normalised by true and predicted values), ROC and precision-recall (PR) curves (weighted with one-vs-all class weights). Corresponding plotting functions are defined in `utils/plotting.py`. 

After that, one will be able to inspect and compare them across various runs using `mlflow` UI (see Tracking section above). To do that, please click on the run of interest (under the column `Start Time`) in the main table with all runs and scroll down to a section `Artifacts` and head over to `plots` folder. 

Furthermore, in the `Metrics` section metric values, also computed inside of `evaluate.py` should appear. At the moment this includes area under ROC curve for each class (`roc_auc_*`), [average precision](https://scikit-learn.org/stable/modules/generated/sklearn.metrics.average_precision_score.html) (`pr_average_prec_*`) and confusion matrix elements (`cm`), all of them separately for train and test samples.