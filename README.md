# ML framework for HTT analyses

## Environment setup
It is recommended to run the code or install complementary packages from within a dedicated [conda](https://www.anaconda.com) environment, which contains all the necessary packages and can be set up from a `conda.yaml` file.

1) Firstly, download and install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) following [these instructions](https://conda.io/projects/conda/en/latest/user-guide/install/index.html). For example, for Linux x86_64:
```bash
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Following the instructions one will be asked to specify the directory where Miniconda will be installed. In case of using the framework at `naf`, **make sure to specify the path in nfs area**: e.g. `/nfs/dust/cms/user/{USERNAME}/miniconda3`. This is needed to ensure that sufficient space is available to download python packages. Also, type _yes_ once asked for initialisation of conda and if there is no need for the conda `base` environment to be activated on startup,  execute `conda config --set auto_activate_base false`. Then, relaunch the shell/relogin for the installation to take effect.

2) After that, clone the repo and install the conda environment (note: this might take some time):
```bash
git clone https://github.com/yaourtpourtoi/ml-framework.git
cd ml-framework
conda env create -f conda.yaml
```

3) Once the conda environment is created, it can be activated/deactivated with:
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

There are two paths in this preprocessing dataflow: the one to skim the data set for training and the other to skim it to make predictions. This is done by the same `preprocessing.py` script and the flag `for_training` in the input cfg file indicates the path to be followed within the script. Therefore, there are two configs which define what and how needs to be skimmed, each corresponding to either training or prediction paths: `configs/preprocess/training_data.yaml` and  `configs/preprocess/prediction_data.yaml` respectively. These configs are passed to `preprocessing.py` with [`hydra`](https://hydra.cc/docs/intro), see its documentation for more details on the usage.

For example, in order to prepare the data for training on 2018 data one needs to execute:

```bash
python preprocess.py --config-name training_data.yaml year=2018
# python preprocess.py --config-name prediction_data.yaml year=2018 ### preprocessing of data for prediction
```

This will produce out of the input ROOT files in `input_path` folder `hdf5` skims in the `output_path`, which can be further passed to a model of one's choice.

## Model training
To track the model training [`mlflow`](https://mlflow.org/docs/latest/index.html) project has been set up, see its description in `MLproject` file. There is currently two entry points: _main_ (multiclass classification problem, default) and _binary_ (binary classification problem), where each runs `python train.py` with necessary parameters from `configs/train.yaml` added/overriden. There is `hydra` also used under the hood to parse those parameters.  

Internally, [`FoldYielder`](https://lumin.readthedocs.io/en/stable/core_concepts.html#reading-fold-files) class of `lumin` is used to extract pandas DataFrames from input foldfiles and pass it on to the model. At the moment, only gradient boosted on trees (aka BDT) with [`lightgbm`](https://lightgbm.readthedocs.io/en/latest/) is implemented to solve the classification problem. More models and architectures will be interfaced with the framework in the nearest future.

To train and track the model create an experiment (unless already done) and run it with `mlflow` specifying:
*  a corresponding entry point (with `-e` option, defaults to `main`)
*  name of the experiment for the run to be assigned to (`--experiment-name test`)
*  `--no-conda` to avoid creating new conda environment and running from there
*  mlflow params with their values (`-P num_iterations=5`, optional, see `MLproject` for all of them and their default values)
*  project directory (`.` - current)

```bash
mlflow run --experiment-name test -P year=2018 -P num_iterations=5 --no-conda .
```

`mlflow` takes care of logging and saving all the basic information about the training, including the model and optional metrics/artifacts (if specified in `train.py`). This is by default logged into `mlruns/{experiment_ID}/{run_ID}` folder inside of the framework directory.

_P.S.:_ Oppositely to the manual installation, running `mlflow run` without `--no-conda` flag automatically creates a conda environment from `conda.yaml` cfg file and runs the code from there.

## Tracking results
Once the training is done, `mlflow` provides a [UI interface](https://www.mlflow.org/docs/latest/tracking.html#tracking-ui) to inspect and compare the logged results across experiments/runs. Firstly, in case of running the code on a remote machine, find out its hostname with:

```bash
echo ${USERNAME}@${HOSTNAME}
```

Then, to run `mlflow` UI execute from the project directory the command (`-p` specifies the port id, default is 5000):
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

## Making predictions
Given the trained model, one can now produce predictions for further inference for the given set of `hdf5` files (skimmed by `preprocess.py`). This is performed with `predict.py` script which loads the model with `mlflow` given its `experiment_ID` and `run_ID`, opens each of the input fold files with `FoldYielder` and passes the data to the model. The output in the form of _maximum class probability_ and the _corresponding class_ along with `misc_features` is saved into the output ROOT file through an [`RDataFrame`](https://root.cern/doc/master/classROOT_1_1RDataFrame.html) class. Lastly, `predict.py` uses the configuration file `configs/predict.yaml` to fetch the necessary parameters, e.g. the list input files or `run_ID`. Note, that the default name of the config file is specified in `@hydra.main()` decorator inside of `predict.py` and not required to be passed in the command line. That is, to produce predictions corresponding to 2018 year, mlflow run "abcd" and other parameters from `configs/predict.yaml` as default, execute:

```bash
python predict.py year=2018 mlflow_runID=abcd # insert the corresponding run ID here
```

This will produce in `output_path` ROOT files with predictions, which can be now used in the next steps of the analysis. For example, using [`TTree` friends](https://root.cern.ch/root/htmldoc/guides/users-guide/Trees.html#example-3-adding-friends-to-trees) they can be easily augment the original input ROOT files as a new branch based on a common index variable (`evt` in the example below):  
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
