for YEAR in 2016_pre 2016_post 2017 2018
do 
    cp configs/preprocess/training_data/bbH_tt_$YEAR.yaml configs/preprocess/training_data/bbH_tt.yaml
    python preprocess.py --config-path configs/preprocess/training_data --config-name bbH_tt.yaml year=$YEAR
    MY_EXPERIMENT_NAME=TrainingTOP$YEAR
    mlflow run --experiment-name $MY_EXPERIMENT_NAME -P year=$YEAR -P num_iterations=500 -P n_splits=2 --no-conda .
done
