echo ${SHELL}
EXPERIMENTNAME=Weight8020
EXPERIMENTID=9
YEAR=2022
CONFIG_NAME=lowpT_Taus.yaml

for var in "$@"
do
    echo "Doing ${var}"
    echo "------------------------"
    echo " "

    python preprocess.py --config-path configs/preprocess/training_data --config-name $CONFIG_NAME year=$YEAR 'class_weights={0: 1, 1: $var}'
    mlflow run --experiment-name $EXPERIMENTNAME --no-conda . 2>out_by_$var.txt 
    d=`cat out_by_$var.txt | grep "=== Run (ID "`
    d=$(sed -n "s/^.*'\(.*\)'.*$/\1/p" <<< $d)
    python predict.py --config-name for_evaluation.yaml year=2022 experiment_id=$EXPERIMENTID run_id=$d
    python evaluate.py experiment_id=$EXPERIMENTID run_id=$d dataset=test
    python evaluate.py experiment_id=$EXPERIMENTID run_id=$d dataset=train
done
