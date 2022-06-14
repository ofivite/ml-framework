echo ${SHELL}

for var in "$@"
do
    echo "Doing ${var}"
    echo "------------------------"
    echo " "
    sed -e "s;%BGWEIGHT%;$var;g" configs/preprocess/training_data/lowpT_Taus.txt > configs/preprocess/training_data/lowpT_Taus.yaml
    python preprocess.py --config-path configs/preprocess/training_data --config-name lowpT_Taus.yaml year=2022
    mlflow run --experiment-id 6 --no-conda .
done
