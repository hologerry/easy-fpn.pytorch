BACKBONE=$1
OUTPUTS_DIR=$2
if ! ([[ -n "${BACKBONE}" ]] && [[ -n "${OUTPUTS_DIR}" ]]); then
    echo "Argument BACKBONE or OUTPUTS_DIR is missing"
    exit
fi

python train.py -s=receipt -b=${BACKBONE} -o=${OUTPUTS_DIR} --learning_rate=0.00125 --weight_decay=0.0001 --image_min_side=800 --image_max_side=1333 --num_steps_to_snapshot=3000 --step_lr_sizes="[60000, 120000]" --num_steps_to_finish=140000