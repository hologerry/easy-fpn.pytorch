BACKBONE=$1
CHECKPOINT=$2
if ! ([[ -n "${BACKBONE}" ]] && [[ -n "${CHECKPOINT}" ]]); then
    echo "Argument BACKBONE or CHECKPOINT is missing"
    exit
fi

python eval.py -s=receipt -b=${BACKBONE} --image_min_side=800 --image_max_side=1333 --rpn_post_nms_top_n=1000 ${CHECKPOINT}