BACKBONE=$1
CHECKPOINT=$2
INPUT_DIR=$3
OUTPUT_DIR=$4
if ! ([[ -n "${BACKBONE}" ]] && [[ -n "${CHECKPOINT}" ]] && [[ -n "${INPUT_DIR}" ]] && [[ -n "${OUTPUT_DIR}" ]]); then
    echo "Argument BACKBONE or CHECKPOINT or INPUT_DIR or OUTPUT_DIR is missing"
    exit
fi

python infer.py -s=receipt -b=${BACKBONE} -c=${CHECKPOINT} --image_min_side=800 --image_max_side=1333 --rpn_post_nms_top_n=1000 ${INPUT_DIR} ${OUTPUT_DIR}