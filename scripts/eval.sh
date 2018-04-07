export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1
export MXNET_ENABLE_GPU_P2P=0
# export PYTHONPATH=${PYTHONPATH}:incubator-mxnet/python/

TRAIN_DIR=/mnt/models/
DATASET=coco
SET=train
TEST_SET=val2017

# Test
python3.6 eval_maskrcnn.py \
    --network resnet_fpn \
    --has_rpn \
    --dataset ${DATASET} \
    --image_set ${TEST_SET} \
    --prefix ${TRAIN_DIR}final \
    --result_path /mnt/models/v1 \
    --epoch 0 \
    --gpu 0

# python3.6 data/cityscape/cityscapesscripts/evaluation/evalInstanceLevelSemanticLabeling.py
