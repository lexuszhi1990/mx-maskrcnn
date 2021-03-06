export MXNET_CUDNN_AUTOTUNE_DEFAULT=0
export PYTHONUNBUFFERED=1
export MXNET_ENABLE_GPU_P2P=0
# export PYTHONPATH=${PYTHONPATH}:incubator-mxnet/python/

TRAIN_DIR=/mnt/models/v3
DATASET=coco
SET=train2017
TEST_SET=val2017
mkdir -p ${TRAIN_DIR}

# Train
CUDA_VISIBLE_DEVICES=7 python3.6 train_alternate_mask_fpn.py \
    --network resnet_fpn \
    --dataset ${DATASET} \
    --image_set ${SET} \
    --root_path ${TRAIN_DIR} \
    --prefix ${TRAIN_DIR} \
    --pretrained ./model/final \
    --pretrained_epoch 0 \
    --gpu 0 |& tee -a ${TRAIN_DIR}/train.log

