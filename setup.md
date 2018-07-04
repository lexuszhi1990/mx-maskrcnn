### setup on 177

GPU:
docker run --network host --rm -it -v /home/fulingzhi/workspace/mx-maskrcnn:/app -v /mnt/gf_mnt/datasets/cityscape:/mnt/data/cityscape -v /mnt/gf_mnt/datasets/cocoapi:/mnt/data/coco -v /mnt/gf_mnt/models:/mnt/models mxnet-cu90/python:1.2.0-roialign

CPU:
docker run --network host --rm -it -v /home/fulingzhi/workspace/mx-maskrcnn:/app -v /mnt/gf_mnt/datasets/cityscape:/mnt/data/cityscape mxnet/python:1.2.0-roialign

make clean && make
bash scripts/demo.sh

### setup on 172

docker run --network host --rm -it -v /home/david/fashionAI/mx-maskrcnn:/app-dev -v /data/david/cocoapi:/mnt/data/coco -v /data/david/models/mask-rcnn:/mnt/models -v /data/david/logdir/mask-rcnn:/mnt/logs mxnet-cu90/python:1.2.0-roialign

for fai
```
docker run --network host --rm -it -v /home/david/fashionAI/mx-maskrcnn:/app-dev -v /data/david/cocoapi:/mnt/data/coco -v /data/david/fai_kp/base_dataset:/mnt/data/fai_kp -v /data/david/models/fai-mask-rcnn:/mnt/models -v /data/david/logdir/fai-mask-rcnn:/mnt/logs mxnet-cu90/python:1.2.0-roialign
```

for coco:
```
docker run --network host --rm -it -v /home/david/fashionAI/mx-maskrcnn:/app-dev -v /data/david/cocoapi:/mnt/data/coco -v /data/david/models/mask-rcnn:/mnt/models -v /data/david/logdir/mask-rcnn:/mnt/logs mxnet-cu90/python:1.2.0-roialign
```

demo:
bash scripts/demo.sh

train:
bash scripts/train_alternate.sh
bash scripts/train_alternate_fai_kp.sh

### packages

apt-get install -y python3.6-dev python3-tk
pip3.6 install opencv-python easydict cython
pip3.6 install matplotlib



/data/david/fai_kp/base_dataset/build/v1/rpn1-0005.params
/mnt/data/fai_kp/build/v1/rpn1-0005.params
