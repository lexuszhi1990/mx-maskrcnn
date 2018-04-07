### setup on 177

GPU:
docker run --network host --rm -it -v /home/fulingzhi/workspace/mx-maskrcnn:/app -v /mnt/gf_mnt/datasets/cityscape:/mnt/data/cityscape mxnet-cu90/python:1.2.0-roialign

CPU:
docker run --network host --rm -it -v /home/fulingzhi/workspace/mx-maskrcnn:/app -v /mnt/gf_mnt/datasets/cityscape:/mnt/data/cityscape mxnet/python:1.2.0-roialign

make clean && make
bash scripts/demo.sh

### setup on 172

docker run --network host --rm -it -v /home/david/fashionAI/mx-maskrcnn:/app-dev -v /data/david/cocoapi:/mnt/data/coco -v /data/david/models/mask-rcnn:/mnt/models -v /data/david/logdir/mask-rcnn:/mnt/logs mxnet-cu90/python:1.2.0-roialign


demo:
bash scripts/demo.sh


### packages

apt-get install -y python3.6-dev
pip3.6 install opencv-python easydict cython

