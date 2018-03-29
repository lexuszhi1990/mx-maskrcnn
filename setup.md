### setup

GPU:
docker run --network host --rm -it -v /home/fulingzhi/workspace/mx-maskrcnn:/app -v /mnt/gf_mnt/datasets/cityscape:/mnt/data/cityscape mxnet-cu90/python:1.2.0-roialign

CPU:
docker run --network host --rm -it -v /home/fulingzhi/workspace/mx-maskrcnn:/app -v /mnt/gf_mnt/datasets/cityscape:/mnt/data/cityscape mxnet/python:1.2.0-roialign

make clean && make
bash scripts/demo.sh

### packages

apt-get install -y python3.6-dev
pip3.6 install opencv-python easydict cython

