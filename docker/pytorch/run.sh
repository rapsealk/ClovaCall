docker run --gpus '"device=0"' -it --rm -p 10020:8888 -v ${PWD}:/workspace nvcr.io/nvidia/pytorch:21.08-py
# docker run --gpus 0 -it --rm -p 10020:8888 -v ${PWD}:/workspace nvcr.io/nvidia/pytorch:21.08-py