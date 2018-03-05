# export CC=cuda-gcc
# export CXX=cuda-g++
export NO_CUDA=1
export DEBUG=1
python setup.py build develop
