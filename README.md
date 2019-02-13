# evolo
[![CircleCI](https://circleci.com/gh/tsukar/evolo.svg?style=shield)](https://circleci.com/gh/tsukar/evolo)

Evolutionary YOLO

## Requirements

- Python 3.6.7
- Jupyter Notebook

## Usage

Clone this repository and [AlexeyAB/darknet](https://github.com/AlexeyAB/darknet), respectively.

```
$ git clone https://github.com/tsukar/evolo.git
$ git clone https://github.com/AlexeyAB/darknet.git
$ cd darknet/
```

Edit the `Makefile` so that you can use your GPU.

```diff
@@ -1,7 +1,7 @@
-GPU=0
-CUDNN=0
+GPU=1
+CUDNN=1
 CUDNN_HALF=0
-OPENCV=0
+OPENCV=1
 AVX=0
 OPENMP=0
 LIBSO=0
@@ -30,7 +30,7 @@ OS := $(shell uname)
 # ARCH= -gencode arch=compute_72,code=[sm_72,compute_72]
 
 # GTX 1080, GTX 1070, GTX 1060, GTX 1050, GTX 1030, Titan Xp, Tesla P40, Tesla P4
-# ARCH= -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61
+ARCH= -gencode arch=compute_61,code=sm_61 -gencode arch=compute_61,code=compute_61
 
 # GP100/Tesla P100 <96> DGX-1
 # ARCH= -gencode arch=compute_60,code=sm_60
```

Just `make` and copy the compiled binary into the root directory of this repository.

```
$ make
$ cp darknet ../evolo/
$ cd ../evolo/
```

Download the pre-trained weights from YOLOv2 official site.

```
$ wget https://pjreddie.com/media/files/darknet19_448.conv.23
```

Download [train.zip](https://drive.google.com/open?id=12wEAyNHoxt-rBUEZZH0ym9OGuxGWAXBX) and [test.zip](https://drive.google.com/open?id=1rmph2beAowAkuGkUzykH9i8YR9gFKFGB) (currently not available - please contact me at `tsukada [at] iba.t.u-tokyo.ac.jp` if you need them). Put the decompressed `*.jpg` files in `data/x-ray/train/`, `data/x-ray/test/`, respectively.

Run Jupyter Notebook and open `evolution.ipynb` -> `Run All` to start evolution.

```
$ jupyter notebook
```

After evolution, open `evaluation.ipynb` -> `Run All` for evaluation.
