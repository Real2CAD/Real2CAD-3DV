# Real2CAD-3DV
Shape Matching of Real 3D Object Data to Synthetic 3D CADs (3DV project @ ETHZ)

**Group Member:** Yue Pan, Yuanwen Yue, Bingxin Ke, Yujie He

**Supervisor:** Dr. Iro Armeni, Shengyu Huang

[**report**](https://github.com/Real2CAD/Real2CAD-3DV/blob/main/doc/3DV_report_Real2CAD.pdf) |  [**presentation**](https://github.com/Real2CAD/Real2CAD-3DV/blob/main/doc/3DV_Final_Pre_Real2CAD.pdf) | [**demo**]()

----

## Data preparation and preprocessing
TBA


## How to use
1. Train the model, monitor it via wandb

```shell
cd ./src
# configure the path and parameters in train_scannet.sh 
bash train_scannet.sh 
```

2. Evaluate the model on ScanNet or 2D3DS dataset

```
# configure the path and parameters in eval_xxx.sh
bash eval_scannet.sh
bash eval_2d3ds.sh
```

## Related Projects 
We thanks greatly for the following projects for the backbone and the datasets.
 - [Scan2CAD](https://github.com/skanti/Scan2CAD)
 - [JointEmbedding](https://github.com/xheon/JointEmbedding)
 - [ShapeNet](https://shapenet.org/)
 - [ScanNet](http://www.scan-net.org/)
 - [2D3DS](https://github.com/alexsax/2D-3D-Semantics)
