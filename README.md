# PointPillars: 3D Object Detection from Point Clouds including Wheelchair Users

This repository is forked from PointPillars implementation of [zhulf0804](https://github.com/zhulf0804/PointPillars).
A slight modification is performed on top of the implementation to add a new class of object, wheelchair users.

## Contents:

- [Setup](#setup)
  - [Cloning repository](#cloning-repository)
  - [Conda environment](#setup-conda-environment)
  - [Compile](#compile)
  - [Dataset](#dataset)
    - [Convert dataset](#convert-dataset)
    - [Prepare dataset](#prepare-dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Plotting](#plotting)
- [Overview of PointPillars](#overview)
- [Results](#results)

### Setup
#### Cloning repository
```
git clone https://github.com/tibetakinci/PointPillars.git
cd PointPillars
```

#### Setup conda environment
Please visit the [Anaconda install page](https://docs.anaconda.com/free/anaconda/install/index.html) if you do not already have conda installed.
```
conda create -n PointPillars
conda activate PointPillars
conda install -c anaconda pip
pip install -r requirements.txt
```

### Compile
This block extends pre-implemented C++ functions using [CudaExtension](https://pytorch.org/tutorials/advanced/cpp_extension.html). 
```
cd ops
python setup.py develop
```

### Dataset
Format of the dataset should be as follows:
```
dataset_name
|── training
    ├── label_2 (.txt)
    └── velodyne (.bin)
|── testing
    └── velodyne (.bin)
```

#### Convert dataset
To convert dataset into specified format, a [convert_dataset.py](convert_dataset.py) is implemented. 
This script is utilized to convert point cloud raw data from rosbag to .bin file, label data from .mat file to .txt file.  
Before running [convert_dataset.py](convert_dataset.py), please export labeled data from Lidar Labeler app in MATLAB as .csv file by running below code box in MATLAB.
```
load('your_mat_file.mat')
tt = gTruth.LabelData
writetimetable(tt, 'your_csv_file_name.csv')
```
After exporting .csv file, please run the script with correct arguments.
```
python convert_dataset.py --bag_data_root your_rosbag_data_root --csv_data_root your_csv_data_root --pc_output_root your_point_cloud_root --label_output_root your_label_data_root --start_id starting_index
```
> Note: *pc_output_root* and *label_output_root* arguments are not required. If not specified, the script will create a new folder at parent directory as *velodyne* and *label_2* respectively.

> Note: *start_id* argument is not required. It is in integer type. It is implemented as starting index for output files. Default value is 0

#### Prepare dataset
Please run the below code to pickle dataset objects for more efficient utilization by the model.
```
cd PointPillars
python pre_process_kitti.py --data_root your_path_to_kitti
```
After running [pre_process_kitti.py](pre_process_dataset.py) script, dataset folder should look like this:
```
dataset_name
|── training
    ├── label_2 (.txt)
    └── velodyne (.bin)
|── testing
    └── velodyne (.bin)
|── kitti_gt_database (.bin)
|── kitti_infos_train.pkl
|── kitti_infos_val.pkl
|── kitti_infos_test.pkl
|── kitti_infos_trainval.pkl
|── kitti_dbinfos_train.pkl
```

# [PointPillars: Fast Encoders for Object Detection from Point Clouds](https://arxiv.org/abs/1812.05784) 

A Simple PointPillars PyTorch Implenmentation for 3D Lidar(KITTI) Detection. [[Zhihu](https://zhuanlan.zhihu.com/p/521277176)]

- It can be run without installing [Spconv](https://github.com/traveller59/spconv), [mmdet](https://github.com/open-mmlab/mmdetection) or [mmdet3d](https://github.com/open-mmlab/mmdetection3d). 
- Only one detection network (PointPillars) was implemented in this repo, so the code may be more easy to read. 
- Sincere thanks for the great open-souce architectures [mmcv](https://github.com/open-mmlab/mmcv), [mmdet](https://github.com/open-mmlab/mmdetection) and [mmdet3d](https://github.com/open-mmlab/mmdetection3d), which helps me to learn 3D detetion and implement this repo.

## mAP on KITTI validation set (Easy, Moderate, Hard)

| Repo | Metric | Overall | Pedestrian | Cyclist | Car |
| :---: | :---: | :---: | :---: | :---: | :---: |
| this repo | 3D-BBox | 73.3259 62.7834 59.6278 | 51.4642 47.9446 43.8040 | 81.8677 63.6617 60.9126 | 86.6456 76.7439 74.1668 | 
| [mmdet3d v0.18.1](https://github.com/open-mmlab/mmdetection3d/tree/v0.18.1) | 3D-BBox  | 72.0537, 60.1114, 55.8320 | 52.0263, 46.4037, 42.4841 | 78.7231, 59.9526, 57.2489 | 85.4118, 73.9780, 67.7630 |
| this repo | BEV | 77.8540 69.8003 66.6699 | 59.1687 54.3456 50.5023 | 84.4268 67.1409 63.7409 | 89.9664 87.9145 85.7664 | 
| [mmdet3d v0.18.1](https://github.com/open-mmlab/mmdetection3d/tree/v0.18.1) | BEV | 76.6485, 67.7609, 64.5605 | 59.0778, 53.3638, 48.4230 | 80.9328, 63.3447, 60.0618 | 89.9348, 86.5743, 85.1967 |
| this repo | 2D-BBox | 80.5097 74.6120 71.4758 | 64.6249 61.4201 57.5965 | 86.2569 73.0828 70.1726 | 90.6471 89.3330 86.6583 |
| [mmdet3d v0.18.1](https://github.com/open-mmlab/mmdetection3d/tree/v0.18.1) | 2D-BBox | 78.4938, 73.4781, 70.3613 | 62.2413, 58.9157, 55.3660 | 82.6460, 72.3547, 68.4669 | 90.5939, 89.1638, 87.2511 |
| this repo | AOS | 74.9647 68.1712 65.2817 | 49.3777 46.7284 43.8352 | 85.0412 69.1024 66.2801 | 90.4752 88.6828 85.7298 |
| [mmdet3d v0.18.1](https://github.com/open-mmlab/mmdetection3d/tree/v0.18.1) | AOS | 72.41, 66.23, 63.55 | 46.00, 43.22, 40.94 | 80.85, 67.20, 63.63 | 90.37, 88.27, 86.07 |

- **Note: Here, we report [mmdet3d v0.18.1](https://github.com/open-mmlab/mmdetection3d/tree/v0.18.1) (2022/02/09-2022/03/01) performance based on the officially provided [checkpoint](https://github.com/open-mmlab/mmdetection3d/tree/v0.18.1/configs/pointpillars#kitti). Much improvements were made in the [mmdet3d v1.0.0rc1](https://github.com/open-mmlab/mmdetection3d/tree/v1.0.0rc1)**. 

## Detection Visualization

![](./figures/pc_pred_000134.png)
![](./figures/img_3dbbox_000134.png)

## [Compile] 

```
cd ops
python setup.py develop
```

## [Datasets]

1. Download

    Download [point cloud](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_velodyne.zip)(29GB), [images](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_image_2.zip)(12 GB), [calibration files](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_calib.zip)(16 MB)和[labels](https://s3.eu-central-1.amazonaws.com/avg-kitti/data_object_label_2.zip)(5 MB)。Format the datasets as follows:
    ```
    kitti
        |- training
            |- calib (#7481 .txt)
            |- image_2 (#7481 .png)
            |- label_2 (#7481 .txt)
            |- velodyne (#7481 .bin)
        |- testing
            |- calib (#7518 .txt)
            |- image_2 (#7518 .png)
            |- velodyne (#7418 .bin)
    ```

2. Pre-process KITTI datasets First

    ```
    cd PointPillars/
    python pre_process_kitti.py --data_root your_path_to_kitti
    ```

    Now, we have datasets as follows:
    ```
    kitti
        |- training
            |- calib (#7481 .txt)
            |- image_2 (#7481 .png)
            |- label_2 (#7481 .txt)
            |- velodyne (#7481 .bin)
            |- velodyne_reduced (#7481 .bin)
        |- testing
            |- calib (#7518 .txt)
            |- image_2 (#7518 .png)
            |- velodyne (#7518 .bin)
            |- velodyne_reduced (#7518 .bin)
        |- kitti_gt_database (# 19700 .bin)
        |- kitti_infos_train.pkl
        |- kitti_infos_val.pkl
        |- kitti_infos_trainval.pkl
        |- kitti_infos_test.pkl
        |- kitti_dbinfos_train.pkl
    
    ```

## [Training]

```
cd PointPillars/
python train.py --data_root your_path_to_kitti
```

## [Evaluation]

```
cd PointPillars/
python evaluate.py --ckpt pretrained/epoch_160.pth --data_root your_path_to_kitti 
```

## [Test]

```
cd PointPillars/

# 1. infer and visualize point cloud detection
python test.py --ckpt pretrained/epoch_160.pth --pc_path your_pc_path 

# 2. infer and visualize point cloud detection and gound truth.
python test.py --ckpt pretrained/epoch_160.pth --pc_path your_pc_path --calib_path your_calib_path  --gt_path your_gt_path

# 3. infer and visualize point cloud & image detection
python test.py --ckpt pretrained/epoch_160.pth --pc_path your_pc_path --calib_path your_calib_path --img_path your_img_path


e.g. [infer on val set 000134]

python test.py --ckpt pretrained/epoch_160.pth --pc_path /home/lifa/data/KITTI/training/velodyne_reduced/000134.bin

or

python test.py --ckpt pretrained/epoch_160.pth --pc_path /home/lifa/data/KITTI/training/velodyne_reduced/000134.bin --calib_path /home/lifa/data/KITTI/training/calib/000134.txt --img_path /home/lifa/data/KITTI/training/image_2/000134.png --gt_path /home/lifa/data/KITTI/training/label_2/000134.txt

```

## Acknowledements

Thanks for the open souce code [mmcv](https://github.com/open-mmlab/mmcv), [mmdet](https://github.com/open-mmlab/mmdetection) and [mmdet3d](https://github.com/open-mmlab/mmdetection3d).
