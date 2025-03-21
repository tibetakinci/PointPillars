# PointPillars: 3D Object Detection from Point Clouds including Wheelchair Users

This repository is forked from PointPillars implementation of [zhulf0804](https://github.com/zhulf0804/PointPillars).
A slight modification is performed on top of the implementation to add a new class of object, wheelchair users.

## Contents:

- [Setup](#setup)
  - [Cloning repository](#cloning-repository)
  - [Conda environment](#setup-conda-environment)
  - [Compile](#compile)
  - [Dataset](#dataset)
    - [Convert datasets](#convert-datasets)
    - [Prepare dataset](#prepare-dataset)
- [Training](#training)
- [Evaluation](#evaluation)
- [Testing](#testing)
- [Remarks](#remarks)

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
After conversion and before preparation, the format of the dataset should be as follows:
```
dataset_name
|── ImageSets
    ├── test (.txt)
    ├── train (.txt)
    ├── trainval (.txt)
    └── val (.txt)
|── training
    ├── label_2 (.txt)
    └── velodyne (.bin)
|── testing
    └── velodyne (.bin)
```

#### Convert datasets
##### Convert custom dataset
To convert dataset into specified format, a [convert_custom_dataset.py](convert_custom_dataset.py) is implemented. 
This script is utilized to convert point cloud raw data from rosbag to .bin file, label data from .mat file to .txt file.  
Before running [convert_custom_dataset.py](convert_custom_dataset.py), please export labeled data from Lidar Labeler app in MATLAB as .csv file by running below code box in MATLAB.
```
load('your_mat_file.mat')
tt = gTruth.LabelData
writetimetable(tt, 'your_csv_file_name.csv')
```
After exporting .csv file, please run the script with correct arguments.
```
python convert_custom_dataset.py --bag_data_root your_rosbag_data_root --csv_data_root your_csv_data_root --pc_output_root your_point_cloud_root --label_output_root your_label_data_root --start_id starting_index
```
> Note: *pc_output_root* and *label_output_root* arguments are not required. If not specified, the script will create a new folder at parent directory as *velodyne* and *label_2* respectively.

> Note: *start_id* argument is not required. It is in integer type. It is implemented as starting index for output files. Default value is 0

##### Convert KITTI
To convert KITTI dataset into specified format, [convert_kitti.py](convert_kitti.py) script has been implemented. The raw label data KITTI dataset provides is in camera coordinates. The location on all x, y, z coordinates are given in camera plane. In order to convert those parameters, calibration file has been utilized. [convert_kitti.py](convert_kitti.py) script basically does the conversion from camera to lidar coordinates and gets rid of difficulty, truncated, occluded, bbox parameters from the raw label data.
```
python convert_kitti.py --label_root kitti_label_data_root --calib_root kitti_calib_data_root --output_root desired_output_root
```

#### Prepare dataset
Please run the below code to pickle dataset objects for more efficient utilization by the model.
```
cd PointPillars
python pre_process_dataset.py --data_root your_path_to_dataset
```
After running [pre_process_dataset.py](pre_process_dataset.py) script, dataset folder should look like this:
```
dataset_name
|── ImageSets
    ├── test (.txt)
    ├── train (.txt)
    ├── trainval (.txt)
    └── val (.txt)
|── training
    ├── label_2 (.txt)
    └── velodyne (.bin)
|── testing
    └── velodyne (.bin)
|── {dataset_name}_gt_database (.bin)
|── {dataset_name}_infos_train.pkl
|── {dataset_name}_infos_val.pkl
|── {dataset_name}_infos_test.pkl
|── {dataset_name}_infos_trainval.pkl
|── {dataset_name}_dbinfos_train.pkl
```
#### Training 
You can train the model by simply running below code
```
python train.py --data_root your_path_to_dataset
```
Or retrain a pretrained model by running below code
```
python train.py --data_root your_path_to_dataset --ckpt your_path_to_checkpoint_file
```

> The list of arguments and explanations:  
> *dataset_name:* Dataset name, set default to 'custom'  
> *saved_path:* Root for output directory which includes checkpoint and summary files, set default to 'pillar_logs'  
> *batch_size, num_workers, nclasses:* Model related params, type: int  
> *init_lr, max_epochs, log_freq, ckpt_freq_epoch:* Training related params, type: int  
> *no_cuda:* Whether to use CUDA or not on training
 
#### Evaluation
In order to retrieve the metrics of pretrained model evaluation is done by running:
```
python evaluate.py --data_root your_path_to_dataset --ckpt your_path_to_pth_file
```

> Note: Once the model is not trained sufficiently, it will make predictions with less scores. Score threshold for evaluation is set to 0.1 at [pointpillars.py](model/pointpillars.py). If the evaluation script fails, please adjust the *self.score_thr* variable.

> The list of arguments and explanations:  
> *dataset_name:* Dataset name, set default to 'custom'  
> *saved_path:* Root for output directory which includes checkpoint and summary files, set default to 'results'  
> *batch_size, num_workers, nclasses:* Model related params, type: int  
> *no_cuda:* Whether to use CUDA or not on training

#### Testing
To infer and visualize point cloud detection
```
python test.py --ckpt your_path_to_pth_file --pc_path your_pc_path
```

To infer and visualize point cloud detection and ground truth.
```
python test.py --ckpt your_path_to_pth_file --pc_path your_pc_path --gt_path your_gt_path
```

> The list of arguments and explanations:  
> *dataset_name:* Dataset name, set default to 'custom'  
> *no_cuda:* Whether to use CUDA or not on training

### Remarks
#### Pretrained models
Pretrained models are located at [pretrained](pretrained/) directory at root. There are total of four subfolders including the previously trained model on KITTI dataset only. Summaries (loss functions, learning rate, momentum) can be visualized by running below code.
‘’’
tensorboard —logdir pretrained
‘’’

#### Issue with loss function
Sometimes loss functions throws an error. That is because of the input of binary cross entropy at line 47 of [loss.py](loss/loss.py). The reason is that binary cross entropy should input values between [0, 1]. Sometimes the input can become NaN and both *sigmoid* and *one_hot* functions can’t convert. A resource I found online is [here](). One suggested solution; using operation *torch.clamp()*. Please keep in mind that this is a very rare error to produce. 