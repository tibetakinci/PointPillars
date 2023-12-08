import argparse
import numpy as np
import os

from utils import read_calib, read_label, write_label_filtered, bbox_camera2lidar
from convert_custom_dataset import filter_files


def convert_to_lidar_coordinate(calib_dict, annotation_dict, out_root, file_name):
    file_path = '/'.join(out_root, file_name)
    names = annotation_dict['name']
    dimensions = annotation_dict['dimensions']
    location = annotation_dict['location']
    rotation_y = annotation_dict['rotation_y']
    
    bboxes_camera = np.concatenate([location, dimensions, rotation_y[:, None]], axis=-1) # (N, 7)
    tr_velo_to_cam = calib_dict['Tr_velo_to_cam']
    r0_rect = calib_dict['R0_rect']
    bboxes_lidar = bbox_camera2lidar(bboxes_camera, tr_velo_to_cam, r0_rect)

    result = {
        'name': [],
        'dimensions': [],
        'location': [],
        'rotation_y': []
    }
    result['name'].append(names)
    result['dimensions'].append(bboxes_lidar[3:6])
    result['location'].append(bboxes_lidar[0:3])
    result['rotation_y'].append([bboxes_lidar[6]])

    write_label_filtered(result, file_path)


def convert_dataset(label_root, calib_root, out_root):
    if not os.path.exists(label_root) or not os.path.exists(calib_root):
        raise FileNotFoundError
    
    if out_root is None:
        out_root = os.path.join(os.path.abspath(label_root), 'velodyne')

    os.makedirs(out_root, exist_ok=True)

    label_files = sorted(filter_files(os.listdir(label_root), '.txt'))
    calib_files = sorted(filter_files(os.listdir(calib_root), '.txt'))
    print(label_files)
    print(calib_files)
    assert len(label_files) == len(calib_files)

    for index in range(len(calib_files)):
        assert os.path.splitext(label_files[index])[0] == os.path.splitext(calib_files[index])[0]
        calib_dict = read_calib(calib_root)
        annotation_dict = read_label(label_root)
        convert_to_lidar_coordinate(calib_dict, annotation_dict, out_root, calib_files[index])


def main(args):
    label_root = args.label_root
    calib_root = args.calib_root
    output_root = args.output_root

    convert_dataset(label_root, calib_root, output_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset infomation')
    parser.add_argument('--label_root', default='../datasets/ROSBAG',
                        help='root for label data of KITTI', required=True)
    parser.add_argument('--calib_root', default='../datasets/ROSBAG/export',
                        help='root for calibration data of KITTI', required=True)
    parser.add_argument('--output_root', default='../datasets/ROSBAG/export',
                        help='output root for new label files')
    args = parser.parse_args()

    main(args)
