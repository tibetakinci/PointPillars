import argparse
from csv import reader
import rosbag
from sensor_msgs import point_cloud2
import numpy as np
import os

from utils import write_points, write_label


def convert_bag2bin(filename, *args):
    if args.pc_output_root:
        output_path = args.pc_output_root
    else:
        directory = 'velodyne'
        output_path = os.path.join(os.path.abspath(filename), directory)
    bag = rosbag.Bag(filename)
    output_name = 0
    for topic, msg, t in bag.read_messages(topics=['/mec_algo_cloud']):
        pc = point_cloud2.read_points_list(msg, field_names=['x', 'y', 'z', 'intensity'], skip_nans=True)
        pc = np.array(pc, dtype=np.float32)
        out_filename = '{:0>6}.bin'.format(output_name)
        output_file_name = '{0}/{1}'.format(output_path, out_filename)

        write_points(pc, output_file_name)
        output_name += 1

    bag.close()


def convert_csv2txt(filename, *args):
    if args.label_output_root:
        output_path = args.label_output_root
    else:
        directory = 'label_2'
        output_path = os.path.join(os.path.abspath(filename), directory)
    suffix = os.path.splitext(filename)[1]
    assert suffix == '.csv'
    with open(filename) as tt:
        csvreader = reader(tt)
        next(csvreader)

        output_name = 0
        for row in csvreader:
            if row[1]:
                out_filename = '{:0>6}.txt'.format(output_name)
                output_file_name = '{0}/{1}'.format(output_path, out_filename)
                result = {
                    'name': 'Wheelchair',
                    'truncated': '0.00',
                    'occluded': '3',
                    'alpha': '-10',
                    'bbox': ' '.join(np.zeros(4, dtype=np.float32)),
                    'dimensions': np.array(row[4:6]),
                    'location': np.array(row[1:3]),
                    'rotation_y': row[9]
                }

                write_label(result, output_file_name)
            output_name += 1

    tt.close()


def main(args):
    bag_root = args.bag_data_root
    csv_root = args.csv_data_root
    pc_output_root = args.pc_output_root
    label_output_root = args.label_output_root
    print(pc_output_root)
    pass

    for filename in os.listdir(bag_root):
        convert_bag2bin(filename, pc_output_root)

    for filename in os.listdir(csv_root):
        convert_csv2txt(filename, label_output_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset infomation')
    parser.add_argument('--bag_data_root', default='../datasets/ROSBAG',
                        help='your root for rosbag datas')
    parser.add_argument('--csv_data_root', default='../datasets/ROSBAG/export',
                        help='your root for csv files')
    parser.add_argument('--pc_output_root', help='output root for point cloud data')
    parser.add_argument('--label_output_root', help='output root for label data')
    args = parser.parse_args()

    main(args)