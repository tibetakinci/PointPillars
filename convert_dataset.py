import argparse
from csv import reader
import rosbag
from sensor_msgs import point_cloud2
import numpy as np
import os

from utils import write_points, write_label


def convert_bag2bin(bag_root):
    output_path = os.path.splitext(bag_root)[0:2]  # TODO
    bag = rosbag.Bag(bag_root)
    output_name = 0
    for topic, msg, t in bag.read_messages(topics=['/mec_algo_cloud']):
        pc = point_cloud2.read_points_list(msg, field_names=['x', 'y', 'z', 'intensity'], skip_nans=True)
        pc = np.array(pc, dtype=np.float32)
        filename = '{:0>6}.bin'.format(output_name)
        output_file_name = '{0}/{1}'.format(output_path, filename)

        write_points(pc, output_file_name)
        output_name += 1

    bag.close()


def convert_csv2txt(csv_root):
    output_path = os.path.splitext(csv_root)[0:2]  #TODO
    suffix = os.path.splitext(csv_root)[1]
    assert suffix == '.csv'
    with open(csv_root) as tt:
        csvreader = reader(tt)
        next(csvreader)

        output_name = 0
        for row in csvreader:
            if row[1]:
                filename = '{:0>6}.txt'.format(output_name)
                output_file_name = '{0}/{1}'.format(output_path, filename)
                result = {
                    'name': 'Wheelchair',
                    'truncated': '0.00',
                    'occluded': '3',
                    'alpha': '-10',
                    'bbox': ' '.join(np.zeros(4, dtype=np.float32)),
                    'dimensions': np.array(row[2:4]),
                    'location': np.array(row[4:6]),
                    'rotation_y': row[7]
                }

                write_label(result, output_file_name)
            output_name += 1

    tt.close()


def main(args):
    bag_root = args.bag_data_root
    csv_root = args.csv_data_root

    convert_bag2bin(bag_root)
    convert_csv2txt(csv_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset infomation')
    parser.add_argument('--bag_data_root', default='../datasets/ROSBAG',
                        help='your root for rosbag datas')
    parser.add_argument('--csv_data_root', default='../datasets/ROSBAG/export',
                        help='your root for csv files')
    args = parser.parse_args()

    main(args)