import argparse
from csv import reader
import rosbag
from sensor_msgs import point_cloud2
import numpy as np
import os

from utils import write_points, write_label


def read_bag(bag_root, bag_dir):
    msgs = []
    bag = rosbag.Bag(str.join(bag_root, bag_dir))
    for topic, msg, t in bag.read_messages(topics=['/mec_algo_cloud']):
        msgs.append(msg)
    bag.close()
    return msgs


def read_csv(csv_root, csv_dir):
    rows = {}
    suffix = os.path.splitext(csv_dir)[1]
    assert suffix == '.csv'
    with open(str.join(csv_root, csv_dir)) as tt:
        csvreader = reader(tt)
        for row in csvreader:
            row_id = 0
            if row[1]:
                rows[row_id] = row
            row_id += 1
    tt.close()
    return rows


def convert_bag2bin(msg, output_root, id):
    pc = point_cloud2.read_points_list(msg, field_names=['x', 'y', 'z', 'intensity'], skip_nans=True)
    pc = np.array(pc, dtype=np.float32)
    out_filename = str.join(id, '.bin')
    output_file_name = '{0}/{1}'.format(output_root, out_filename)
    write_points(pc, output_file_name)


def convert_csv2txt(row, output_root, id):
    out_filename = str.join(id, '.txt')
    output_file_name = '{0}/{1}'.format(output_root, out_filename)
    result = {
        'name': 'Wheelchair',
        'truncated': '0.00',
        'occluded': '3',
        'alpha': '-10',
        'bbox': ' '.join(np.zeros(4, dtype=np.float32)),
        'dimensions': np.array(row[4:6]),
        'location': np.array(row[1:3]),
        'rotation_y': row[9],
        'score': '0'
    }

    write_label(result, output_file_name)


def convert_dataset(bag_root, csv_root, pc_output_root, label_output_root):
    id = 0
    csv_files = sorted(os.listdir(csv_root))
    bag_files = sorted(os.listdir(bag_root))
    assert len(csv_files) == len(bag_files)
    for file_id in range(len(csv_files)):
        assert os.path.splitext(csv_files[file_id])[0] == os.path.splitext(bag_files[file_id])[0]
        msgs = read_bag(bag_root, bag_files[file_id])
        rows = read_csv(csv_root, csv_files[file_id])
        for row in rows:
            convert_bag2bin(msgs[row], pc_output_root, id)
            convert_csv2txt(rows[row], label_output_root, id)
            id += 1


def main(args):
    bag_root = args.bag_data_root
    csv_root = args.csv_data_root
    pc_output_root = args.pc_output_root
    label_output_root = args.label_output_root

    convert_dataset(bag_root, csv_root, pc_output_root, label_output_root)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset infomation')
    parser.add_argument('--bag_data_root', default='../datasets/ROSBAG',
                        help='your root for rosbag datas', required=True)
    parser.add_argument('--csv_data_root', default='../datasets/ROSBAG/export',
                        help='your root for csv files', required=True)
    parser.add_argument('--pc_output_root', help='output root for point cloud data')
    parser.add_argument('--label_output_root', help='output root for label data')
    args = parser.parse_args()

    main(args)
