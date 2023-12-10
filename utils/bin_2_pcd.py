import open3d as o3d
import numpy as np
import struct
import argparse


def bin_to_pcd(binFileName):
    size_float = 4
    list_pcd = []
    with open(binFileName, "rb") as f:
        byte = f.read(size_float * 4)
        while byte:
            x, y, z, intensity = struct.unpack("ffff", byte)
            list_pcd.append([x, y, z])
            byte = f.read(size_float * 4)
    np_pcd = np.asarray(list_pcd)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(np_pcd)
    return pcd


def main(args):
    bin_file = args.bin_file_name
    pcd_file = args.pcd_file_name

    pcd = bin_to_pcd(bin_file)
    o3d.io.write_point_cloud(pcd_file, pcd)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Dataset infomation')
    parser.add_argument('--bin_file_name', default='/mnt/ssd1/lifa_rdata/det/kitti', help='directory for .bin file to convert')
    parser.add_argument('--pcd_file_name', default='/mnt/ssd1/lifa_rdata/det/kitti', help='directory to save .pcd file that has been converted')
    args = parser.parse_args()

    main(args)
