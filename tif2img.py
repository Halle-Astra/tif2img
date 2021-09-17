import os
from osgeo import gdal
import numpy as np
import cv2
import matplotlib.pyplot as plt
from PIL import Image
from argparse import ArgumentParser
import pandas as pd


# https://blog.csdn.net/weixin_42306688/article/details/112544078,
# https://blog.csdn.net/NingAnMe/article/details/98587363
def tif2img(input_str, save_root='imgs', split=False, split_folder='splits', split_method='square', split_stride=2048,
            split_h=6, split_w=6, show=False, split_show=False):
    if not os.path.isdir(input_str):
        files = [input_str]
    else:
        files = [os.path.join(input_str, i) for i in os.listdir(input_str)]

    info_ls = []

    for filepath in files:
        filename = os.path.split(filepath)[-1]
        filename = os.path.splitext(filename)[0]
        save_path = os.path.join(save_root, filename + '.png')
        dataset = gdal.Open(filepath)
        cols = dataset.RasterXSize  # 图像长度
        rows = dataset.RasterYSize  # 图像宽度
        # info_ls.append([filename] + list(dataset.GetGeoTransform()) + [rows, cols])

        # if not os.path.exists(save_path):
        # if True:
        r, g, b = [dataset.GetRasterBand(i).ReadAsArray() for i in range(1, 4)]
        img = cv2.merge([r, g, b])

        if show:
            plt.figure(figsize=(15, 15))
            plt.imshow(img)
            plt.xticks([]), plt.yticks([])  # 不显示坐标轴
            plt.show()

        if not os.path.exists(save_root):
            os.makedirs(save_root)

        if not os.path.exists(save_path):
            Image.fromarray(img.astype(np.uint8)).save(save_path)
            print('{}.tif(or TIF) transformed into {}.png.')

        if True:  # split:
            h_pn, w_pn = split_img(img, split_folder, filename, method=split_method, stride=split_stride,
                      h_pn=split_h, w_pn=split_w, show=split_show)

        info_ls.append([filename] + list(dataset.GetGeoTransform()) + [rows, cols, h_pn, w_pn ])

    info_df = pd.DataFrame(info_ls, columns=['filename', 'long', 'x_stride_long', 'y_stride_long',
                                             'lat', 'x_stride_lat', 'y_stride_lat', 'height', 'width',
                                             'h_split_number', 'w_split_number'])
    info_df.to_csv('info.csv', index=False)


def split_img(img, save_root, filename, method='square', stride=2048, h_pn=6, w_pn=6, show=False):
    if method == 'square':
        h_pn, w_pn = split_img_to_squares(img, save_root, filename, stride=stride, show=show)
    elif method == 'numbers':
        h_pn, w_pn = split_img_by_numbers(img, save_root, filename, h_pn=h_pn, w_pn=w_pn, show=show)
    else:
        print('Argument `method` is neither `square` nor `numbers`.')
    return h_pn, w_pn


def split_img_subroutine(img, save_root, filename, h_pn, w_pn, h_stride, w_stride, show=False):
    if show:
        plt.figure(figsize=(20, 30))

    cnt = 0
    for i in range(h_pn):
        for j in range(w_pn):
            cnt += 1
            h_begin = i * h_stride
            h_end = (i + 1) * h_stride
            w_begin = j * w_stride
            w_end = (j + 1) * w_stride
            save_path = os.path.join(save_root, f'{filename}_{cnt}.png')
            if not os.path.exists(save_path):
                img_patch = img[h_begin:h_end, w_begin:w_end]
                Image.fromarray(img_patch).save(save_path)
                print(os.path.join(save_root, f'{filename}_{cnt}.png'), 'generated.')

                if show:
                    plt.subplot(h_pn, w_pn, cnt)
                    plt.imshow(img_patch)
    if show:
        plt.show()


def split_img_to_squares(img, save_root, filename, stride=2048, show=False):
    save_root = os.path.join(save_root, filename)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    h_pn = int(np.ceil(img.shape[0]/stride))
    w_pn = int(np.ceil(img.shape[1]/stride))

    split_img_subroutine(img, save_root, filename, h_pn=h_pn, w_pn=w_pn, h_stride=stride, w_stride=stride, show=show)
    return h_pn, w_pn


def split_img_by_numbers(img, save_root, filename, h_pn=6, w_pn=6, show=False):
    save_root = os.path.join(save_root, filename)
    if not os.path.exists(save_root):
        os.makedirs(save_root)

    h_stride = img.shape[0] // h_pn
    h_stride += 1
    w_stride = img.shape[1] // w_pn
    w_stride += 1
    # print(h_stride,w_stride)

    split_img_subroutine(img, save_root, filename, h_pn=h_pn, w_pn=w_pn, h_stride=h_stride, w_stride=w_stride, show=show)
    return h_pn, w_pn


if __name__ == '__main__':
    argparser = ArgumentParser()
    argparser.add_argument('-i', '--input', default='tifs', type=str,
                           help='can be a tif or a folder contains tifs only.')
    argparser.add_argument('--save_root', default='imgs', type=str,
                           help='folder name of image read from tif will be saved.')
    argparser.add_argument('-s', '--split', default=True, type=bool,
                           help='will split image into patches when it evaluated `True`.')
    argparser.add_argument('--split_folder', default='splits', type=str, help='folder name of patches will be saved.')
    argparser.add_argument('--split_h', default=6, type=int, help='the number of patches will be generated by height.')
    argparser.add_argument('--split_w', default=6, type=int, help='the number of patches will be generated by width.')
    argparser.add_argument('--show', default=False, type=bool,
                           help='will show the image read from tif when it evaluated `True`.')
    argparser.add_argument('--split_show', default=False, type=bool, help='will show patches when it evaluated `True`.')
    args = argparser.parse_args()

    tif2img(input_str=args.input,
            save_root=args.save_root,
            split=args.split,
            split_folder=args.split_folder,
            split_h=args.split_h,
            split_w=args.split_w,
            show=args.show,
            split_show=args.split_show
            )
