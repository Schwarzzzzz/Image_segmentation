from PIL import Image
import os
import time
import argparse
from tkinter import messagebox as msgbox


def segmentation(IMGpath, img_split_row, img_split_col):
    # 选择图片
    img_path = IMGpath  # 图片路径
    inImg = Image.open(img_path)
    inImgWidth = inImg.width
    inImgHeight = inImg.height

    # 判断是否选择
    if img_path != '':
        # 要保存的图片路径(保存为png图片格式)
        if os.path.dirname(img_path) == "":
            img_path = os.getcwd() + "//" + img_path
        img_save = os.path.dirname(img_path) + "\\cutted\\" + str(img_split_row) + "x" + str(img_split_col) + "\\"
        if not os.path.exists(img_save):
            os.makedirs(img_save)
        # 分割图片
        img_ext_name = os.path.splitext(os.path.basename(img_path))[1]  # 去除文件扩展名
        split_size_w = inImgWidth / img_split_col
        split_size_h = inImgHeight / img_split_row
        for r in range(img_split_row):
            for c in range(img_split_col):
                split_area = (split_size_w * c, r * split_size_h, split_size_w * (c + 1), split_size_h * (r + 1))
                # print( (r*img_split_col+c+1) );
                if r*img_split_col+c+1 < 10:
                    inImg.crop(split_area).resize((int(split_size_w), int(split_size_h))).save(img_save + "00" + str(r * img_split_col + c + 1) + img_ext_name)
                elif r*img_split_col+c+1 < 100:
                    inImg.crop(split_area).resize((int(split_size_w), int(split_size_h))).save(img_save + "0" + str(r * img_split_col + c + 1) + img_ext_name)
                else:
                    inImg.crop(split_area).resize((int(split_size_w), int(split_size_h))).save(img_save + str(r * img_split_col + c + 1) + img_ext_name)
        # 结束
        print("图片分割结束，一共" + str(img_split_col) + "*" + str(img_split_row) + "=" + str(
            img_split_col * img_split_row) + "张图片")

        # 返回小图片长宽
        return int(split_size_w), int(split_size_h)

    else:
        msgbox.showwarning('操作警示', '未选择文件')

if __name__=="__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-row', type=int)
    parser.add_argument('-col', type=int)
    args = parser.parse_args()
    segmentation('./input.jpg', args.row, args.col)