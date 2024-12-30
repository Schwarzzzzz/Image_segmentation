# encoding: utf-8
# author: qbit
# date: 2020-09-2
# summary: 给宽图片上下补白边，让其满足一定比例，然后缩放到指定尺寸

import math
from PIL import Image

def makeBorder(IMGpath, SAVEpath):
    r"""
    给宽图片补白边，让其满足一定比例，然后缩放到指定尺寸
    inImgPath: 输入图片路径
    outImgPath: 输出图片路径
    width: 最终宽度
    height: 最终高度
    """
    inImgPath = IMGpath  # 图片路径
    inImg: Image.Image = Image.open(inImgPath)
    bgWidth = inImg.width
    bgHeight = inImg.height
    print("图片缩放分辨率：" + str(bgWidth) + "*" + str(bgHeight))
    target_width = math.ceil(bgWidth/256)*256
    target_height = math.ceil(bgHeight/256)*256
    print("图片输出分辨率：" + str(target_width) + "*" + str(target_height))

    # 创建一个白色背景图片
    bgImg: Image.Image = Image.new("RGB", (target_width, target_height), (255, 255, 255))
    bgImg.paste(inImg, (0, 0))

    bgImg.save(SAVEpath)

