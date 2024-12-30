import PIL.Image as Image
import os



# 定义图像拼接函数
def combine(IMGlist, IMGrow, IMGcolumn, image_width, image_height, SAVEpath):
    img_format = ['.png', '.jpg']  # 图片格式
    image_row = IMGrow  # 图片间隔，也就是合并成一张图后，一共有几行
    image_column = IMGcolumn  # 图片间隔，也就是合并成一张图后，一共有几列

    # 获取图片集地址下的所有图片名称
    image_names = []
    for i in range(1, image_row * image_column + 1):
        image_names.append(str(i) + "_fake.png")

    # 简单的对于参数的设定和实际图片集的大小进行数量判断
    if len(IMGlist) != image_row * image_column:
        print(len(IMGlist))
        raise ValueError("合成图片的参数和要求的数量不能匹配！")

    to_image = Image.new('RGB', (image_column * int(image_width), image_row * int(image_height)))  # 创建一个新图
    # 循环遍历，把每张图片按顺序粘贴到对应位置上
    for y in range(1, image_row + 1):
        for x in range(1, image_column + 1):
            from_image = IMGlist[image_column * (y - 1) + x - 1]
            to_image.paste(from_image, ((x - 1) * int(image_width), (y - 1) * int(image_height)))
    print("图片拼接完成")
    return to_image.save(SAVEpath)  # 保存新图

