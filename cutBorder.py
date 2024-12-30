from PIL import Image

def cutBorder(orginalIMGpath, finalIMGpath, SAVEpath):
    cutted_img = Image.open(finalIMGpath) # 打开final.jpg文件，并赋值给img

    original_Img = Image.open(orginalIMGpath)
    inImgWidth = original_Img.width
    inImgHeight = original_Img.height

    region = cutted_img.crop((0, 0, inImgWidth, inImgHeight))  # 0,0表示要裁剪的位置的左上角坐标，inImgWidth, inImgHeight表示右下角。
    print("边框裁剪完成")
    region.save(SAVEpath)  # 将裁剪下来的图片保存
