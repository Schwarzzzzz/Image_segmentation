from PIL import Image
import time, os

def get_pixel_colors(image_path, save_path, filename):
    try:
        # 打开图像文件
        image = Image.open(image_path)

        # 获取图像的像素数据
        pixel_data = image.load()

        # 获取图像的宽度和高度
        width, height = image.size

        # 存储像素颜色的列表
        pixel_colors = []

        # 遍历图像的每个像素
        for y in range(height):
            for x in range(width):
                # 获取像素的RGB颜色值
                r, g, b = pixel_data[x, y]

                if not (abs(r - g) <= 20 and abs(r - b) <= 20 and abs(g - b) <= 20 and 180 <= r <= 210) and not (abs(r - 80) <= 10 and abs(g - 100) <= 10 and abs(b - 110) <= 10):
                        pixel_data[x, y] = (255, g, b)

        modified_image_path = save_path + filename + ".jpg"
        image.save(modified_image_path)

    except IOError:
        print("无法打开图像文件:", image_path)

def process_images_in_folder(folder_path, save_path):
    # 遍历文件夹内的所有文件
    for filename in os.listdir(folder_path):
        # 拼接文件的完整路径
        file_path = os.path.join(folder_path, filename)
        
        # 判断文件是否是图像文件（可根据需要增加其他文件类型的判断条件）
        if os.path.isfile(file_path) and file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.gif')):
            get_pixel_colors(file_path, save_path, filename)


image_path = "D:\\Curriculum_document\\2022Summer\\Image-segmentation\\colorDiv\\1080P\\"  # 替换为你的图片文件路径
save_path = "D:\\Curriculum_document\\2022Summer\\Image-segmentation\\colorDiv\\1080P_Div\\"

start_time = time.time()
process_images_in_folder(image_path, save_path)
end_time = time.time()
elapsed_time = end_time - start_time
print("代码执行耗时: {:.4f} 秒".format(elapsed_time))
