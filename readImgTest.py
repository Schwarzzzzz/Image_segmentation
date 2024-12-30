from PIL import Image
import matplotlib.pyplot as plt
from gpu_memory_log import gpu_memory_log

print("-------------------------------------")
print("memory_allocated: ", gpu_memory_log()[0])
print("max_memory_allocated: ", gpu_memory_log()[1])
print("memory_cached: ", gpu_memory_log()[2])
print("max_memory_cached: ", gpu_memory_log()[3])
print("Total_Memory: ", gpu_memory_log()[4])
print("-------------------------------------")
img = Image.open("D:\Curriculum_document\\2022Summer\yolov5-master\data\images\\val\\4320P\\000000002685.jpg")
print("-------------------------------------")
print("memory_allocated: ", gpu_memory_log()[0])
print("max_memory_allocated: ", gpu_memory_log()[1])
print("memory_cached: ", gpu_memory_log()[2])
print("max_memory_cached: ", gpu_memory_log()[3])
print("Total_Memory: ", gpu_memory_log()[4])
print("-------------------------------------")


plt.figure("Image")  # 图像窗口名称
plt.imshow(img)
plt.axis('on')  # 关掉坐标轴为 off
plt.title('image')  # 图像题目

# 必须有这个，要不然无法显示
plt.show()