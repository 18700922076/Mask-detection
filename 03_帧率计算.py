import time
import cv2
# 计算帧率
start = time.time()  # 开始时间
#################
#project
image="photo"
#################
end = time.time()  # 结束时间
fps = 1 / (end - start)  # 帧率
frame = cv2.putText(image, "fps:{:.3f}".format(fps), (3, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)  # 绘制