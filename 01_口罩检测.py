##############
import cv2
import time
import numpy as np

def facesdetecter(image):
    start = time.time()  # 开始时间
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)  # 将图片转化成灰度
    image = cv2.GaussianBlur(image, (5, 5), 0)
    #image2 = image.copy()
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)  # 将图片转化成HSV格式
    #cv2.imshow("hsv",hsv)#显示HSV图
    H, S, V = cv2.split(hsv)

    thresh_h = cv2.inRange(H, 5, 34)  # 0-180du 提取人体肤色区域(12,34)

    cv2.imshow("hsv-H-threshold", thresh_h)  # 显示二值化图

    eyes = eyes_cascade.detectMultiScale(gray, 1.3, 2)  # 眼睛检测



    # for (x,y,w,h) in lefteye:
    # frame = cv2.rectangle(image,(x,y),(x+w,y+h),(255,0,250),2)
    # 计算帧率
    total_area_mask = 0
    total_area_eyes = 0
    if len(eyes) == 2:
        # 眼睛区域
        if eyes[0][0]>eyes[1][0]:
            rect_eyes = [(eyes[1][0], eyes[1][1], eyes[0][0] + eyes[0][2] - eyes[1][0], eyes[0][3])]
        else:
            rect_eyes = [(eyes[0][0], eyes[0][1], eyes[1][0] + eyes[1][2] - eyes[0][0], eyes[1][3])]

        for (x, y, w, h) in rect_eyes:
            #cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            thresh_eyes = thresh_h[y:y + h, x:x + w]


            contours, hierarchy = cv2.findContours(thresh_eyes, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 寻找前景

            cv2.drawContours(thresh_eyes, contours, -1, (0, 0, 255), 3)
            #cv2.imshow("thresh_eyes", thresh_eyes)
            for cont in contours:
                Area = cv2.contourArea(cont)  # 计算轮廓面积           
                total_area_eyes += Area
        print("total_area_eyes=", total_area_eyes)
        # 口罩区域
        rect_mask = [(x, y + h, w, h * 2)]
        for (x, y, w, h) in rect_mask:
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 255), 2)

            thresh_mask = thresh_h[y:y + h, x:x + w]
            # image2[y:y+h,x:x+w]=thresh_h
            contours, hierarchy = cv2.findContours(thresh_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # 寻找前景
            #cv2.drawContours(image, contours, -1, (0, 0, 255), 3)
            for cont in contours:
                Area = cv2.contourArea(cont)  # 计算轮廓面积           
                total_area_mask += Area
        print("total_area_mask=", total_area_mask)
        # print("{}-prospect:{}".format(count,Area),end="  ") #打印出每个前景的面积
        if total_area_eyes < total_area_mask:
            print("------------无口罩")
        if total_area_eyes > total_area_mask:
            print("------------------戴口罩")

            # cv2.imshow("hsv-H-threshold-roi",thresh_h)#显示二值化图
    end = time.time()  # 结束时间
    fps = 1 / (end - start)  # 帧率
    frame = cv2.putText(image, "fps:{:.3f}".format(fps), (3, 15), cv2.FONT_HERSHEY_COMPLEX, 0.5, (255, 0, 0), 1)  # 绘制
    cv2.imshow("face", image)  # 显示最终图片
    #cv2.imshow("face_f", image2)  # 显示肤色检测图片
    # line_detect_possible_demo(image)#霍夫变换找直线


def mogseparate(image):
    fgmask = mog.apply(image)
    ret, binary = cv2.threshold(fgmask, 220, 255, cv2.THRESH_BINARY)
    cv2.imshow("fgmask", fgmask)
    binary = cv2.morphologyEx(binary, cv2.MORPH_OPEN, se)
    backgimage = mog.getBackgroundImage()
    # 查找轮廓
    # contours, hierarchy = cv2.findContours(binary,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
    # cv2.drawContours(binary,contours,-1,(0,0,255),3)
    # binary = cv2.erode(binary, None, iterations=4)# 腐蚀
    cv2.imshow('erode', binary)
    cv2.imshow("backgimage", backgimage)
    cv2.imshow("frame", image)
    cv2.imshow("binary", binary)


def knnseperate(image):
    mog_sub_mask = mog2_sub.apply(image)
    knn_sub_mask = knn_sub.apply(image)

    cv2.imshow('original', image)
    cv2.imshow('MOG2', mog_sub_mask)
    cv2.imshow('KNN', knn_sub_mask)





face_cascade = cv2.CascadeClassifier("haarcascade_frontalface_alt2.xml")
face_cascade.load("data/haarcascades/haarcascade_frontalface_alt2.xml")  # 一定要告诉编译器文件所在的具体位置
'''此文件是opencv的haar人脸特征分类器'''
eyes_cascade = cv2.CascadeClassifier("haarcascade_eye_tree_eyeglasses.xml")
eyes_cascade.load("data/haarcascades/haarcascade_eye_tree_eyeglasses.xml")  # 一定要告诉编译器文件所在的具体位置
'''此文件是opencv的haar眼镜特征分类器'''

upperbody_cascade = cv2.CascadeClassifier("haarcascade_upperbody.xml")
upperbody_cascade.load("data/haarcascades/haarcascade_upperbody.xml")  # 一定要告诉编译器文件所在的具体位置
'''此文件是opencv的haar上半身特征分类器'''

mouth_cascade = cv2.CascadeClassifier("haarcascade_mcs_mouth.xml")
mouth_cascade.load("data/haarcascades/haarcascade_mcs_mouth.xml")  # 一定要告诉编译器文件所在的具体位置
'''此文件是opencv的haar上半身特征分类器'''

nose_cascade = cv2.CascadeClassifier("haarcascade_mcs_nose.xml")
nose_cascade.load("data/haarcascades/haarcascade_mcs_nose.xml")  # 一定要告诉编译器文件所在的具体位置
'''此文件是opencv的haar上半身特征分类器'''

lefteye_cascade = cv2.CascadeClassifier("haarcascade_lefteye_2splits.xml")
lefteye_cascade.load("data/haarcascades/haarcascade_lefteye_2splits.xml")  # 一定要告诉编译器文件所在的具体位置
'''此文件是opencv的haar上半身特征分类器'''

mog = cv2.createBackgroundSubtractorMOG2()
se = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

knn_sub = cv2.createBackgroundSubtractorKNN()
mog2_sub = cv2.createBackgroundSubtractorMOG2()

if __name__ == '__main__':

    k_write = 1
    capture = cv2.VideoCapture(0)
    while (True):

        ref, frame = capture.read()
        if ref == False:
            print("打开摄像头错误")
            break
        #cv2.imshow("frame",frame)
        # 等待30ms显示图像，若过程中按“Esc”退出
        c = cv2.waitKey(30) & 0xff
        if c == 27:
            capture.release()
            break
        facesdetecter(frame)
    capture.release()
    cv2.destroyAllWindows()