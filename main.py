"""

                                        Блок "Нихуя интересного"
-------------------------------------------------------------------------------------------------------\
        1 ПОПЫТКА                                                                                       |
Папка - 1Try                                                                                            |
Нейронка - Yolov4, веса недоучились из-за ограничения GPU на колабе                                     |
Датасет - 300 размеченных фоток, рандомные фотки ручек из гугла                                         |
Результат - ничего не видит                                                                             |
                                                                                                        |
        2 ПОПЫТКА                                                                                       |
Папка - 2Try                                                                                            |
Нейронка - Yolov4 (пытались продолжить обучить предыдущие веса (есть такой функционал))                 |
Датасет - 300 размеченных фоток, рандомные фотки ручек из гугла                                         |
Результат - ничего не видит                                                                             |
                                                                                                        |
        3 ПОПЫТКА                                                                                       |
Папка - 3Try                                                                                            |
Нейронка - Yolov4 (пытались продолжить обучить веса из 2 попытки)                                       |
Датасет - 300 размеченных фоток, рандомные фотки ручек из гугла                                         |
Результат - ничего не видит                                                                             |
                                                                                                        |
        4 ПОПЫТКА                                                                                       |
Папка - 4Try                                                                                            |
Нейронка - Yolov4 (пытались продолжить обучить веса из 3 попытки)                                       |
Датасет - 300 размеченных фоток, рандомные фотки ручек из гугла                                         |
Результат - ничего не видит                                                                             |
                                                                                                        |
        5 ПОПЫТКА                                                                                       |
Папка - 5Try                                                                                            |
Нейронка - Yolov4 Tiny (первая попытка свапнуть нейронку)                                               |
Датасет - 300 размеченных фоток, рандомные фотки ручек из гугла                                         |
Результат - ничего не видит                                                                             |
-------------------------------------------------------------------------------------------------------/

                                        Блок "Оно живое"
-------------------------------------------------------------------------------------------------------\
        6 ПОПЫТКА                                                                                       |
Папка - 6Try                                                                                            |
Нейронка - Yolov4 Tiny (начали обучать с нуля, поигрались побольше с кфг)                               |
Датасет - 300 размеченных фоток, создали собственный датасет конкертно 1 ручки                          |
Результат - наокнец-то начала распознавать все, что похоже на ручку по форме                            |
                                                                                                        |
        7 ПОПЫТКА                                                                                       |
Папка - 7Try                                                                                            |
Нейронка - Yolov4 Tiny (продолжение обучения предыдущих весов)                                          |
Датасет - 300 размеченных фоток, наш датасет                                                            |
Результат - стала намного лучше видеть границы объекта, виден прогресс                                  |
                                                                                                        |
        8 ПОПЫТКА                                                                                       |
Папка - 8Try                                                                                            |
Нейронка - Yolov4 Tiny (продолжение обучения предыдущих весов)                                          |
Датасет - 300 размеченных фоток, наш датасет                                                            |
Результат -                                                                                             |
-------------------------------------------------------------------------------------------------------/

"""

#"""
import cv2
import time

CONFIDENCE_THRESHOLD = 0.2
NMS_THRESHOLD = 0.4
COLORS = [(0, 255, 255), (255, 255, 0), (0, 255, 0), (255, 0, 0)]

class_names = []
with open("7Try/Classes.txt", "r") as f:
    class_names = [cname.strip() for cname in f.readlines()]

vc = cv2.VideoCapture(0)

net = cv2.dnn.readNet("7Try/yolov4.weights", "7Try/yolov4.cfg")
net.setPreferableBackend(cv2.dnn.DNN_BACKEND_CUDA)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CUDA_FP16)

model = cv2.dnn_DetectionModel(net)
model.setInputParams(size=(416, 416), scale=1 / 255, swapRB=True)

while cv2.waitKey(1) < 1:
    (grabbed, frame) = vc.read()
    if not grabbed:
        exit()

    start = time.time()
    classes, scores, boxes = model.detect(frame, CONFIDENCE_THRESHOLD, NMS_THRESHOLD)
    end = time.time()

    start_drawing = time.time()
    for (classid, score, box) in zip(classes, scores, boxes):
        color = COLORS[int(classid) % len(COLORS)]
        label = "%s : %f" % (class_names[classid], score)
        cv2.rectangle(frame, box, color, 2)
        cv2.putText(frame, label, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
    end_drawing = time.time()

    fps_label = "FPS: %.2f (drawtime of %.2fms)" % (
    1 / (end - start), (end_drawing - start_drawing) * 1000)
    cv2.putText(frame, fps_label, (0, 25), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)
    #frame = cv2.resize(frame, (1920, 1080))
    cv2.imshow("detections", frame)
#"""


#Обработка фотографий. Чисто потестить добавил, делитну потом
"""


import cv2

img = cv2.imread("7Try/9.jpg")

with open('7Try/Classes.txt', 'r') as f:
    classes = f.read().splitlines()

net = cv2.dnn.readNet("7Try/yolov4.weights", "7Try/yolov4.cfg")

model = cv2.dnn_DetectionModel(net)
model.setInputParams(scale=1 / 255, size=(416, 416), swapRB=True)

classIds, scores, boxes = model.detect(img, confThreshold=0.6, nmsThreshold=0.4)

for (classId, score, box) in zip(classIds, scores, boxes):
    cv2.rectangle(img, (box[0], box[1]), (box[0] + box[2], box[1] + box[3]),
                  color=(0, 255, 0), thickness=5)

    text = '%s: %.2f' % (classes[classId], score)
    cv2.putText(img, text, (box[0], box[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 3,
                color=(0, 255, 0), thickness=3)

img = cv2.resize(img, (720, 720))
cv2.imshow('Image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

"""