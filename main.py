import cv2

def Prepare():
    classNames = []
    classFile = "SystemFiles/coco.names"
    with open(classFile, 'rt') as f:
        classNames = f.readlines()
    configPath = 'SystemFiles/ssd_mobilenet_v3_large_coco_2020_01_14.pbtxt'
    weightsPath = 'SystemFiles/frozen_inference_graph.pb'

    net = cv2.dnn_DetectionModel(weightsPath, configPath)
    net.setInputSize(320, 320)
    net.setInputScale(1.0 / 127.5)
    net.setInputMean((127.5, 127.5, 127.5))
    net.setInputSwapRB(True)
    return (classNames, net)

def ExplorePicture(img, net, classNames):
    classIds, confs, bbox = net.detect(img, confThreshold=0.5)
    arrayThings = []
    for classId, confidence, box in zip(classIds.flatten(), confs.flatten(), bbox):
        if classId == 74:
            arrayThings.append((box, classNames[classId - 1].strip()))
    return arrayThings

def FilterImage(arrayThings, filterName):
    count = 0
    for v in arrayThings:  # если 2 mouse, то выводим ошибку на сайт: не можем считать мышь или перегруженное кол-во обектов
        # - отправьте новую фотку
        if v[1] == filterName:
            count += 1
            if count > 1:
                return (False, coordinates) ## здесь можно кидать exception, но эт спорно
                break
            coordinates = v[0]

    if count:
        return (True, coordinates)
    return (False, [0, 0, 0, 0]) ## и здесь можно кидать exception

def GetObjectImg(coordinates, img):
    return img[coordinates[1]:coordinates[1] + coordinates[3],
           coordinates[0]:coordinates[0] + coordinates[2]]
    ## первая координата - x, вторая y; далее идут их изменения до конечной


img = cv2.imread('images/TestMouse.jpg')
classNames, net = Prepare()
arrayThings = ExplorePicture(img, net, classNames)
isFound, coordinates = FilterImage(arrayThings, "mouse") ## на этом этапе можно сделать метод с ошибками
if isFound:
    img = GetObjectImg(coordinates, img)
    cv2.imshow("cropped", img)
    cv2.waitKey(0)


