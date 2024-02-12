import cv2 as cv
import numpy as np
import argparse
import math

# Methods

def GetUnitLength(x,y,height):
    res = math.sqrt((x[0]-y[0])*(x[0]-y[0])+(x[1]-y[1])*(x[1]-y[1]))
    return res/height
#
#
def GetActualLength(x,unitLength):
    return x/unitLength

def GetDistance(x,y):
    distance = math.sqrt((x[0]-y[0])*(x[0]-y[0])+(x[1]-y[1])*(x[1]-y[1]))
    return(distance)

# Given 

Given_Height = 174

Given_AL_Neck_LShoulder = 21
Given_AL_Neck_RShoulder = 21
Given_AL_LShoulder_LElbow = 27
Given_AL_RShoulder_RElbow = 27
Given_AL_LElbow_LWrist = 27
Given_AL_RElbow_RWrist = 27
Given_AL_Neck_LHip = 45
Given_AL_Neck_RHip = 45
Given_AL_LHip_LKnee = 45
Given_AL_RHip_RKnee = 45
Given_AL_LKnee_LAnkle = 38
Given_AL_RKnee_RAnkle = 38


#   Logic

parser = argparse.ArgumentParser()
parser.add_argument('--input', help='bb.jpg')
parser.add_argument('--thr', default=0.2, type=float, help='Threshold value for pose parts heat map')
parser.add_argument('--width', default=368, type=int, help='Resize input to specific width.')
parser.add_argument('--height', default=368, type=int, help='Resize input to specific height.')

args = parser.parse_args()

BODY_PARTS = { "Nose": 0, "Neck": 1, "RShoulder": 2, "RElbow": 3, "RWrist": 4,
               "LShoulder": 5, "LElbow": 6, "LWrist": 7, "RHip": 8, "RKnee": 9,
               "RAnkle": 10, "LHip": 11, "LKnee": 12, "LAnkle": 13, "REye": 14,
               "LEye": 15, "REar": 16, "LEar": 17, "Background": 18 }

POSE_PAIRS = [ ["Neck", "RShoulder"], ["Neck", "LShoulder"], ["RShoulder", "RElbow"],
               ["RElbow", "RWrist"], ["LShoulder", "LElbow"], ["LElbow", "LWrist"],
               ["Neck", "RHip"], ["RHip", "RKnee"], ["RKnee", "RAnkle"], ["Neck", "LHip"],
               ["LHip", "LKnee"], ["LKnee", "LAnkle"], ["Neck", "Nose"], ["Nose", "REye"],
               ["REye", "REar"], ["Nose", "LEye"], ["LEye", "LEar"] ]

inWidth = args.width
inHeight = args.height

net = cv.dnn.readNetFromTensorflow("graph_opt.pb")

cap = cv.VideoCapture("image.jpg")

while cv.waitKey(1) < 0:
    hasFrame, frame = cap.read()
    if not hasFrame:
        cv.waitKey()
        break

    frameWidth = frame.shape[1]
    frameHeight = frame.shape[0]
    
    net.setInput(cv.dnn.blobFromImage(frame, 1.0, (inWidth, inHeight), (127.5, 127.5, 127.5), swapRB=True, crop=False))
    out = net.forward()
    out = out[:, :19, :, :]  

    assert(len(BODY_PARTS) == out.shape[1])
  

    points = []
    for i in range(len(BODY_PARTS)):
        heatMap = out[0, i, :, :]

  
        _, conf, _, point = cv.minMaxLoc(heatMap)
        x = (frameWidth * point[0]) / out.shape[3]
        y = (frameHeight * point[1]) / out.shape[2]
        points.append((int(x), int(y)) if conf > args.thr else None)

    for pair in POSE_PAIRS:
        partFrom = pair[0]
        partTo = pair[1]
        assert(partFrom in BODY_PARTS)
        assert(partTo in BODY_PARTS)

        idFrom = BODY_PARTS[partFrom]
        idTo = BODY_PARTS[partTo]

        if points[idFrom] and points[idTo]:
            cv.line(frame, points[idFrom], points[idTo], (0, 255, 0), 3)
            cv.ellipse(frame, points[idFrom], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)
            cv.ellipse(frame, points[idTo], (3, 3), 0, 0, 360, (0, 0, 255), cv.FILLED)

    t, _ = net.getPerfProfile()
    freq = cv.getTickFrequency() / 1000
    cv.putText(frame, '%.2fms' % (t / freq), (10, 20), cv.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0))


    cv.imshow('OpenPose using OpenCV', frame)
     
    for part, (x, y) in zip(BODY_PARTS.keys(), points):
        if (x, y) is not None:
            print(f"{part}: ({x}, {y})")
    
unitLength = GetUnitLength(points[0],points[10],Given_Height)

Neck_LShoulder = GetDistance(points[1],points[5])
Neck_RShoulder = GetDistance(points[1],points[2])
LShoulder_LElbow = GetDistance(points[5],points[6])
RShoulder_RElbow = GetDistance(points[2],points[3])
LElbow_LWrist = GetDistance(points[6],points[7])
RElbow_RWrist = GetDistance(points[3],points[4])
Neck_LHip = GetDistance(points[1],points[11])
Neck_RHip = GetDistance(points[1],points[8])
LHip_LKnee = GetDistance(points[11],points[12])
RHip_RKnee = GetDistance(points[8],points[9])
LKnee_LAnkle = GetDistance(points[9],points[10])
RKnee_RAnkle = GetDistance(points[12],points[13])

AL_Neck_LShoulder = GetActualLength(Neck_LShoulder,unitLength)
AL_Neck_RShoulder = GetActualLength(Neck_RShoulder,unitLength)
AL_LShoulder_LElbow = GetActualLength(LShoulder_LElbow,unitLength)
AL_RShoulder_RElbow = GetActualLength(RShoulder_RElbow,unitLength)
AL_LElbow_LWrist = GetActualLength(LElbow_LWrist,unitLength)
AL_RElbow_RWrist = GetActualLength(RElbow_RWrist,unitLength)
AL_Neck_LHip = GetActualLength(Neck_LHip,unitLength)
AL_Neck_RHip = GetActualLength(Neck_RHip,unitLength)
AL_LHip_LKnee = GetActualLength(LHip_LKnee,unitLength)
AL_RHip_RKnee = GetActualLength(RHip_RKnee,unitLength)
AL_LKnee_LAnkle = GetActualLength(LKnee_LAnkle,unitLength)
AL_RKnee_RAnkle = GetActualLength(RKnee_RAnkle,unitLength)
