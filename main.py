# Creative IT Design I (2018 Spring Semester)
# Team No.3 (Seungjae Yoo, Mintae Kim, Kyungbin, Choi)
# Project 'Composing Helper Aparatus at Clay Shooting Gun for Blinds'
# Written by Mintae Kim, Seungjae Yoo
# Reference by synth.py by serge-rgb (Copyright (C) 2012 Sergio Gonzalez)
# Reference by Abc Xyz
# Reference by Oleg Kokorin
# Reference by Adrian Rosebrock
# Reference by Ryan


# 1. Libraries
import alsaaudio as alsa
import math
import array
import os

import time
import threading
import signal

import cv2
import picamera
import picamera.array
import numpy as np

# 2. Global Variables
# 2-1. Global Variables for both Threads
W_SIZE = 400
H_SIZE = 304

# 2-2. Global Variables for Beep Thread - I
RATE = 44100
FRAMES_PER_SECOND = 44100 # Number of Frames per second
CHUNKS_PER_SECOND = RATE / FRAMES_PER_SECOND # Number of Chunks per second

FREQUENCY_BASE = 587.33 # Frequency in Level 4
MAX_DURATION = 0.8 # Duration in Level 4

W_CENTER = W_SIZE / 2
H_CENTER = H_SIZE / 2 + 48

MAX_LEVEL = 4
LEVEL_DIVISION = 55.0 * (float(W_SIZE) / 800.0)

CENTER_POSITION_RANGE = 70.0 * (float(W_SIZE) / 800.0)
BEEP_ALARM_RANGE = 200.0 * (float(W_SIZE) / 800.0)

INFINITE_DISTANCE = 100000.0

# 3. Declaration of Class 'Position' & 'PositionList'
# Written by Mintae Kim and Seungjae Yoo
class Position : 
    def __init__(self,w,h) :
        self.w = w
        self.h = h

    def distance(self) :
        return ((self.w-W_CENTER)**2+(self.h-H_CENTER)**2)**0.5

    def tostring(self) :
        return str(self.w) + ", " + str(self.h) + ", " + str(self.distance())

class PositionList :
    def __init__(self) :
        self.list = []
        self.freq = 0
        self.duration = 1.0
        self.isHigh = 0
        self.isLeft = 0
        self.dist = INFINITE_DISTANCE

    def update(self) :
        self.setNearestPosition()
        self.setNearestPositionDistance()
        self.setFrequency()
        self.setDuration()
        self.determineHighLow()
        self.determineLeftRight()
        
    def push(self, p) :
        self.list.append(p)

    def clear(self) :
        self.list = []

    def setNearestPosition(self) :
        if (self.size() > 0) :
            minDistance = INFINITE_DISTANCE
            nearestPosition = None
            for p in self.list :
                if p.distance() < minDistance :
                    minDistance = p.distance()
                    nearestPosition = p
            self.n = nearestPosition
        else :
            self.n = None

    def setNearestPositionDistance(self) :
        if (self.n == None) :
            self.dist = INFINITE_DISTANCE
        else :
            self.dist = self.n.distance()

    def setFrequency(self) :
        if (self.size() > 0) :
            self.freq = FREQUENCY_BASE * (1.2 ** (float(MAX_LEVEL) - self.n.distance() / float(LEVEL_DIVISION)))
        else :
            self.freq = 0

    def setDuration(self) :
        if (self.size() > 0) :
            self.duration = MAX_DURATION / (float(MAX_LEVEL) - self.n.distance() / float(LEVEL_DIVISION))
        else :
            self.duration = 1.0

    def determineHighLow(self) :
        if (self.size() > 0) :
            if (abs(self.n.h - H_CENTER) < CENTER_POSITION_RANGE) :
                self.isHigh = 0
            elif (self.n.h < H_CENTER) :
                self.isHigh = 1
            else :
                self.isHigh = -1
        else :
            self.isHigh = 0

    def determineLeftRight(self) :
        if (self.size() > 0) :
            if (abs(self.n.w - W_CENTER) < CENTER_POSITION_RANGE) :
                self.isLeft = 0
            elif (self.n.w < W_CENTER) :
                self.isLeft = 1
            else :
                self.isLeft = -1
        else :
            self.isLeft = 0

    def size(self) :
        return len(self.list)
    
    def tostring(self) :
        string = "<BalloonList>\n"
        for p in self.list :
            string += p.tostring() + "\n"
        return string

# 2-3. Global Variables for Beep Thread - II
balloonList = PositionList()

# 4. Implementation of Threads
# 4-1. Declaration of Camera Thread (using OpenCV)
# Written by Seungjae Yoo
class opencvThread(threading.Thread) :
    def __init__(self) :
        threading.Thread.__init__(self)
        self.exit = threading.Event()
        print "opencvThread created"

    def openCVMain(self) :
        # global variables
        global balloonList
        
        # initial camera setting
        global W_SIZE
        global H_SIZE
        camera = picamera.PiCamera()
        camera.resolution = (W_SIZE, H_SIZE)
        stream = picamera.array.PiRGBArray(camera, size=(W_SIZE, H_SIZE))

        # make boundary for color detection
        lb = np.array([150, 60, 100], dtype=np.uint8)
        ub = np.array([180, 255, 255], dtype=np.uint8)
        lb1 = np.array([0, 60, 100], dtype=np.uint8)
        ub1 = np.array([10, 255, 255], dtype=np.uint8)

        time.sleep(0.1)
        for raw in camera.capture_continuous(stream, format="bgr", use_video_port=True) :

            if self.exit.is_set() : break # exit thread
            # image capture from screen
            frame = raw.array

            # image processing
            framehsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
            mask0 = cv2.inRange(framehsv, lb, ub)
            mask1 = cv2.inRange(framehsv, lb1, ub1)
            mask = cv2.bitwise_or(mask0, mask1)
            mask2 = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((20,20)))
            mask3, conts, h = cv2.findContours(mask2.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)

            cv2.circle(frame, (W_CENTER, H_CENTER), int(BEEP_ALARM_RANGE), (0, 0, 255), 3), 

            # make rectangle
            balloonList.clear()
            for i in range(len(conts)) :
                x, y, w, h = cv2.boundingRect(conts[i])
                if w*h < 40 : continue
                balloonList.push(Position(x+w/2,y+h/2))
                cv2.rectangle(frame, (x,y), (x+w,y+h), (255,0,0),3)


            balloonList.update()

            # show screen
            cv2.imshow('image', frame)
            # cv2.imshow('mask', mask)
            # cv2.imshow('mask2', mask2)
            cv2.waitKey(1)
            stream.truncate(0)

    def run(self) :
        self.openCVMain()

# 2-4. Global Variables for Calculation Thread & PlayBeep Thread
distance = INFINITE_DISTANCE # INFINITE
beepArray = []
isUpperSide = 0
isLeftSide = 0
isEmpty = True

# 4-2. Declaration of Calculation Thread
# Written by Mintae Kim
class calculationThread(threading.Thread):

    def __init__(self) :
        threading.Thread.__init__(self)
        self.exit = threading.Event()
        print "calcThread created"

    def synthSineBeep(self, freq, duration) :

        totalFrameNum = int(FRAMES_PER_SECOND * duration)
        samples = xrange(0, int(totalFrameNum / 2))

        beepSound = array.array('h', [int(2 ** 12 * math.sin(2 * math.pi * freq * sample * 1.0 / RATE)) for sample in samples])

        beepMute = array.array('h', [0 for sample in samples])

        beepArray = beepSound + beepMute

        return beepArray

    def calculationMain(self) :
        # global variables
        global balloonList

        global distance
        global beepArray
        global isUpperSide
        global isLeftSide
        global isEmpty
        
        while True :
            if self.exit.is_set() : break # exit thread

            isUpperSide = balloonList.isHigh
            isLeftSide = balloonList.isLeft
            distance = balloonList.dist

            print (balloonList.tostring())

            if (balloonList.size() == 0) :
                isEmpty = True
            else :
                isEmpty = False

            beepArray = self.synthSineBeep(balloonList.freq, balloonList.duration)
            
            time.sleep(0.5)

    def run(self) :
        self.calculationMain()

# 2-5. Global Variables for PlayBeep Thread
beepCount = 0

# 4-3. Declaration of Beep Thread
# Written by Mintae Kim
class beepThread(threading.Thread):
    def __init__(self) :
        threading.Thread.__init__(self)
        self.exit = threading.Event()
        print "beepThread created"

    def initPCM(self) :
        pcm = alsa.PCM(type=alsa.PCM_PLAYBACK,
                       mode=alsa.PCM_NORMAL)
        pcm.setchannels(1)
        return pcm

    def initMixer(self) :
        mixer = alsa.Mixer('PCM')
        return mixer

    def beepMain(self) :

        # global variables
        global beepArray
        global isUpperSide
        global isLeftSide
        global isEmpty
        global distance

        global beepCount

        pcm = self.initPCM()
        alsaMixer = self.initMixer()
        
        while True :
            if self.exit.is_set() : break # exit thread

            if (beepArray == []) : continue

            if (distance < BEEP_ALARM_RANGE) :
                pcm.write(beepArray.tostring())

            beepCount = beepCount + 1

            print ("beepcount:")
            print (beepCount)

            if (distance >= BEEP_ALARM_RANGE or beepCount == 5) :
                if (not isEmpty and distance >= BEEP_ALARM_RANGE) :
                    saveCurrentVolume = alsaMixer.getvolume()
                    alsaMixer.setvolume(90)

                    time.sleep(0.5)

                    if (isUpperSide == 1 and isLeftSide == 0) :
                        os.system("aplay N.wav")
                    elif (isUpperSide == 1 and isLeftSide == 1) :
                        os.system("aplay NW.wav")
                    elif (isUpperSide == 0 and isLeftSide == 1) :
                        os.system("aplay W.wav")
                    elif (isUpperSide == -1 and isLeftSide == 1) :
                        os.system("aplay SW.wav")
                    elif (isUpperSide == -1 and isLeftSide == 0) :
                        os.system("aplay S.wav")
                    elif (isUpperSide == -1 and isLeftSide == -1) :
                        os.system("aplay SE.wav")
                    elif (isUpperSide == 0 and isLeftSide == -1) :
                        os.system("aplay E.wav")
                    elif (isUpperSide == 1 and isLeftSide == -1) :
                        os.system("aplay NE.wav")

                    alsaMixer.setvolume(saveCurrentVolume[0])

                if (beepCount == 5) :
                    beepCount = 0

    def run(self) :
        self.beepMain()
            
#5. Class/Method for Signal Handling
# Written by Seungjae Yoo
class shut(Exception) :
    pass

def shutdown(sig, frame) :
    raise shut

#6. Main Method
# Written by Seungjae Yoo
def main() :
    # ctrl+c exception
    signal.signal(signal.SIGINT, shutdown)
    
    OpencvTh = opencvThread()
    CalcTh = calculationThread()
    BeepTh = beepThread()

    global balloonList
    
    try :
        OpencvTh.start()
        CalcTh.start()
        BeepTh.start()
        while True :
            time.sleep(0.5)

    # exception with ctrl+c
    except shut :
        OpencvTh.exit.set()
        CalcTh.exit.set()
        BeepTh.exit.set()
        OpencvTh.join()
        CalcTh.join()
        BeepTh.join()

if __name__ == "__main__" :
    main()
