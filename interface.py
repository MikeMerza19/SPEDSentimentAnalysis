from cgitb import text
from email.mime import image
import string
from turtle import textinput
from PyQt5.QtCore import QObject
from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5 import uic
from PyQt5.QtCore import *
import cv2 
import pickle
# import mediapipe as mp
import numpy as np
import functions as func

import pyttsx3
# from PyQt5.QtChart import QChart, QChartView, QPieSeries


import os
# import tensorflow as tf
# from object_detection.utils import label_map_util
# from object_detection.utils import visualization_utils as viz_utils
# from object_detection.builders import model_builder
# from object_detection.utils import config_util

# import time
# import sys
import shlex
import getpass
import socket

# import asyncio

from cvzone.HandTrackingModule import HandDetector
from cvzone.ClassificationModule import Classifier
import math

import time

detector = HandDetector(maxHands=2)
classifier = Classifier("Model/keras_model.h5" , "Model/labels.txt")
offset = 20
imgSize = 300
imgSize2 = 200

#load, open, read question.txt
q_file = open('Questions.txt', 'r')
data = q_file.read()
data_list = data.split('\n\n')
q_file.close()
q_current = 1

q_logs = []


class MyGUI(QMainWindow):
    def __init__(self):
        super(MyGUI, self).__init__()
        uic.loadUi("signLangGUI.ui", self)
        self.show()

        self.btnStart.clicked.connect(self.VideoCapStart)
        self.btnStop.clicked.connect(self.VideoCapStop)
        self.btnProcessSentiment.clicked.connect(self.processSentimentClicked)        
        self.btnClear.clicked.connect(self.clearContentClicked)
        
        self.txtOutputTemp.setVisible(False)
        self.txtCheckDup.setVisible(False)

        self.lblQuestions.setText(data_list[0])
        self.lblPageNum.setText(str(q_current))
        for count in range(len(data_list)):
            self.cboPage.addItem(str(count+1))

        self.btnFirst.clicked.connect(self.firstQClicked)
        self.btnPrev.clicked.connect(self.prevQClicked)
        self.btnNext.clicked.connect(self.nextQClicked)
        self.btnLast.clicked.connect(self.lastQClicked)
        self.btnGo.clicked.connect(self.goQClicked)

        self.btnSpeech.clicked.connect(self.speechClicked)
        # self.btnSpeechStop.clicked.connect(self.speechStopClicked)

        self.btnExport.clicked.connect(self.exportLogsClicked)
        self.btnReset.clicked.connect(self.resetLogsClicked)

    def firstQClicked(self):
        try:            
            global q_current      
            q_current = 1  
            self.lblQuestions.setText(data_list[0])
            self.lblPageNum.setText(str(q_current))
        except Exception as ex:
            print(ex)

    def prevQClicked(self):
        try:            
            global q_current  
            if q_current != 1:      
                q_current = q_current - 1
                self.lblQuestions.setText(data_list[q_current-1])
                self.lblPageNum.setText(str(q_current))
        except Exception as ex:
            print(ex)

    def nextQClicked(self):
        try:            
            global q_current  
            if q_current != len(data_list):      
                q_current = q_current + 1
                self.lblQuestions.setText(data_list[q_current-1])
                self.lblPageNum.setText(str(q_current))
        except Exception as ex:
            print(ex)


    def lastQClicked(self): 
        try:            
            global q_current       
            q_current = len(data_list)  
            self.lblQuestions.setText(data_list[-1])
            self.lblPageNum.setText(str(q_current))
        except Exception as ex:
            print(ex)

    def goQClicked(self):
        try:            
            global q_current      
            q_current = 1  
            self.lblQuestions.setText(data_list[0])
            self.lblPageNum.setText(str(q_current))
        except Exception as ex:
            print(ex)

    def speechClicked(self):
        try:            
            engine = pyttsx3.init()
            engine.say(self.lblQuestions.text())
            engine.runAndWait()
        except Exception as ex:
            print(ex)

    def exportLogsClicked(self):
        try:            
            global q_logs
            fileName = "Logs/Question_Logs_" + time.strftime("%Y%m%d-%H%M%S") + ".txt"
            with open(fileName, 'w') as output:
                for row in q_logs:                    
                    output.write(row + '\n\n')
            
        except Exception as ex:
            print(ex)

    def resetLogsClicked(self):
        try:            
            global q_logs            
            q_logs.clear()
            self.txtLogs.clear()
        except Exception as ex:
            print(ex)


    def VideoCapStart(self):                 
        self.Worker1 = Worker1()
        self.Worker1.start()
        self.Worker1.ImageUpdate.connect(self.ImageUpdateSlot)
        self.Worker1.ImageUpdate2.connect(self.ImageUpdateSlot2)
        self.Worker1.PredictedText.connect(self.PredictedTextSlot)

    def VideoCapStop(self):
        self.Worker1.stop()
        self.Worker1.ThreadActive = False
        
    def ImageUpdateSlot(self, Image):
        self.FeedLabel.setPixmap(QPixmap.fromImage(Image))

    def ImageUpdateSlot2(self, Image):
        self.FeedLabel_2.setPixmap(QPixmap.fromImage(Image))

    def PredictedTextSlot(self, Text):
        self.lblPrediction.setText(Text.text())
        checkVal = self.txtOutputTemp.toPlainText()
        checkDup = self.txtCheckDup.toPlainText()      
        outputArr = checkVal.split(',')
        if(len(outputArr) > 1):
            del outputArr[-1]
            
        # print(outputArr[-1])
        if not outputArr:
            self.txtOutput.insertPlainText(Text.text() + " ")
            self.txtOutputTemp.insertPlainText(Text.text() + ",")
        else:
            if checkVal != "":                
                if Text.text() != outputArr[-1]:
                    if Text.text() == "clear":
                        del outputArr[-1]
                        if len(outputArr) > 1:
                            del outputArr[0]
                        self.txtOutput.clear()
                        self.txtOutputTemp.clear()
                        self.txtOutput.insertPlainText(' '.join(outputArr))                    
                        self.txtOutputTemp.insertPlainText(','.join(outputArr))
                        print(','.join(outputArr))
                    else:                        
                        cursor = self.txtOutput.textCursor()
                        cursor.movePosition(cursor.End)
                        self.txtOutput.setTextCursor(cursor)
                        self.txtOutput.insertPlainText(Text.text() + " ")   
                        self.txtOutputTemp.insertPlainText(Text.text() + ",")
            else:
                if Text.text() != "clear":
                    del outputArr[0]
                    outputArr.append(Text.text())
                    self.txtOutput.clear()
                    self.txtOutputTemp.clear()
                    self.txtOutput.insertPlainText(' '.join(outputArr))                    
                    self.txtOutputTemp.insertPlainText(','.join(outputArr))
                                      
    def processSentimentClicked(self):                                     
        if self.lblSentimentAnalysis.text() != "---":
            self.Worker2.stop()
            self.lblSentimentAnalysis.setText("---")    

            self.progNegative.setValue(0)
            self.progNeutral.setValue(0)            
            self.progPositive.setValue(0)

            self.lblPercentage.setText("---")        

        self.Worker2 = Worker2()  
        self.Worker2.start()                   
        self.Worker2.Output = self.txtOutput.toPlainText()    
        self.Worker2.PredictSentiment.connect(self.PredictedSentimentSlot)  
        self.Worker2.PredictProbability.connect(self.PredictedProbabilitySlot)                       

    def clearContentClicked(self):
        if self.lblSentimentAnalysis.text() != "---":
            self.Worker2.stop()
            self.txtOutput.clear()
            self.lblSentimentAnalysis.setText("---")    
        else:
            self.txtOutput.clear()
            self.lblSentimentAnalysis.setText("---")
        
    def PredictedSentimentSlot(self, Text):
        try:
            global q_logs
            global q_current            
            logEntry = "Question " + str(q_current) + ": " + Text.text()
        
            self.lblSentimentAnalysis.setText(Text.text())              

            if q_logs:
                if q_logs[-1] != logEntry:        
                    q_logs.append(logEntry)
                    self.txtLogs.clear()
                    self.txtLogs.insertPlainText('\n\n'.join(q_logs))
            else:
                q_logs.append(logEntry)
                self.txtLogs.clear()
                self.txtLogs.insertPlainText('\n\n'.join(q_logs))            
        except Exception as ex: 
            self.lblSentimentAnalysis.setText("---")

            self.progNegative.setValue(0)
            self.progNeutral.setValue(0)            
            self.progPositive.setValue(0)

            self.lblPercentage.setText("---")
            print(ex)

    def PredictedProbabilitySlot(self, Text):
        try:            
            text_value = str(Text.text())
            list_data = text_value.split(", ")            
            data_as_float = [round((float(item)*100), 2) for item in list_data]
            
            self.progNegative.setValue(int(data_as_float[0]))
            self.progNeutral.setValue(int(data_as_float[1]))            
            self.progPositive.setValue(int(data_as_float[2]))

            if data_as_float[0] > data_as_float[1] and data_as_float[0] > data_as_float[2]:
                self.lblPercentage.setText(str(data_as_float[0]) + "%")
            elif data_as_float[1] > data_as_float[0] and data_as_float[1] > data_as_float[2]:
                self.lblPercentage.setText(str(data_as_float[1]) + "%")
            elif data_as_float[2] > data_as_float[0] and data_as_float[2] > data_as_float[1]:
                self.lblPercentage.setText(str(data_as_float[2]) + "%")
        except Exception as ex:
            self.progNegative.setValue(0)
            self.progNeutral.setValue(0)            
            self.progPositive.setValue(0)

            self.lblPercentage.setText("---")

            print(ex)

class Worker2(QThread):
    PredictSentiment = pyqtSignal(QLabel)   
    PredictProbability = pyqtSignal(QLabel)    
    Output = ""

    def run(self):
        self.ThreadActive = True

        while self.ThreadActive:
            output_value = self.Output
            sentiment_result, sentiment_proba = func.PredictSentiment(output_value)

            res = sentiment_result
            res2 = sentiment_proba
            test = QLabel(res)
            test2 = QLabel(res2)

            self.PredictSentiment.emit(test) 
            self.PredictProbability.emit(test2) 


            # self.ThreadActive = False        

    def stop(self):
        self.ThreadActive = False
        self.quit()

class Worker1(QThread):
    ImageUpdate = pyqtSignal(QImage)    
    ImageUpdate2 = pyqtSignal(QImage)    
    PredictedText = pyqtSignal(QLabel)

    def run(self):
        self.ThreadActive = True

        # Setup capture
        cap = cv2.VideoCapture(0)

        labels = ["Clear","Dislike","Hello","I","Like","Love","No","Okay","Peace","Thank You","Yes","You"]

        while self.ThreadActive:            
            while True:
                try:                    
                    success, img = cap.read()
                    imgOutput = img.copy()
                    hands, img = detector.findHands(img)
                    if hands:
                        hand = hands[0]
                        x, y, w, h = hand['bbox']

                        imgWhite = np.ones((imgSize, imgSize, 3), np.uint8)*255

                        imgCrop = img[y-offset:y + h + offset, x-offset:x + w + offset]
                        imgCropShape = imgCrop.shape

                        aspectRatio = h / w

                        if aspectRatio > 1:
                            k = imgSize / h
                            wCal = math.ceil(k * w)
                            imgResize = cv2.resize(imgCrop, (wCal, imgSize))
                            imgResizeShape = imgResize.shape
                            wGap = math.ceil((imgSize-wCal)/2)
                            imgWhite[:, wGap: wCal + wGap] = imgResize
                            prediction , index = classifier.getPrediction(imgWhite, draw= False)
                            print(prediction, index)

                        else:
                            k = imgSize / w
                            hCal = math.ceil(k * h)
                            imgResize = cv2.resize(imgCrop, (imgSize, hCal))
                            imgResizeShape = imgResize.shape
                            hGap = math.ceil((imgSize - hCal) / 2)
                            imgWhite[hGap: hCal + hGap, :] = imgResize
                            prediction , index = classifier.getPrediction(imgWhite, draw= False)

                        cv2.putText(imgOutput,labels[index],(x,y-30),cv2.FONT_HERSHEY_COMPLEX,2,(0,0,0),2) 
                        cv2.rectangle(imgOutput,(x-offset,y-offset),(x + w + offset, y+h + offset),(0,255,0),4) 
                
                    
                        Imagee = cv2.cvtColor(imgOutput, cv2.COLOR_BGR2RGB)
                        FlippedImage = cv2.flip(Imagee, 1)       
                        ConvertToQtFormat = QImage(Imagee.data, Imagee.shape[1], Imagee.shape[0], QImage.Format_RGB888)
                        Pic = ConvertToQtFormat.scaled(640, 480, Qt.KeepAspectRatio)

                        Imagee2 = cv2.cvtColor(imgWhite, cv2.COLOR_BGR2RGB)
                        FlippedImage2 = cv2.flip(Imagee2, 1)       
                        ConvertToQtFormat2 = QImage(Imagee2.data, Imagee2.shape[1], Imagee2.shape[0], QImage.Format_RGB888)
                        Pic2 = ConvertToQtFormat2.scaled(340, 280, Qt.KeepAspectRatio)

                        self.ImageUpdate.emit(Pic)  
                        self.ImageUpdate2.emit(Pic2)  

                        try:
                            concatText = labels[index]
                            test = QLabel(concatText)
                            self.PredictedText.emit(test)   
                        except:
                            concatText = "Value Error"
                            test = QLabel(concatText)
                            self.PredictedText.emit(test)
                except Exception as ex:
                    print(ex)

        cap.release()

    def stop(self):
        self.ThreadActive = False
        self.quit()

def main():
    app = QApplication([])
    window = MyGUI()
    app.exec_()

if __name__ == '__main__':
    main()
