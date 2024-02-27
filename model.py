from ultralytics import YOLO
import yaml
import cv2 as cv 
import matplotlib.pyplot as plt

model = YOLO("yolov8n.pt")
results = model.train(data="/Users/oumaima/Downloads/Car Logos Detection/data.yaml",epochs=100)
#results.predict("/Users/oumaima/Downloads/civic.png.webp" , save = True , save_txt = True)
model.save("/Users/oumaima/Downloads/model/weights.pth")  
