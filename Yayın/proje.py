import streamlit as st
import cv2
from ultralytics import YOLO
import numpy as np
import os

paths=r"C:\Users\dkadi\YOLO\Yayın\yolo26n.pt"
model_path=os.path.join(os.path.dirname(__file__), paths)
model=YOLO(model_path)

uploade_file=st.file_uploader("Bir Görüntü Yükleyin",type=["jpg","jpeg","png"])

if uploade_file is not None:
    file_bytes=np.array(bytearray(uploade_file.read()),dtype=np.uint8)
    img = cv2.imdecode(file_bytes, 1)
    
    results = model(img)[0]
    thresold = 0.5
    
    for result in results.boxes.data.tolist():
        x1,y1,x2,y2,score,class_id = result
        x1,y1,x2,y2,class_id = int(x1),int(y1),int(x2),int(y2),int(class_id)
        
        if score>thresold:
            cv2.rectangle(img, (x1,y1), (x2,y2) , (0,0,255), 2)
            class_name = results.names[class_id]
            score = score * 100
            
            text = f"{class_name}:%{score:.2f}"
            cv2.putText(img, text, (x1,y1-10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0,255,0) , 2 ,cv2.LINE_AA)
            
    st.image(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))            
        
        
        
        