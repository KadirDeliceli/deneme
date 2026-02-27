from ultralytics import YOLO
import streamlit as st
from collections import defaultdict
import cv2
import tempfile

model = YOLO("yolo26n.pt")
video_file = st.file_uploader("Bir Video Yükleyin", type=["mp4","avi","mkv"])

if video_file is not None:

    tfile = tempfile.NamedTemporaryFile(delete=False)
    tfile.write(video_file.read())

    cap = cv2.VideoCapture(tfile.name)

    frame_placeholder = st.empty()
    class_counts = defaultdict(set)

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break  

        results = model.track(frame, persist=True)

        if results[0].boxes is not None and results[0].boxes.id is not None:
            track_ids = results[0].boxes.id.int().cpu().tolist()
            classes = results[0].boxes.cls.int().cpu().tolist()
            
            for track_id, _cls in zip(track_ids, classes):
                class_name = results[0].names[_cls]
                class_counts[class_name].add(track_id)

        annotated_frame = results[0].plot()

        y_pos = 30
        for class_name, ids in class_counts.items():
            text = f"{class_name}: {len(ids)}"
            cv2.putText(annotated_frame, text, (10, y_pos),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
            y_pos += 30

        frame_placeholder.image(
            cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)
        )

    cap.release()

    total_unique_ids = set()
    for ids in class_counts.values():
        total_unique_ids.update(ids)


    st.write("Toplam geçen nesne sayısı:", len(total_unique_ids))
