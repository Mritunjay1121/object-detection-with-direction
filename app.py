import streamlit as st
from ultralytics import YOLO
import cv2
import time
import numpy as np
import torch
from PIL import Image
import tempfile
import warnings
warnings.filterwarnings('ignore')

def get_direction(old_center, new_center, min_movement=10):
    if old_center is None or new_center is None:
        return "stationary"
    
    dx = new_center[0] - old_center[0]
    dy = new_center[1] - old_center[1]
    
    if abs(dx) < min_movement and abs(dy) < min_movement:
        return "stationary"
    
    if abs(dx) > abs(dy):
        return "right" if dx > 0 else "left"
    else:
        return "down" if dy > 0 else "up"

class ObjectTracker:
    def __init__(self):
        self.tracked_objects = {}
        self.object_count = {}
    
    def update(self, detections):
        current_objects = {}
        results = []
        
        for detection in detections:
            x1, y1, x2, y2 = detection[0:4]
            center = ((x1 + x2) // 2, (y1 + y2) // 2)
            class_id = detection[5]
            
            object_id = f"{class_id}_{len(self.object_count.get(class_id, []))}"
            
            min_dist = float('inf')
            closest_id = None
            
            for prev_id, prev_data in self.tracked_objects.items():
                if prev_id.split('_')[0] == str(class_id):
                    dist = np.sqrt((center[0] - prev_data['center'][0])**2 + 
                                 (center[1] - prev_data['center'][1])**2)
                    if dist < min_dist and dist < 100:
                        min_dist = dist
                        closest_id = prev_id
            
            if closest_id:
                object_id = closest_id
            else:
                if class_id not in self.object_count:
                    self.object_count[class_id] = []
                self.object_count[class_id].append(object_id)
            
            prev_center = self.tracked_objects.get(object_id, {}).get('center', None)
            direction = get_direction(prev_center, center)
            
            current_objects[object_id] = {
                'center': center,
                'direction': direction,
                'detection': detection
            }
            
            results.append((detection, object_id, direction))
        
        self.tracked_objects = current_objects
        return results


    
def main():
    st.title("Real-time Object Detection with Direction")
    
    # File uploader for video
    uploaded_file = st.file_uploader("Choose a video file", type=['mp4', 'avi', 'mov'])
    
    # Add start button
    start_detection = st.button("Start Detection")
    
    # Add stop button
    stop_detection = st.button("Stop Detection")
    
    if uploaded_file is not None and start_detection:
        # Create a session state to track if detection is running
        if 'running' not in st.session_state:
            st.session_state.running = True
            
        # Save uploaded file temporarily
        tfile = tempfile.NamedTemporaryFile(delete=False)
        tfile.write(uploaded_file.read())
        
        # Load model
        with st.spinner('Loading model...'):
            model = YOLO('yolov8x.pt',verbose=False)
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
            model.to(device)
        
        tracker = ObjectTracker()
        cap = cv2.VideoCapture(tfile.name)
        
        direction_colors = {
            "left": (255, 0, 0),
            "right": (0, 255, 0),
            "up": (0, 255, 255),
            "down": (0, 0, 255),
            "stationary": (128, 128, 128)
        }
        
        # Create placeholder for video frame
        frame_placeholder = st.empty()
        # Create placeholder for detection info
        info_placeholder = st.empty()
        
        st.success("Detection Started!")
        
        while cap.isOpened() and st.session_state.running:
            success, frame = cap.read()
            if not success:
                break
            
            # Run detection
            results = model(frame, 
                          conf=0.25,
                          iou=0.45,
                          max_det=20,
                          verbose=False)[0]
            
            detections = []
            for box in results.boxes.data:
                x1, y1, x2, y2, conf, cls = box.tolist()
                detections.append([int(x1), int(y1), int(x2), int(y2), float(conf), int(cls)])
            
            tracked_objects = tracker.update(detections)
            
            # Dictionary to store detection counts
            detection_counts = {}
            
            for detection, obj_id, direction in tracked_objects:
                x1, y1, x2, y2, conf, cls = detection
                color = direction_colors.get(direction, (128, 128, 128))
                
                cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
                
                label = f"{model.names[int(cls)]} {direction} {conf:.2f}"
                # Increased font size and thickness
                font_scale = 1.2
                thickness = 3
                text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)[0]
                
                # Increased padding for label background
                padding_y = 15
                cv2.rectangle(frame, 
                             (int(x1), int(y1) - text_size[1] - padding_y), 
                             (int(x1) + text_size[0], int(y1)), 
                             color, -1)
                
                cv2.putText(frame, label, 
                           (int(x1), int(y1) - 5), 
                           cv2.FONT_HERSHEY_SIMPLEX, 
                           font_scale, 
                           (255, 255, 255), 
                           thickness)
                
                # Count detections by class
                class_name = model.names[int(cls)]
                detection_counts[class_name] = detection_counts.get(class_name, 0) + 1
            
            # Convert BGR to RGB
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Update frame
            frame_placeholder.image(frame_rgb, channels="RGB", use_column_width=True)
            
            # Update detection info
            info_text = "Detected Objects:\n"
            for class_name, count in detection_counts.items():
                info_text += f"{class_name}: {count}\n"
            info_placeholder.text(info_text)
            
            # Check if stop button is pressed
            if stop_detection:
                st.session_state.running = False
                break
            
        cap.release()
        st.session_state.running = False
        st.warning("Detection Stopped")
        
    elif uploaded_file is None and start_detection:
        st.error("Please upload a video file first!")

if __name__ == "__main__":
    main()