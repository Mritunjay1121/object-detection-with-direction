from ultralytics import YOLO
from ultralytics import YOLOv10

import cv2
import time
import numpy as np
import torch

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
    # Use YOLOv8x with optimizations
    # model = YOLO('yolov8x.pt')

    model = YOLOv10.from_pretrained("Ultralytics/YOLOv8")

    
    # Enable GPU if available and set half precision
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)
    
    if device.type != 'cpu':
        torch.backends.cudnn.benchmark = True
    
    tracker = ObjectTracker()
    video_path = "test2.mp4"
    cap = cv2.VideoCapture(video_path)
    
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    
    cv2.namedWindow("YOLOv8x Detection with Direction", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("YOLOv8x Detection with Direction", 1280, 720)
    
    direction_colors = {
        "left": (255, 0, 0),
        "right": (0, 255, 0),
        "up": (0, 255, 255),
        "down": (0, 0, 255),
        "stationary": (128, 128, 128)
    }
    
    # FPS calculation
    fps_start_time = time.time()
    fps_counter = 0
    fps_display = 0
    
    # Process every 2nd frame for better performance
    frame_skip = 2
    frame_count = 0
    
    print(f"Running on device: {device}")
    
    while cap.isOpened():
        success, frame = cap.read()
        if not success:
            break
        
        frame_count += 1
        if frame_count % frame_skip != 0:
            continue
        
        # Update FPS
        fps_counter += 1
        if time.time() - fps_start_time > 1:
            fps_display = fps_counter * frame_skip  # Adjust for skipped frames
            fps_counter = 0
            fps_start_time = time.time()
        
        # Optimize inference
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
        
        # Draw FPS
        cv2.putText(frame, f"FPS: {fps_display}", 
                   (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
        
        # Draw total detections
        cv2.putText(frame, f"Detections: {len(tracked_objects)}", 
                   (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 
                   1, (0, 255, 0), 2)
        
        for detection, obj_id, direction in tracked_objects:
            x1, y1, x2, y2, conf, cls = detection
            color = direction_colors.get(direction, (128, 128, 128))
            
            cv2.rectangle(frame, (int(x1), int(y1)), (int(x2), int(y2)), color, 2)
            
            label = f"{model.names[int(cls)]} {direction} {conf:.2f}"
            text_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)[0]
            
            cv2.rectangle(frame, 
                         (int(x1), int(y1) - text_size[1] - 10), 
                         (int(x1) + text_size[0], int(y1)), 
                         color, -1)
            
            cv2.putText(frame, label, 
                       (int(x1), int(y1) - 5), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.imshow("YOLOv8x Detection with Direction", frame)
        
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()