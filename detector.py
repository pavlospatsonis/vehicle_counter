import cv2
from ultralytics import YOLO
import numpy as np
import librosa
import librosa.display
import numpy as np
import matplotlib.pyplot as plt
from moviepy.editor import VideoFileClip
import os 

# Array containing the actual counts of vehicles in the 10 evaluation videos
actual_counts = np.array([5, 4, 10, 7, 4, 6, 7, 5, 5, 4])
model_counts = []

path_to_videos = os.listdir("evaluation_vids/")
path_to_videos = ["evaluation_vids/" + x for x in path_to_videos]

for video_index in range(len(path_to_videos)):

    # Extracting audio from video
    video_path = path_to_videos[video_index]
    audio_path = 'audio.wav' # file doesn't need to already exist

    video = VideoFileClip(video_path) # Load the video file
    audio = video.audio # Extract the audio from the video
    audio.write_audiofile(audio_path) # Save the extracted audio to a file

    # Load the YOLOv8 model
    model = YOLO('yolov8n.pt')

    # Define vehicle classes 
    vehicle_classes = [1, 2, 3, 5, 6, 7]

    counter = 0 # counter for vehicles
    af_index = 0 # audio frame index

    # Open the video file
    cap = cv2.VideoCapture(video_path)

    pt1 = (100, 350) # detection line start point coordinates
    pt2 = (1100, 350)  # detection line end point coordinates

    offset = 2

    audio_data, sr = librosa.load(audio_path) # loading audio

    # Loop through the video frames
    while cap.isOpened():
        # Read a frame from the video
        success, frame = cap.read()
        audio_data, sr = librosa.load(audio_path, duration = 0.0333, offset = 0.0333*af_index) # loading audio
        num_of_samples = audio_data.size
        energy = librosa.feature.rms(y = audio_data, frame_length = num_of_samples, hop_length = num_of_samples)
        
        # Draw line
        cv2.line(img=frame, pt1=pt1, pt2=pt2, color=(0, 0, 255), thickness=1)

        if success:
            # Run YOLOv8 inference on the frame
            results = model(frame, classes = vehicle_classes)

            for result in results:
                boxes = result.boxes
                for box in boxes:
                    # # bound box for cv
                    x1, y1, x2, y2 = box.xyxy[0]
                    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                    
                    center_x = int((x1 + x2) / 2)
                    center_y = int((y1 + y2) / 2)
                    center_point = (center_x, center_y)
                    
                    # Red circle at the center point
                    cv2.circle(frame, center_point, 3, (0, 0, 255), -1)

                    # V e h i c l e  C o u n t e r
                    if center_point[1]<(pt1[1]+offset) and center_point[1]>(pt1[1]-offset) and energy[0,0] > 0.01 :
                        counter+=1
                        cv2.line(img=frame, pt1=pt1, pt2=pt2, color=(0, 255, 0), thickness=2)
                        cv2.putText(frame, str(counter), (x2,y2), cv2.FONT_HERSHEY_PLAIN, fontScale=2, color = (255,255,255), thickness=2)       

            # Visualize the results on the frame
            annotated_frame = results[0].plot()

            # Display the annotated frame
            cv2.imshow("YOLOv8 Inference", annotated_frame)

            # Break the loop if 'q' is pressed
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        else:
            # Break the loop if the end of the video is reached
            break
        
        af_index += 1

    # Release the video capture object and close the display window
    cap.release()
    cv2.destroyAllWindows()

    model_counts.append(counter)

AE = np.abs(np.array(actual_counts) - np.array(model_counts))
MAE = np.mean(AE)

print("MAE = " + str(MAE))