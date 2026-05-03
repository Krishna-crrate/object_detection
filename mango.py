import cv2
import numpy as np
import tkinter as tk
from tkinter import messagebox, filedialog
from PIL import Image, ImageTk
import threading

# Load the pre-trained YOLO model
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")

with open("coco.names", "r") as f:
    classes = [line.strip() for line in f.readlines()]

# Global variables for video capture
cap = None
running = True
video_path = None

def capture_frames():
    global cap
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open camera.")
        return

    while running:
        ret, frame = cap.read()
        if ret:
            small_frame = cv2.resize(frame, (640, 480))
            process_frame(small_frame)

            # Show the frame in a window
            cv2.imshow("Object Detection", small_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_video(video_source):
    global running
    cap = cv2.VideoCapture(video_source)
    if not cap.isOpened():
        messagebox.showerror("Error", "Could not open video file.")
        return

    while running:
        ret, frame = cap.read()
        if not ret:
            break

        small_frame = cv2.resize(frame, (640, 480))
        process_frame(small_frame)

        # Show the frame in a window
        cv2.imshow("Object Detection", small_frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

def process_frame(small_frame):
    height, width, _ = small_frame.shape

    # Prepare the image for YOLO
    blob = cv2.dnn.blobFromImage(small_frame, 1/255, (416, 416), (0, 0, 0), swapRB=True, crop=False)
    net.setInput(blob)
    outputs = net.forward(net.getUnconnectedOutLayersNames())

    boxes, confidences, class_ids = [], [], []
    for output in outputs:
        for detection in output:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.3:  # Adjust confidence threshold
                x, y, w, h = detection[0:4] * np.array([width, height, width, height])
                x = int(x - w / 2)
                y = int(y - h / 2)
                boxes.append([x, y, int(w), int(h)])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indices = cv2.dnn.NMSBoxes(boxes, confidences, score_threshold=0.5, nms_threshold=0.4)

    if len(indices) > 0:
        for i in indices.flatten():
            x, y, w, h = boxes[i]
            label = f"{classes[class_ids[i]]}: {confidences[i]:.2f}"
            cv2.rectangle(small_frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
            cv2.putText(small_frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

def start_object_detection():
    global video_path
    if video_path:
        threading.Thread(target=process_video, args=(video_path,), daemon=True).start()
    else:
        threading.Thread(target=capture_frames, daemon=True).start()

def live_cam_detection():
    global video_path
    video_path = None  # Clear the video path to ensure live detection
    threading.Thread(target=capture_frames, daemon=True).start()

def quit_program():
    global running
    running = False
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
    root.destroy()

def update_scrolling_label():
    global label_x
    label_x -= 5  # Change this value to adjust speed
    if label_x < -scrolling_text_width:
        label_x = 1000  # Reset position to start again
    canvas.coords(scrolling_label_id, label_x, 10)  # Update the label's position (y = 10 for higher up)
    root.after(50, update_scrolling_label)  # Update every 50 ms

def upload_video():
    global video_path
    video_path = filedialog.askopenfilename(
        filetypes=[("Video files", "*.mp4;*.avi;*.mkv")]
    )
    if video_path:
        messagebox.showinfo("Video Selected", f"Selected video: {video_path}")

# Create main window
root = tk.Tk()
root.title("Object Detection")
root.geometry("1000x1000")  # Set window size
root.configure(bg="#f0f0f0")

# Load and display logo
logo_image = Image.open("logo.png")  # Ensure you have a logo image named "logo.png"
logo_image = logo_image.resize((1000, 800), Image.LANCZOS)  # Resize logo
logo_photo = ImageTk.PhotoImage(logo_image)

# Create label for logo
logo_label = tk.Label(root, image=logo_photo, bg="#f0f0f0")
logo_label.pack(pady=20)

# Create a label with "OBJECT DETECTION" text
title_label = tk.Label(root, text="OBJECT DETECTION", bg="#f0f0f0", font=("Algerian", 24, "bold"), fg="#333")
title_label.place(relx=0.5, rely=0.1, anchor=tk.CENTER)  # Position at the top center of the logo

# Create a canvas for the scrolling label
canvas = tk.Canvas(root, width=1000, height=50, bg="#f0f0f0", highlightthickness=0)
canvas.place(relx=0.5, rely=0.2, anchor=tk.CENTER)  # Position higher up

# Scrolling label text
scrolling_text = "--------OBJECT DETECTION IS ACTIVATED-------                                                                    -------OBJECT DETECTION IS ACTIVATED-------"
scrolling_label_id = canvas.create_text(1000, 10, text=scrolling_text, font=("Arial Rounded MT Bold", 25), fill="#333", anchor="nw")

# Calculate the width of the scrolling text
scrolling_text_width = canvas.bbox(scrolling_label_id)[2] - canvas.bbox(scrolling_label_id)[0]

# Initial x position for scrolling
label_x = 1000  # Start off-screen to the right

# Start scrolling
update_scrolling_label()

# Create buttons and place them on the logo
start_button = tk.Button(root, text="VIDEO DETECTION", command=start_object_detection, bg="#4CAF50", fg="white", font=("Cooper Black", 16), width=20, height=2)
start_button.place(relx=0.3, rely=0.7, anchor=tk.CENTER)  # Centered on logo

live_cam_button = tk.Button(root, text="Live Cam Detection", command=live_cam_detection, bg="#2196F3", fg="white", font=("Cooper Black", 16), width=20, height=2)
live_cam_button.place(relx=0.7, rely=0.7, anchor=tk.CENTER)  # Centered below the start button

quit_button = tk.Button(root, text="Quit", command=quit_program, bg="#f44336", fg="white", font=("Cooper Black", 16), width=20, height=2)
quit_button.place(relx=0.5, rely=0.85, anchor=tk.CENTER)  # Centered below the live cam button

# Add an upload button at the top-right corner of the logo
upload_button = tk.Button(root, text="Upload Video", command=upload_video, bg="#2196F3", fg="white", font=("Cooper Black", 16), width=15, height=2)
upload_button.place(relx=0.95, rely=0.05, anchor=tk.NE)  # Position at top-right corner

# Run the application
root.mainloop()
