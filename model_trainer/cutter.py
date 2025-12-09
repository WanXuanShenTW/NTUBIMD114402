import cv2
import os
import tkinter as tk
from tkinter import filedialog, messagebox

def select_video():
    video_path = filedialog.askopenfilename(
        title="Select Input Video", 
        filetypes=[("Video Files", "*.mp4 *.avi *.mov")]
    )
    if video_path:
        video_entry.delete(0, tk.END)
        video_entry.insert(0, video_path)

def select_output_dir():
    output_dir = filedialog.askdirectory(title="Select Output Folder")
    if output_dir:
        output_entry.delete(0, tk.END)
        output_entry.insert(0, output_dir)

def split_into_segments():
    video_path = video_entry.get()
    output_dir = output_entry.get()
    
    try:
        segment_frames = int(segment_entry.get())
        if segment_frames <= 0:
            raise ValueError
    except ValueError:
        messagebox.showerror("Error", "Please enter a valid positive integer for frames per segment.")
        return

    if not os.path.exists(video_path):
        messagebox.showerror("Error", "Please select an existing video file.")
        return
    if not os.path.isdir(output_dir):
        messagebox.showerror("Error", "Please select a valid output folder.")
        return

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        messagebox.showerror("Error", "Unable to open input video.")
        return

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use mp4v codec
    
    segment_index = 0
    total_written = 0

    while True:
        output_filename = os.path.join(output_dir, f"segment_{segment_index+1}.mp4")
        out = cv2.VideoWriter(output_filename, fourcc, fps, (width, height))
        
        written = 0
        while written < segment_frames:
            ret, frame = cap.read()
            if not ret:
                break
            out.write(frame)
            written += 1
            total_written += 1
        out.release()
        segment_index += 1
        
        # Break if end of file
        if written < segment_frames:
            break

    cap.release()
    messagebox.showinfo("Done", f"Video has been split into {segment_index} segments, {total_written} frames processed in total.")

if __name__ == "__main__":
    root = tk.Tk()
    root.title("Video Splitter (by Fixed Frame Count)")

    tk.Label(root, text="Input Video File:").grid(row=0, column=0, padx=10, pady=10)
    video_entry = tk.Entry(root, width=50)
    video_entry.grid(row=0, column=1, padx=10, pady=10)
    tk.Button(root, text="Browse Video", command=select_video).grid(row=0, column=2, padx=10, pady=10)

    tk.Label(root, text="Output Folder:").grid(row=1, column=0, padx=10, pady=10)
    output_entry = tk.Entry(root, width=50)
    output_entry.grid(row=1, column=1, padx=10, pady=10)
    tk.Button(root, text="Browse Folder", command=select_output_dir).grid(row=1, column=2, padx=10, pady=10)

    tk.Label(root, text="Frames per Segment:").grid(row=2, column=0, padx=10, pady=10)
    segment_entry = tk.Entry(root, width=10)
    segment_entry.grid(row=2, column=1, padx=10, pady=10)
    segment_entry.insert(0, "30")  # Default value is 30 frames

    tk.Button(root, text="Start Splitting", command=split_into_segments).grid(row=3, column=1, padx=10, pady=20)

    root.mainloop()