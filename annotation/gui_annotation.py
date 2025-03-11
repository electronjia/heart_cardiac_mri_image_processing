
import tkinter as tk
import cv2
from PIL import Image, ImageTk, ImageDraw
import csv
import os
import numpy as np
import pandas as pd
import pydicom
from config import *

class AnnotationGUI:
    def __init__(self, frame, save_path, sampling_rate, scale_factor=2):
        self.frame = frame
        self.save_path = save_path
        self.drawing = False
        self.last_x = None
        self.last_y = None
        self.pixels = []
        self.running = True  # Flag to indicate if GUI is running
        self.original_frame = frame
        self.sampling_rate = sampling_rate
        self.scale_factor = scale_factor
        self.x_orig = 0
        self.y_orig = 0


        # Ensure it's a valid NumPy array and properly scaled
        if isinstance(self.original_frame, np.ndarray):
            if self.original_frame.dtype != np.uint8:
                self.original_frame = (self.original_frame / np.max(self.original_frame) * 255).astype(np.uint8)
        else:
            raise ValueError("original_frame is not a valid NumPy array")

        # Initialize the GUI window
        self.root = tk.Tk()
        self.root.title("Pixel-by-Pixel Annotation")

        # Make the window fullscreen
        self.root.attributes("-fullscreen", True)

        # Bind the "Escape" key to exit fullscreen
        self.root.bind("<Escape>", self.exit_fullscreen)

        # Convert the OpenCV image to PIL format for Tkinter
        self.image = Image.fromarray(self.original_frame)

        # Scale the image
        width, height = self.image.size
        new_size = (width * self.scale_factor, height * self.scale_factor)
        self.scaled_image = self.image.resize(new_size, Image.LANCZOS)

        self.tk_image = ImageTk.PhotoImage(self.scaled_image)

        # Create a Canvas widget
        self.canvas = tk.Canvas(self.root, width=self.scaled_image.width, height=self.scaled_image.height)

        # Center the canvas on the screen
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        x_offset = (screen_width - self.scaled_image.width) // 2
        y_offset = (screen_height - self.scaled_image.height) // 2
        self.canvas.place(x=x_offset, y=y_offset)
        self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW)

        # Keep a reference
        self.canvas.image = self.tk_image  # This prevents garbage collection

        # Bind mouse events
        self.canvas.bind("<ButtonPress-1>", self.start_draw)
        self.canvas.bind("<B1-Motion>", self.draw_pixel)
        self.canvas.bind("<ButtonRelease-1>", self.end_draw)

        # Add save button
        save_button = tk.Button(self.root, text="Save Annotation", command=self.save_annotation)
        save_button.place(relx=0.4, rely=0.05, anchor="nw")

        # Add a button to load the polygon from CSV
        load_button = tk.Button(self.root, text="Show Annotation", command=self.load_and_map_annotation)
        load_button.place(relx=0.4, rely=0.1, anchor="nw")


        # Add next button
        next_button = tk.Button(self.root, text=f"Next {self.sampling_rate}", command=self.next_frame)
        next_button.place(relx=0.5, rely=0.05, anchor="nw")

        # Add previous button
        previous_button = tk.Button(self.root, text=f"Previous {self.sampling_rate}", command=self.previous_frame)
        previous_button.place(relx=0.6, rely=0.05, anchor="nw")

        # Add exit button
        exit_button = tk.Button(self.root, text="Exit", command=self.destroy)
        exit_button.place(relx=0.7, rely=0.05, anchor="nw")

        # Add start over button
        star_over_button = tk.Button(self.root, text="Start Over", command=self.start_over)
        star_over_button.place(relx=0.7, rely=0.1, anchor="nw")

        # Add redo button
        redo_button = tk.Button(self.root, text="Redo", command=self.redo)
        redo_button.place(relx=0.7, rely=0.15, anchor="nw")

        # Add a button to forward 5 frames
        five_next_button = tk.Button(self.root, text=f"Next {2*self.sampling_rate}", command=self.next_N_frames)
        five_next_button.place(relx=0.5, rely=0.1, anchor="nw")

        # Add a button to backtrack 5 frames
        five_previous_button = tk.Button(self.root, text=f"Previous {2*self.sampling_rate}", command=self.previous_N_frames)
        five_previous_button.place(relx=0.6, rely=0.1, anchor="nw")

        # Add a button to show next video
        next_video_button = tk.Button(self.root, text="Next Video", command=self.get_next_video)
        next_video_button.place(relx=0.5, rely=0.15, anchor="nw")

        # Add a button to show previous video
        previous_video_button = tk.Button(self.root, text="Previous Video", command=self.get_previous_video)
        previous_video_button.place(relx=0.6, rely=0.15, anchor="nw")


    def exit_fullscreen(self, event=None):
        """Exit fullscreen mode."""
        self.root.attributes("-fullscreen", False)

    def start_draw(self, event):
        self.drawing = True
        self.last_x = event.x
        self.last_y = event.y

    def draw_pixel(self, event):
        if self.drawing:
            self.canvas.create_line(self.last_x, self.last_y, event.x, event.y, fill="red", width=2)
            self.pixels.append((event.x, event.y))
            self.last_x = event.x
            self.last_y = event.y

    def next_frame(self):
        
        self.next_frame_val = 1
        self.root.destroy()

    def previous_frame(self):
        self.root.destroy()
        self.previous_frame_val = -1

    def next_N_frames(self):
        # This will be multiplied by X value used to skip between frames of video
        self.skip_frames_factor = 2
        
        self.root.destroy()

    def previous_N_frames(self):
        self.root.destroy()

        # This will be multiplied by X value used to skip between frames of video
        self.previous_frames_factor = 2
         
    def start_over(self):
        self.root.destroy()
        self.start_over_val = True

    def redo(self):
        self.canvas.delete("all")
        self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW)
        self.pixels = []

    def end_draw(self, event):
        self.drawing = False

    def get_next_video(self):
        self.next_video = True
        self.root.destroy()

    def get_previous_video(self):
        self.previous_video = True
        self.root.destroy()

    def save_annotation(self):
        if self.pixels:

            if self.pixels[0] != self.pixels[-1]:
                self.pixels.append(self.pixels[0])

            # Create a blank image to store the mask
            mask = Image.new("RGB", (self.scaled_image.width, self.scaled_image.height), 'blue')  # "L" mode is for grayscale images
       
            draw = ImageDraw.Draw(mask)
            
            # Create the polygon on the mask
            draw.polygon(self.pixels, fill=255)  # Fill the polygon with white (255)

            # Convert mask to numpy array to get the coordinates of the filled region
            mask_array = np.array(mask)

            # Get the coordinates of the pixels that are inside the polygon
            # Find all the red-colored pixels (since we filled the polygon with red)
            # Gives (y,x) coordinates
            filled_pixels = np.column_stack(np.where(np.all(mask_array == [255, 0, 0], axis=-1)))

            if len(filled_pixels) > 0:
                # Save the filled pixels as annotation in the CSV
                with open(self.save_path, "w", newline="") as file:
                    writer = csv.writer(file)
                    writer.writerow(["x", "y", "x_orig", "y_orig"])

                    for y_resized, x_resized in filled_pixels:
                        # Convert to original coordinates (rescaled)
                        x_cropped = x_resized / self.scale_factor
                        y_cropped = y_resized / self.scale_factor
                        x_orig_csv = x_cropped + self.x_orig
                        y_orig_csv = y_cropped + self.y_orig
                        writer.writerow([x_cropped, y_cropped, x_orig_csv, y_orig_csv])

                # Get the filepaths
                mask_path = self.save_path.replace(".csv", "_mask.png")
                frame_path = self.save_path.replace(".csv", ".png")

                # Save the mask and frame
                self.create_mask_from_csv(self.save_path,mask_path,self.original_frame.shape[:2])
                cv2.imwrite(frame_path, self.original_frame)

                print(f"Annotation (1 csv, 2 png) saved to {self.save_path}")
            else:
                print("No filled pixels found inside the polygon.")

        # If no annotation was made and save button was clicked, save black mask
        else:
            # Create a black mask
            mask =  np.zeros_like(self.original_frame)

            # Save the mask and frame
            mask_path = self.save_path.replace(".csv", "_mask.png")
            frame_path = self.save_path.replace(".csv", ".png")

            cv2.imwrite(mask_path, mask)
            cv2.imwrite(frame_path, self.original_frame)
            print(f"Annotation (invisible target) saved to {self.save_path}")


    def destroy(self):
        """Safely close the GUI."""
        self.running = False
        self.root.destroy()

    def run(self):
        self.root.mainloop()

    def load_and_map_annotation(self):

        if os.path.exists(self.save_path):

            try:
                self.canvas.delete("all")
                self.canvas.create_image(0, 0, image=self.tk_image, anchor=tk.NW)

                # Read the CSV file
                with open(self.save_path, 'r') as file:
                    reader = csv.reader(file)
                    header = next(reader)  # Skip the header
                    coordinates = [
                        (int(float(x_orig_csv)), int(float(y_orig_csv))) 
                        for _, _, x_orig_csv, y_orig_csv in reader
                    ]

                    coordinates_cropped_scaled = [
                        (int(float(x_cropped) * self.scale_factor), int(float(y_cropped)* self.scale_factor)) 
                        for x_cropped, y_cropped, _, _ in reader
                    ]
                    
                # Scale the original coordinates based on the resize factor and origin offset
                scaled_coordinates = [
                    (
                        int((x_orig_csv - self.x_orig) * self.scale_factor),
                        int((y_orig_csv - self.y_orig) * self.scale_factor)
                    )
                    for x_orig_csv, y_orig_csv in coordinates
                ]

                # Draw the polygon on the canvas NEED TO CHANGE TO DRAW ONLY
                self.canvas.create_polygon(
                    scaled_coordinates, outline="blue", fill="", width=2
                )

            except Exception as e:
                print(f"Error loading polygon from CSV: {e}")
        else:
            print("Annotation csv missing. Save the annotation first.")

    def create_mask_from_csv(self,csv_file, output_mask_path, frame_size):
        # Read the CSV file
        df = pd.read_csv(csv_file)
        
        # Extract the x_orig and y_orig columns
        coordinates = df[['x_orig', 'y_orig']].values.astype(int)
        
        # Create a blank mask with the same size as the frame
        mask = np.zeros(frame_size, dtype=np.uint8)
        
        
        # Fill the area defined by the coordinates (polygon)
        cv2.fillPoly(mask, [coordinates], color=255)  # 255 for white in grayscale
        
        # Save the mask
        cv2.imwrite(output_mask_path, mask)


def process_mri(frames_filepaths, total_frames_to_annotate=total_frames_to_annotate):
    
    frame_num = len(frames_filepaths)
    frame_idx = 0
    sampling_rate = round(frame_num / total_frames_to_annotate)
    print(f"Skipping every {sampling_rate} frames to get a total of {total_frames_to_annotate} frames per patient.")


    while frame_idx < frame_num:
        frame_filepath = frames_filepaths[frame_idx]
        dicom_image = pydicom.dcmread(frame_filepath)
        image = dicom_image.pixel_array
        print(f"Frame: {frame_idx} ")

        # Launch Annotation GUI
        save_path = frame_filepath.replace(".dcm", ".csv")
        annotation_gui = AnnotationGUI(image, save_path, sampling_rate)
        annotation_gui.run()

        # Check if GUI requested to quit
        if not annotation_gui.running:
            cv2.destroyAllWindows()
            return "quit"
        
        if hasattr(annotation_gui, "skip_frames_factor") and annotation_gui.skip_frames_factor > 0:
            frame_idx += sampling_rate*annotation_gui.skip_frames_factor
            # print(f"Skipping to frame: {frame_idx}")

        if hasattr(annotation_gui, "previous_frame_val") and annotation_gui.previous_frame_val < 0:
            frame_idx -= sampling_rate 

            if frame_idx < 0:
                frame_idx = 0

            # print(f"Previous frame: {frame_idx}")

        if hasattr(annotation_gui, "previous_frames_factor") and annotation_gui.previous_frames_factor > 0:
            frame_idx -= sampling_rate*annotation_gui.previous_frames_factor

            if frame_idx < 0:
                frame_idx = 0

        if hasattr(annotation_gui, "start_over_val") and annotation_gui.start_over_val:
            frame_idx = 0
            # print(f"Redo the annotation starting from frame: {frame_idx}")

        if hasattr(annotation_gui, "next_video") and annotation_gui.next_video:
            cv2.destroyAllWindows()
            return "next_video"

        if hasattr(annotation_gui, "previous_video") and annotation_gui.previous_video:
            cv2.destroyAllWindows()
            return "previous_video"

        if hasattr(annotation_gui, "next_frame_val") and annotation_gui.next_frame_val > 0:
                
            # Skip X frames between every next frame
            frame_idx += sampling_rate

    cv2.destroyAllWindows()
    return "next_video"


def main():
    xlsx_filepaths = pd.read_excel(patient_data_excel_path, sheet_name=patient_filepaths)
    patient_ids = xlsx_filepaths['patient_id'].unique()

    current_patient_idx = 0

    while 0 <= current_patient_idx < len(patient_ids):

        patient_id = patient_ids[current_patient_idx]
        frames_filepaths = xlsx_filepaths.loc[xlsx_filepaths['patient_id']==patient_id, 'dcm_image_filepath'].tolist()

        print(f"Processing MRI of patient: {patient_id}")

        frames_filepaths = [os.path.join(user_handle, i) for i in sorted(frames_filepaths)]

        # print(patient_id, frames_filepaths)
        for frame_filepath in frames_filepaths:
            if not os.path.exists(frame_filepath):
                print(f"The MRI image does not exist: {frame_filepath}")

        action = process_mri(frames_filepaths)

        if action == "next_video":
            current_patient_idx += 1
        elif action == "previous_video":
            current_patient_idx -= 1
        elif action == "quit":
            print("Exiting video processing.")
            break


# Run the gui
if __name__ == "__main__":
    main()