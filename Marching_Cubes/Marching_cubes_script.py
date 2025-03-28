import pydicom
import numpy as np
import os
import cv2
import imageio
from scipy.ndimage import distance_transform_edt, gaussian_filter
import matplotlib.pyplot as plt
from skimage.filters import threshold_otsu, sobel
from skimage.measure import label, regionprops, marching_cubes, mesh_surface_area
from skimage.segmentation import clear_border, watershed
from skimage.morphology import remove_small_objects, binary_opening, binary_closing, disk
from mpl_toolkits.mplot3d.art3d import Poly3DCollection
from scipy.spatial import Delaunay
from skimage.io import imsave
import pandas as pd
from config import *

def marching_cubes_implementation(folder_path):
    def load_dicom_series(folder_path):
        """
        Load a DICOM image series from a folder and return the 3D volume and voxel spacing.

        Parameters:
        - folder_path (str): Path to the folder containing DICOM files.

        Returns:
        - volume (numpy array): 3D NumPy array of image data.
        - spacing (tuple): (slice thickness, pixel spacing in x, pixel spacing in y).
        """
        dicom_files = []

        # Load DICOM files from folder
        for filename in sorted(os.listdir(folder_path)):
            if filename.endswith('.dcm'):
                filepath = os.path.join(folder_path, filename)
                try:
                    dicom_files.append(pydicom.dcmread(filepath))
                except Exception as e:
                    print(f"⚠️ Warning: Could not read {filename} - {e}")

        if not dicom_files:
            raise ValueError("❌ No valid DICOM files found in the folder.")

        # Try sorting by ImagePositionPatient if available
        try:
            dicom_files.sort(key=lambda x: float(x.ImagePositionPatient[2]))
        except AttributeError:
            print("⚠️ ImagePositionPatient not found. Using default order.")

        # Stack images into a 3D NumPy array
        volume = np.stack([file.pixel_array for file in dicom_files])

        # Extract voxel spacing
        try:
            slice_thickness = float(dicom_files[0].SliceThickness)
        except AttributeError:
            slice_thickness = 1.0  # Default value if missing
            print("⚠️ SliceThickness not found. Using default value of 1.0.")

        try:
            pixel_spacing = dicom_files[0].PixelSpacing
            spacing = (slice_thickness, float(pixel_spacing[0]), float(pixel_spacing[1]))
        except AttributeError:
            spacing = (slice_thickness, 1.0, 1.0)  # Default pixel spacing
            print("⚠️ PixelSpacing not found. Using default (1.0, 1.0).")

        print(f"Loaded DICOM series: {len(dicom_files)} slices")
        print(f"Volume shape: {volume.shape}")
        print(f"Voxel spacing: {spacing}")

        return volume, spacing, dicom_files

    # Example usage for patient SCD0000101
    # folder_path = r"C:\Users\Kaiwen Liu\OneDrive - University of Toronto\Desktop\github_repo\heart_cardiac_mri_image_processing\data\SCD_IMAGES_01\SCD0000101\CINESAX_300" # Change depending on where you patient files are
    # folder_path = r"C:\Users\Kaiwen Liu\OneDrive - University of Toronto\Desktop\github_repo\heart_cardiac_mri_image_processing\data\SCD_IMAGES_05\SCD0004501\CINESAX_1100"
    # folder_path = r"C:\Users\Kaiwen Liu\OneDrive - University of Toronto\Desktop\github_repo\heart_cardiac_mri_image_processing\data\SCD_IMAGES_05\SCD0004201\CINESAX_302"




    def normalize_volume(volume):
        # Clip out extreme values (optional, helps with outliers)
        volume = np.clip(volume, np.percentile(volume, 1), np.percentile(volume, 99))

        # Min-max normalization to [0, 1]
        volume = volume.astype(np.float32)
        volume -= volume.min()
        volume /= volume.max()

        return volume
    
    def apply_center_weighting(volume, alpha=1):
        """
        Enhance the image intensity towards the center using a Gaussian weight map.

        Parameters:
        - volume: 3D MRI volume.
        - alpha: Weighting factor (0.0 - 1.0).
        """
        z, y, x = volume.shape

        # Create a Gaussian weight map centered in the middle
        yy, xx = np.meshgrid(np.linspace(-1, 1, x), np.linspace(-1, 1, y))
        distance = np.sqrt(xx**2 + yy**2)
        weight_map = np.exp(-4 * distance**2)  # Gaussian-like falloff

        weighted_volume = np.zeros_like(volume)

        # Apply the weight map to each slice
        for i in range(z):
            weighted_volume[i] = volume[i] * (1 + alpha * weight_map)

        return weighted_volume
    
    def crop_center(volume, crop_size):
        """
        Crop the center region of a 3D volume.

        Parameters:
        - volume: 3D NumPy array (z, y, x).
        - crop_size: Tuple (crop_height, crop_width).

        Returns:
        - Cropped volume.
        """
        z, y, x = volume.shape
        crop_height, crop_width = crop_size

        # Calculate center coordinates
        center_y, center_x = y // 2, x // 2

        # Define cropping boundaries
        y_min = max(center_y - crop_height // 2, 0)
        y_max = min(center_y + crop_height // 2, y)
        x_min = max(center_x - crop_width // 2, 0)
        x_max = min(center_x + crop_width // 2, x)

        # Crop the volume
        cropped_volume = volume[:, y_min:y_max, x_min:x_max]

        return cropped_volume
    
    def remove_border_components(binary_slice):
        """
        Remove connected components touching the borders of the image.

        Parameters:
        - binary_slice (numpy array): A 2D binary image.

        Returns:
        - cleaned_slice (numpy array): Binary image with border-touching components removed.
        """
        # Label connected components in the binary mask
        labeled_slice = label(binary_slice)

        # Remove components touching the image borders
        cleaned_slice = clear_border(labeled_slice)

        # Convert back to binary
        return cleaned_slice > 0

    def segment_heart_otsu(volume):
        """
        Segment the heart using Otsu's thresholding and remove border-touching components.

        Steps:
        1. Apply Otsu's thresholding to segment potential heart regions.
        2. Remove any connected components touching the image borders.
        3. Return the cleaned segmented volume.

        Parameters:
        - volume (numpy array): 3D NumPy array containing the image stack.

        Returns:
        - segmented_volume (numpy array): Binary volume with only the heart region.
        """

        segmented_volume = np.zeros_like(volume, dtype=bool)  # Initialize output

        for i in range(volume.shape[0]):  # Iterate over slices
            slice_img = volume[i]

            # Step 1: Apply Otsu’s thresholding
            thresh = threshold_otsu(slice_img)
            binary_slice = slice_img > thresh  # Convert to binary mask

            # Step 2: Remove border-touching components
            cleaned_slice = remove_border_components(binary_slice)

            # Store in output volume
            segmented_volume[i] = cleaned_slice

        return segmented_volume.astype(np.uint8)  # Convert to uint8 for visualization
    
    def filter_circular_regions(segmented_volume, circularity_thresh=0.1, min_size=30):
        """
        Keep only circular regions in a segmented 3D volume.

        Parameters:
        - segmented_volume: 3D binary NumPy array (segmentation mask).
        - circularity_thresh: Circularity threshold (higher = more circular).
        - min_size: Minimum region size to keep (removes small noise).

        Returns:
        - Filtered 3D binary mask with mostly circular regions.
        """
        filtered_volume = np.zeros_like(segmented_volume)

        for z in range(segmented_volume.shape[0]):
            slice_mask = segmented_volume[z]

            # Label connected components
            labeled_mask = label(slice_mask)

            for region in regionprops(labeled_mask):
                # Calculate circularity: (4 * pi * area) / perimeter^2
                if region.perimeter > 0:  # Avoid division by zero
                    circularity = (4 * np.pi * region.area) / (region.perimeter ** 2)

                    # Keep region if circular enough and above min size
                    if circularity >= circularity_thresh and region.area >= min_size:
                        filtered_volume[z][labeled_mask == region.label] = 1

        return filtered_volume
    
    def improved_watershed(segmented_volume, blur_sigma=2, distance_exp=0.7, seed_quantile=0.65):
        """
        Applies an improved Watershed segmentation across all slices in a 3D volume.
        - Uses Gaussian-blurred Sobel for smoother edge detection.
        - Enhances distance map contrast for better LV separation.
        - Adapts seed selection to account for intensity variations.

        Parameters:
        - segmented_volume (ndarray): Binary 3D array from Otsu thresholding.
        - blur_sigma (float): Standard deviation for Gaussian blur (default: 1.5).
        - distance_exp (float): Exponent for distance transform enhancement (default: 1.2).
        - seed_quantile (float): Quantile threshold for selecting seeds (default: 0.5).

        Returns:
        - watershed_labels (ndarray): 3D array with labeled regions after Watershed segmentation.
        """

        # Initialize output volume
        watershed_labels = np.zeros_like(segmented_volume, dtype=np.int32)

        for i in range(segmented_volume.shape[0]):  # Process each slice
            binary_slice = segmented_volume[i]

            if np.sum(binary_slice) == 0:
                continue  # Skip empty slices

            # Step 1: Apply Gaussian blur to smooth noise before Sobel
            smoothed_slice = gaussian_filter(binary_slice.astype(float), sigma=blur_sigma)

            # Step 2: Compute edges using Sobel on the smoothed image
            edges = sobel(smoothed_slice)

            # Step 3: Compute the distance transform (inner areas have higher values)
            distance_map = distance_transform_edt(binary_slice)

            # Step 4: Enhance distance contrast for better watershed performance
            distance_map = distance_map ** distance_exp

            # Step 5: Adapt seed threshold based on the intensity distribution
            seed_threshold = np.quantile(distance_map[distance_map > 0], seed_quantile)
            seeds = distance_map > seed_threshold

            # Step 6: Label connected components in the seed map
            markers = label(seeds)

            # Step 7: Apply Watershed with edges as the barrier
            ws_result = watershed(edges, markers, mask=binary_slice)

            # Store result
            watershed_labels[i] = ws_result

        return watershed_labels
    
    def LV_isolation(watershed_labels, min_size=50, max_size=4000, 
                 min_size_reference=400, slice_range=(41, 100), 
                 sample_slices=4):
        """
        Identifies the Left Ventricle (LV) in a reference slice and extends segmentation
        by selecting the region in each slice that has both:
        1. The highest proportion of pixel overlap with the reference LV region.
        2. The closest centroid distance to the reference LV region.
        3. For slices below the reference, regions 1.5x or larger than the reference LV are ignored.

        Parameters:
        - watershed_labels (ndarray): 3D labeled array from segmentation.
        - min_size (int): Global minimum allowable region size.
        - max_size (int): Global maximum allowable region size.
        - min_size_reference (int): Minimum size for selecting the reference LV region.
        - slice_range (tuple): Range of slices to search for the reference LV.
        - sample_slices (int): Number of slices to sample within slice_range.
        - visualize (bool): If True, displays the chosen LV region.

        Returns:
        - lv_volume (ndarray): 3D binary array containing only the LV segmentation.
        """

        lv_volume = np.zeros_like(watershed_labels, dtype=np.uint8)
        num_slices = watershed_labels.shape[0]

        # Step 1: Identify LV in a well-segmented reference slice
        lower_half_slices = np.linspace(slice_range[0], slice_range[1], sample_slices, dtype=int)
        best_lv_centroid = None
        best_lv_label = None
        chosen_slice = None
        reference_lv_size = None  # Store reference LV size

        for i in lower_half_slices:
            slice_labels = label(watershed_labels[i])
            if np.max(slice_labels) == 0:
                continue  # Skip empty slices

            # Apply reference slice threshold
            regions = [r for r in regionprops(slice_labels) if min_size_reference <= r.area <= max_size]

            if not regions:
                continue

            # Find the most circular LV region
            most_circular_region = max(regions, key=lambda r: (4 * np.pi * r.area) / (r.perimeter ** 2 + 1e-5))  
            best_lv_centroid = most_circular_region.centroid
            reference_lv_size = most_circular_region.area  # Store reference LV size
            chosen_slice = i
            break  # Take the first well-defined LV region

        if best_lv_centroid is None:
            # raise ValueError("Could not identify a well-segmented LV region in slices 41-100.")
            # If no valid region is found, default to the center slice
            chosen_slice = watershed_labels.shape[0] // 2  # Get the middle slice index
            reference_lv_size = None  # No valid size measurement

        # Step 2: Find the label containing this LV centroid
        slice_labels = label(watershed_labels[chosen_slice])
        for region in regionprops(slice_labels):
            if min_size <= region.area <= max_size:  # Global threshold applied here
                if region.bbox[0] <= best_lv_centroid[0] <= region.bbox[2] and \
                region.bbox[1] <= best_lv_centroid[1] <= region.bbox[3]:
                    best_lv_label = region.label
                    break  

        if best_lv_label is None:
            raise ValueError("Could not map the chosen LV centroid to a label in the reference slice.")


        # Assign LV label in the best slice
        lv_volume[chosen_slice] = (slice_labels == best_lv_label)  

        # Step 3: Use the reference LV as a mask for all slices
        reference_mask = lv_volume[chosen_slice]  # Fixed reference LV mask
        reference_centroid = best_lv_centroid  # Fixed reference LV centroid

        for i in range(num_slices):
            if i == chosen_slice:
                continue  

            slice_labels = label(watershed_labels[i])
            if np.max(slice_labels) == 0:
                continue  

            # Apply global min/max size threshold
            regions = [r for r in regionprops(slice_labels) if min_size <= r.area <= max_size]  
            best_region = None
            best_score = 0  # Higher is better (overlap & proximity combined)

            # Step 1: Find the best region based on overlap ratio & centroid distance
            for region in regions:
                # Apply additional constraint only for slices **below** the reference (i.e., slices 1–41)
                if i < chosen_slice and region.area >= 1.5 * reference_lv_size:
                    continue  # Skip regions that are 1.5x larger than the reference LV

                region_mask = (slice_labels == region.label)
                overlap = np.sum(reference_mask & region_mask)  # Count overlapping pixels
                overlap_ratio = overlap / (region.area + 1e-5)  # Proportion of region that overlaps

                # Compute centroid distance
                centroid_x, centroid_y = region.centroid
                distance = np.sqrt((centroid_x - reference_centroid[0]) ** 2 + (centroid_y - reference_centroid[1]) ** 2)

                # Define the score: prioritize overlap but prefer closer centroids
                score = overlap_ratio - (0.001 * distance)  # Small weight for distance

                if score > best_score:
                    best_score = score
                    best_region = region

            # Step 2: Assign the best region found
            if best_region:
                lv_volume[i] = (slice_labels == best_region.label)

        return lv_volume

    def clean_segmentation(volume, radius=1):
        struct_elem = disk(radius)  # 2D circular element

        cleaned_volume = np.zeros_like(volume)
        for i in range(volume.shape[0]):
            cleaned_slice = binary_closing(binary_opening(volume[i], struct_elem), struct_elem)
            cleaned_volume[i] = cleaned_slice

        return cleaned_volume

    def extract_surface_marching_cubes(original_volume, segmented_volume, spacing):
        """
        Extracts a 3D surface mesh from the original DICOM volume using Marching Cubes.
        The segmentation mask is used as a threshold to isolate the left ventricle region.

        Parameters:
        - original_volume (ndarray): The 3D DICOM volume (slices, height, width).
        - segmented_volume (ndarray): The 3D binary segmentation mask (LV = 1, Background = 0).
        - spacing (tuple): (dz, dy, dx) voxel spacing for accurate scaling.

        Returns:
        - vertices (ndarray): Mesh vertices.
        - faces (ndarray): Mesh faces (triangles).
        """
        if original_volume.ndim != 3 or segmented_volume.ndim != 3:
            raise ValueError(f"Expected 3D input, but got shape {original_volume.shape}")

        thresholded_volume = np.where(segmented_volume > 0, original_volume, 0)

        valid_pixels = thresholded_volume[thresholded_volume > 0]
        if len(valid_pixels) == 0:
            print("Warning: No valid pixels in thresholded volume. Skipping.")
            return None, None

        level = np.percentile(valid_pixels, 50)

        vertices, faces, _, _ = marching_cubes(thresholded_volume, level=level, spacing=spacing)
        return vertices, faces

    def compute_lv_metrics(vertices, faces, spacing):
        """
        Computes Left Ventricle volume and surface area from Marching Cubes output.
        
        Parameters:
        - vertices (ndarray): Extracted 3D points from marching cubes.
        - faces (ndarray): Triangular faces from marching cubes.
        - spacing (tuple): Voxel spacing (dz, dy, dx) in mm.

        Returns:
        - volume (float): LV volume in milliliters (mL).
        - surface_area (float): LV surface area in cm².
        """

        # Convert voxel spacing to cubic mm (scaling factor)
        voxel_volume = spacing[0] * spacing[1] * spacing[2]  # mm³

        # Compute LV Volume using Delaunay triangulation
        tri = Delaunay(vertices)
        volume = np.sum(np.abs(np.linalg.det(vertices[tri.simplices[:, :3]]))) / 6.0  # Volume in mm³
        volume *= 1e-3  # Convert mm³ to mL

        # Compute LV Surface Area (using skimage function)
        surface_area = mesh_surface_area(vertices, faces) * 1e-2  # Convert mm² to cm²

        return volume, surface_area

    def plot_3d_mesh(vertices, faces, save_path=None, patient_name = None):
        """
        Plots the extracted 3D mesh using Matplotlib.
        """
        if vertices is None or faces is None:
            return

        fig = plt.figure(figsize=(8, 8))
        ax = fig.add_subplot(111, projection="3d")

        mesh = Poly3DCollection(vertices[faces], alpha=0.6)
        mesh.set_edgecolor("k")
        ax.add_collection3d(mesh)

        ax.view_init(elev=100, azim=180) 

        ax.set_xlim(vertices[:, 0].min(), vertices[:, 0].max())
        ax.set_ylim(vertices[:, 1].min(), vertices[:, 1].max())
        ax.set_zlim(vertices[:, 2].min(), vertices[:, 2].max())

        ax.set_xlabel("X-axis")
        ax.set_ylabel("Y-axis")
        ax.set_zlabel("Z-axis")
        ax.set_title(f"Patient {patient_name} Mesh of Left Ventricle")

        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()

    def plot_lv_metrics(lv_volumes, lv_surface_areas, num_frames=20):
        """
        Plots LV Volume (mL) and LV Surface Area (cm²) over the cardiac cycle.

        Parameters:
        - lv_volumes (list): List of LV volumes over time.
        - lv_surface_areas (list): List of LV surface areas over time.
        - num_frames (int): Number of frames in the cardiac cycle (default: 20).
        """
        time_points = np.linspace(0, 100, num_frames)  # Normalize time (0-100% cardiac cycle)

        plt.figure(figsize=(10, 5))

        # Plot LV Volume
        plt.subplot(1, 2, 1)
        plt.plot(time_points, lv_volumes, marker="o", linestyle="-", color="b", label="LV Volume")
        plt.xlabel("Cardiac Cycle (%)")
        plt.ylabel("LV Volume (mL)")
        plt.title("LV Volume over Cardiac Cycle")
        plt.legend()
        plt.grid(True)

        # Plot LV Surface Area
        plt.subplot(1, 2, 2)
        plt.plot(time_points, lv_surface_areas, marker="s", linestyle="-", color="r", label="LV Surface Area")
        plt.xlabel("Cardiac Cycle (%)")
        plt.ylabel("LV Surface Area (cm²)")
        plt.title("LV Surface Area over Cardiac Cycle")
        plt.legend()
        plt.grid(True)

        plt.tight_layout()
        plt.show()

    def create_heartbeat_gif(original_volume, segmented_volume, folder_path, frames_per_slice=20, spacing =(10.0, 1.367188, 1.367188)):
        """
        Generates a GIF of the beating heart by iterating through the cardiac cycle.
        
        Parameters:
        - original_volume (ndarray): The 3D DICOM volume (frames, height, width).
        - segmented_volume (ndarray): The 3D segmentation mask (frames, height, width).
        - folder_path (str): Base folder where results should be saved.
        - frames_per_slice (int): Number of frames per depth slice.
        """
        # Extract folder name and create new results directory
        base_folder_name = os.path.basename(os.path.normpath(folder_path))
        results_folder = os.path.join(folder_path, f"{base_folder_name}_marching_cubes_results")
        os.makedirs(results_folder, exist_ok=True)

        frames = []
        temp_frame_paths = []
        lv_volumes = []
        lv_surface_areas = []

        num_frames = frames_per_slice  # Since we have 20 unique time frames
        slices_per_frame = original_volume.shape[0] // num_frames  # Number of spatial slices

        # Loop through each time frame 
        for t in range(num_frames):
            print(f"Processing frame {t + 1}/{num_frames}")

            # Select every 20th slice for the current time frame
            frame_original = original_volume[t::num_frames]  # Shape: (slices_per_frame, height, width)
            frame_segmented = segmented_volume[t::num_frames]  # Shape: (slices_per_frame, height, width)

            vertices, faces = extract_surface_marching_cubes(frame_original, frame_segmented, spacing)

            if vertices is None or faces is None:
                continue  # Skip frames with no valid segmentation

            # Compute LV volume and surface area
            volume, surface_area = compute_lv_metrics(vertices, faces, spacing)
            lv_volumes.append(volume)
            lv_surface_areas.append(surface_area)

            frame_path = os.path.join(results_folder, f"{base_folder_name}_frame_{t+1}.png")
            plot_3d_mesh(vertices, faces, save_path=frame_path, patient_name=base_folder_name)
            temp_frame_paths.append(frame_path)

        # Load images and save GIF
        for frame_path in temp_frame_paths:
            frames.append(imageio.v2.imread(frame_path))

        save_gif_path = os.path.join(results_folder, f"{base_folder_name}_heartbeat.gif")
        imageio.mimsave(save_gif_path, frames, duration=0.1)

        print(f"Saved all frames and heartbeat animation for patient {base_folder_name}")
        volume, surface_area = compute_lv_metrics(vertices, faces, spacing)

        return lv_volumes, lv_surface_areas

    def restore_original_size(mask, original_size, crop_size):
        """
        Restores a cropped mask back to its original size by zero-padding.

        Parameters:
        - mask: Cropped binary mask (2D NumPy array).
        - original_size: Tuple (orig_height, orig_width) of the DICOM slice.
        - crop_size: Tuple (crop_height, crop_width) used during cropping.

        Returns:
        - Restored mask of original DICOM size.
        """
        orig_height, orig_width = original_size
        crop_height, crop_width = crop_size

        # Create a blank mask with original size
        restored_mask = np.zeros((orig_height, orig_width), dtype=np.uint8)

        # Compute placement for the cropped mask
        center_y, center_x = orig_height // 2, orig_width // 2
        y_min = max(center_y - crop_height // 2, 0)
        y_max = min(center_y + crop_height // 2, orig_height)
        x_min = max(center_x - crop_width // 2, 0)
        x_max = min(center_x + crop_width // 2, orig_width)

        # Place the cropped mask inside the full-size mask
        restored_mask[y_min:y_max, x_min:x_max] = mask

        return restored_mask

    def normalize_dicom_image(image):
        """
        Normalize DICOM image pixel values to 0-255 for correct visualization.

        Parameters:
        - image: 2D NumPy array of the DICOM slice.

        Returns:
        - Normalized image (dtype=np.uint8) for saving as PNG.
        """
        image = image.astype(np.float32)
        min_val, max_val = image.min(), image.max()
        
        if max_val > min_val:  # Avoid division by zero
            image = (image - min_val) / (max_val - min_val) * 255.0

        return image.astype(np.uint8)

    def save_slices_and_masks(cleaned_volume, dicom_files, folder_path):
        """
        Saves both the original DICOM slice as a PNG and the corresponding LV mask.

        Parameters:
        - cleaned_volume: 3D NumPy array of LV masks.
        - dicom_files: List of original DICOM objects.
        - folder_path: Directory where DICOM files are stored (files will be saved here).
        """
        # Extract original size from first DICOM slice
        original_size = dicom_files[0].pixel_array.shape
        crop_size = cleaned_volume[0].shape

        for i, dicom_file in enumerate(dicom_files):
            # Extract original DICOM filename (without extension)
            dicom_filename = os.path.splitext(os.path.basename(dicom_file.filename))[0]

            # Restore the mask to its original size
            restored_mask = restore_original_size(cleaned_volume[i], original_size, crop_size)

            # Normalize and save the original DICOM slice
            original_image = normalize_dicom_image(dicom_file.pixel_array)
            original_path = os.path.join(folder_path, f"{dicom_filename}.png")
            imsave(original_path, original_image)

            # Save the mask (even if empty)
            mask_path = os.path.join(folder_path, f"{dicom_filename}_mask_marching_cubes.png")
            imsave(mask_path, (restored_mask * 255).astype(np.uint8))  # Convert binary mask to 0-255

    def calculate_lv_metrics(lv_volumes, lv_surface_areas):
        """
        Calculate important LV metrics for cardiac function assessment.

        Parameters:
        - lv_volumes: List or NumPy array of left ventricle volumes across cardiac cycle (sorted by frame).
        - lv_surface_areas: List or NumPy array of left ventricle surface areas across the cycle.

        Returns:
        - A dictionary of LV metrics, ready for export to Excel.
        """

        # Identify end-diastolic (largest volume) and end-systolic (smallest volume) frames
        ed_index = np.argmax(lv_volumes)  # Frame with largest volume (EDV)
        es_index = np.argmin(lv_volumes)  # Frame with smallest volume (ESV)

        edv = lv_volumes[ed_index]/100  # End-Diastolic Volume
        esv = lv_volumes[es_index]/100  # End-Systolic Volume

        # Calculate Ejection Fraction
        ef = ((edv - esv) / edv) * 100 if edv > 0 else 0  # Avoid division by zero

        # Surface areas
        sa_ed = lv_surface_areas[ed_index]  # Surface area at ED
        sa_es = lv_surface_areas[es_index]  # Surface area at ES

        # Estimate LV diameters (approximated as width of largest cross-section)
        lv_diameter_ed = (3 * edv / (4 * np.pi))**(1/3) * 20  # Approximated from volume
        lv_diameter_es = (3 * esv / (4 * np.pi))**(1/3) * 20  


        # Store results in a dictionary
        metrics = {
            "Ejection Fraction (%)": ef,
            "End-Diastolic Volume (mL)": edv,
            "End-Systolic Volume (mL)": esv,
            "Surface Area at ED (mm²)": sa_ed,
            "Surface Area at ES (mm²)": sa_es,
            "LV Diameter at ED (mm)": lv_diameter_ed,
            "LV Diameter at ES (mm)": lv_diameter_es

        }

        return metrics
    
    volume, spacing, dicom_files = load_dicom_series(folder_path)

    normalized_volume = normalize_volume(volume)

    # Crop the center 256x256 region
    cropped_volume = crop_center(normalized_volume, crop_size=(120, 120))

    # Apply center weighting
    cropped_volume = apply_center_weighting(cropped_volume, alpha=3)

    # Apply segmentation with border removal
    segmented_volume = segment_heart_otsu(cropped_volume)

    # Apply circular filtering to segmentation
    filtered_segmentation = filter_circular_regions(segmented_volume, circularity_thresh=0.15, min_size=100)

    # Apply watershed segmetation 
    watershed_volume = improved_watershed(segmented_volume)

    # Isolate the left ventricle
    left_ventricle_isolation = LV_isolation(watershed_volume)

    # Morphological opening and closing to remove small specks and fill holes
    cleaned_volume = clean_segmentation(left_ventricle_isolation, radius=2)

    # Output gif of cardiac left ventricle cycle from marching cubes reconstruction and output metrics for LV quantification
    result = create_heartbeat_gif(cropped_volume, cleaned_volume, folder_path, spacing = spacing)
    if result is not None:
        lv_volumes, lv_surface_areas = result
    else:
        lv_volumes, lv_surface_areas = [], []
        print("⚠️ Warning: create_heartbeat_gif() returned None. Using empty lists.")

    # Save the masks as well as the original dicom files
    save_slices_and_masks(cleaned_volume, dicom_files, folder_path)

    metrics = calculate_lv_metrics(lv_volumes, lv_surface_areas)

    return metrics

def save_patient_metrics(file_path, patient_id, patient_gender, patient_age, patient_pathology, metrics):
    """
    Appends a patient's LV metrics to an Excel file safely.

    Parameters:
    - file_path: String path to the Excel file.
    - patient_id: Unique patient ID.
    - patient_gender: Patient gender.
    - patient_age: Patient age.
    - patient_pathology: Patient pathology.
    - metrics: Dictionary containing LV metrics.
    """
    
    # Ensure file_path is a valid string
    if not isinstance(file_path, str):
        raise ValueError(f"Expected file_path as a string, but got {type(file_path)}: {file_path}")
    
    # Convert patient data to a DataFrame row
    patient_data = {
        "Patient ID": patient_id,
        "Gender": patient_gender,
        "Age": patient_age,
        "Pathology": patient_pathology,
        **metrics  # Expands all LV metrics into separate columns
    }
    
    new_data = pd.DataFrame([patient_data])

    # Ensure the directory exists before saving
    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    # Check if file exists
    if os.path.exists(file_path):
        try:
            existing_data = pd.read_excel(file_path, engine="openpyxl")
            updated_data = pd.concat([existing_data, new_data], ignore_index=True)
        except Exception as e:
            raise RuntimeError(f"Error reading the existing Excel file: {e}")
    else:
        updated_data = new_data

    # Safe saving - avoid permission issues
    temp_path = file_path + "_tmp.xlsx"
    updated_data.to_excel(temp_path, index=False, engine="openpyxl")

    # Replace the old file with the new one safely
    os.replace(temp_path, file_path)
    
    print(f"✅ Patient {patient_id} data added to {file_path}")



def main():
    xlsx_filepaths = pd.read_excel(patient_data_excel_path, sheet_name=patient_data)
    patient_ids = xlsx_filepaths['patient_id'].unique()
    current_patient_idx = 0

    # patient_metrics_marching_cubes_path = r"C:\Users\Kaiwen Liu\OneDrive - University of Toronto\Desktop\github_repo\heart_cardiac_mri_image_processing\Marching_Cubes\patient_metrics_marching_cubes.xlsx"

    # save_excel_path = os.path.expanduser(save_excel_path) 

    while 0 <= current_patient_idx < len(patient_ids):

        patient_id = patient_ids[current_patient_idx]
        patient_filepaths = xlsx_filepaths.loc[xlsx_filepaths['patient_id']==patient_id, 'filepath'].tolist()

        patient_filepaths = [os.path.join(user_handle, i) for i in sorted(patient_filepaths)]

        print(f"{patient_filepaths}")
        
        patient_age = xlsx_filepaths.loc[xlsx_filepaths['patient_id']==patient_id, 'age'].tolist()
        patient_gender = xlsx_filepaths.loc[xlsx_filepaths['patient_id']==patient_id, 'gender'].tolist()
        patient_pathology = xlsx_filepaths.loc[xlsx_filepaths['patient_id']==patient_id, 'pathology'].tolist()
        # frames_filepaths = xlsx_filepaths.loc[xlsx_filepaths['patient_id']==patient_id, 'dcm_image_filepath'].tolist()

        print(f"Processing MRI of patient: {patient_id}")

        try:
            # Ensure all metric values are scalar (convert lists/arrays to single values)
            metrics = marching_cubes_implementation(patient_filepaths[0])
            save_patient_metrics(patient_metrics_marching_cubes_path, patient_id, patient_gender, patient_age, patient_pathology, metrics)
            current_patient_idx += 1
        except Exception as e:
            print(f"Skipping Patient {patient_id} due to error: {e}")
            current_patient_idx += 1
            continue  # Moves to the next patient
            
        
        # formatted_metrics = {key: (value[0] if isinstance(value, (list, tuple, np.ndarray)) else value) for key, value in metrics.items()}

# Run the gui
if __name__ == "__main__":
    main()