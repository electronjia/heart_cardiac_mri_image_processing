import pandas as pd
from config import *
from PIL import Image
import numpy as np

edge_results_identifier = "_mask_edge_detection"

mask_filepaths_df = pd.read_excel(patient_data_excel_path, sheet_name="mask_filepaths")

def compute_dice_score(pred, target, epsilon=1e-6):
    intersection = (pred * target).sum()
    union = pred.sum() + target.sum()
    return (2. * intersection + epsilon) / (union + epsilon)


def compute_iou(pred, target, epsilon=1e-6):
    """Computes the Intersection over Union (IoU) score."""
    intersection = (pred * target).sum()
    union = (pred + target).sum() - intersection  # Union = Total - Intersection
    return (intersection + epsilon) / (union + epsilon)


new_dataframe_results = pd.DataFrame(columns=["patient_id", "dcm_image_filepath", "image_filepath", "mask_filepath",	"csv_filepath", mahri_results_column,  "dice", "iou"])

for idx, row in mask_filepaths_df.iterrows():

    patient_id = row["patient_id"]
    dcm_img_filepath = row['dcm_image_filepath']
    img_filepath = row['image_filepath']
    mask_filepath = row['mask_filepath']
    csv_filepath = row['csv_filepath']

    print(f"{idx} - > processing {mask_filepath}")

    if not pd.isna(mask_filepath):

        # target mask
        target_filepath = os.path.join(user_handle, mask_filepath)
        target_mask = np.array(Image.open(target_filepath).convert("L"), dtype=np.float32)
        target_mask[target_mask == 255] = 1.0


        # predicted mask
        prediction_filepath = target_filepath.replace("_mask.png", mahri_results_identifier)

        predicted_mask = np.array(Image.open(prediction_filepath).convert("L"), dtype=np.float32)
        predicted_mask[predicted_mask == 255] = 1.0


        new_dice = compute_dice_score(predicted_mask, target_mask)
        new_iou = compute_iou(predicted_mask, target_mask)

        new_dataframe_results.loc[len(new_dataframe_results)] =[
            patient_id, dcm_img_filepath, img_filepath, mask_filepath,	 csv_filepath, os.path.relpath(prediction_filepath, user_handle),
            new_dice, new_iou]

    else:
        new_dataframe_results.loc[len(new_dataframe_results)] = [ patient_id, 

      dcm_img_filepath, img_filepath, mask_filepath, csv_filepath, dcm_img_filepath.replace(".dcm", mahri_results_identifier), None, None
        ]


new_dataframe_results.to_excel(mahri_results_df_filepath, index=False)





