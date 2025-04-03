from solution_1462 import Dataset, generate, enhance
from solution_1462.eval import show_case, evaluate
from solution_1462.core import segment
import os

# Set dataset path
dataset_path = "/Users/shimincheng/Documents/trainset"

# 1. Verify dataset structure
print("Checking dataset structure...")
cases_dir = os.path.join(dataset_path, "Allcases")
labels_dir = os.path.join(dataset_path, "AllLabels")

print(f"Number of files in cases directory: {len(os.listdir(cases_dir))}")
print(f"Number of files in labels directory: {len(os.listdir(labels_dir))}")

# 2. Create dataset object
dataset = Dataset(
    root_dir=dataset_path,
    labelled=True,  # Set to True since you have labels folder
    cache=True      # Use cache for faster processing
)

# 3. Test single image processing
print("\nProcessing first case...")
case_id, image, label = dataset[0]  # Get first image and its label
cleaned, mask = segment(image)      # Perform segmentation

# Display processing results
print(f"Showing results for case {case_id}...")
show_case(
    image=image,
    cleaned=cleaned,
    mask=mask,
    label=label,
    case_id=case_id
)

# 4. Evaluate entire dataset
print("\nStarting dataset evaluation...")
results = evaluate(
    dataset=dataset,
    output_csv="evaluation_results.csv",  # Save evaluation results
    show_cases=3  # Show visualization for 3 cases
)

# 5. Generate segmentation results for all images
print("\nGenerating segmentation results for all images...")
generate(
    dataset=dataset,
    output_dir="segmentation_results",  # Results will be saved in this directory
    label_suffix="_segmented"           # Suffix for segmentation result files
)

print("\nProcessing completed!")
print("- Evaluation results saved to evaluation_results.csv")
print("- Segmentation results saved to segmentation_results directory")