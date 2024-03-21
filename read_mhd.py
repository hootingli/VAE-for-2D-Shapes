import os
import numpy as np
import SimpleITK as sitk
import matplotlib.pyplot as plt

# Set the path to the directory containing segmentation files
segmentation_dir = './Dataset/TrainingDataPart3/'

# Create the output directory if it doesn't exist
output_dir = './Dataset/TrainingDataPart3/Preprocessed3'
os.makedirs(output_dir, exist_ok=True)

# Loop through the segmentation files
for i in range(38,50):  # Assuming you have segmentation files from Case00 to Case25
    # Load the segmentation image
    file_name = f'Case{i:02d}_segmentation.mhd'
    itk_image = sitk.ReadImage(os.path.join(segmentation_dir, file_name))
    image_array = sitk.GetArrayViewFromImage(itk_image)
    length = image_array.shape[0]

    # Loop through the slices in each segmentation
    for j in range(length):
        # Check if all entries in the image are zero
        if np.all(image_array[j, :, :] == 0):
            print(f'Case{i:02d}_{j:02d} is empty. Skipping.')
            continue

        # Reverse the colormap so that 0 displays white and 1 displays black
        plt.imshow(image_array[j, :, :], cmap='gray_r')

        # Remove axis labels and ticks
        plt.axis('off')

        # Save the image with the specified naming format
        output_path = os.path.join(output_dir, f'Case{i:02d}_{j:02d}.png')
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f'Saved Case{i:02d}_{j:02d}.png')

        # Close the plot
        plt.close()

print(f'Saved images to {output_dir}')
