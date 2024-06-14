import numpy as np
from PIL import Image
from scipy.ndimage import label, find_objects


def intersection_over_union(map1, map2):
    # annotation: ground truth label
    # intermap: interpretability map
    union = map1 + map2
    union[union>1] = 1
    iou = np.sum(map1 * map2) / np.sum(union)
    return iou


def pixel_wise_pointing_game(intermap, annotation):
    # Finding the maximum value in the interpretability map
    max_value = np.max(intermap)

    # Find all positions of the maximum value
    max_positions = np.argwhere(intermap == max_value)

    # Check if any of the maximum value positions are within the annotated region
    for position in max_positions:
        if annotation[position[0], position[1]] == 1:
            return 1  # The pixel is within the annotated mask

    return 0  # No maximum value pixels within the annotated mask


def binarize_inter_map(inter_map, top_perc):

    # Binarize intermap -->  k% high value pixels: 1 else 0
    inter_map_reshaped = sorted(inter_map.reshape(224*224))
    k = int(224*224*(top_perc/100)) 
    threshold = inter_map_reshaped[::-1][k]
    inter_map[inter_map >= threshold] = 1
    inter_map[inter_map < threshold] = 0   
    return inter_map


def mask_to_npy(image_path):
    # Load the image using Pillow
    image = Image.open(image_path)

    # Convert the image to grayscale (though it might already be black and white)
    image = image.convert('L')

    # Convert the image to a numpy array
    array = np.array(image)

    # Convert non-zero (white) pixels to 1 and zero (black) pixels to 0
    binary_array = (array > 0).astype(int)
    return binary_array



def binary_map_to_rect_mask(binary_map, save_name=None):
    # Generate labels for each segment of 1s
    labeled_array, num_features = label(binary_map)

    if num_features == 0:
        return np.zeros_like(binary_map, dtype=int)  # Return empty mask if no features
    
    # Find the bounding boxes for all objects
    objects = find_objects(labeled_array)
    
    # Initialize min and max coordinates with large and small values respectively
    # This values will be used in case of no single segment is detected
    min_row, max_row = binary_map.shape[0], 0
    min_col, max_col = binary_map.shape[1], 0
    
    # Calculate the encompassing bounding box
    for obj in objects:
        if obj is not None:
            row_slice, col_slice = obj
            min_row = min(min_row, row_slice.start)
            max_row = max(max_row, row_slice.stop)
            min_col = min(min_col, col_slice.start)
            max_col = max(max_col, col_slice.stop)
    
    # Create a mask with the same shape as binary_map
    mask = np.zeros_like(binary_map, dtype=int)
    
    # Fill the mask within the calculated bounding box
    mask[min_row:max_row, min_col:max_col] = 1

    if save_name:
        # Convert the mask to an image
        mask_image = Image.fromarray((mask * 255).astype(np.uint8))

        # Save the image
        mask_image.save(save_name)

    return mask