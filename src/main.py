import os
import torch
import numpy as np
import pandas as pd


from PIL import Image
from itertools import product

from func_inter import get_test_augmentation, load_model, get_inter_map
from func_eval import intersection_over_union, pixel_wise_pointing_game, mask_to_npy, binarize_inter_map, binary_map_to_rect_mask


def generate_interpretability(dataset_type, init_type, method, device):
    # Load pre-trained ViT model
    model = load_model(dataset_type, init_type, method)
    model.eval()
    model.to(device)

    dataset_src = f"../data/{dataset_type}"
    img_list = sorted(os.listdir(dataset_src))

    output_dir = os.path.join('./output', method)
    os.makedirs(output_dir, exist_ok=True)

    inter_map_list = list()
    for img_name in img_list:
        img_path = os.path.join(dataset_src, img_name)
        # Load image using PIL
        img = Image.open(img_path)
        img = img.convert("RGB")

        # Apply augmentation and add batch dimension
        aug = get_test_augmentation(dataset_type)
        img = aug(img)  # [C, H, W]
        img = img.to(device)

        # Generate interpretability map
        inter_map = get_inter_map(model, img, method, device)
        inter_map_list.append(inter_map)

        # Save interpretability map
        save_path = os.path.join(output_dir, f"{dataset_type}_{init_type}_{img_name.split('.')[0]}.npy")
        np.save(save_path, inter_map)  # Ensure inter_map is on CPU before saving
        print(f'Saving f{save_path}')


# I don't like current structure: will be revised.
def generate_evaluation(dataset_type, evaluation_type, top_perc=None):
    annot_mask_src = f"../data/224_mask_annot_from_dr/{dataset_type}"

    methods = ["gradcam", "attention", "chefer"]
    init_types = ["scratch", "sup", "dino", "mae"]

    annot_mask_list = os.listdir(annot_mask_src)
    eval_dict_method = dict()
    for method in methods:
        method_map_src = f"./output/{method}"
        eval_dict_init = dict()
        for init_type in init_types:
            eval_lst = list()
            for i in annot_mask_list:
                img_name = i.rstrip(".png")
                annot_path = os.path.join(annot_mask_src, i)

                annot_map = mask_to_npy(annot_path)
                inter_map = np.load(os.path.join(method_map_src, f"{dataset_type}_{init_type}_{img_name}.npy"))

                if evaluation_type == 'iou':
                    bin_map = binarize_inter_map(inter_map, top_perc)
                    rect_map = binary_map_to_rect_mask(bin_map)
                    eval_value = intersection_over_union(annot_map, rect_map)
                elif evaluation_type == 'pg':
                    eval_value = pixel_wise_pointing_game(inter_map, annot_map)
                
                eval_lst.append(eval_value)
            
            if evaluation_type == 'iou':
                eval_mean = np.mean(eval_lst)
            elif evaluation_type == 'pg':
                eval_mean = np.sum(eval_lst) / len(annot_mask_list)
                
            eval_dict_init[init_type] = eval_mean
        eval_dict_method[method] = eval_dict_init

    # Convert the dictionary to a DataFrame
    eval_df = pd.DataFrame(eval_dict_method).T  # Transpose for better format

    # Save the DataFrame to a CSV file
    os.makedirs("./output/evaluation/", exist_ok=True)
    if evaluation_type == 'iou':
        csv_filename = f'./output/evaluation/iou{top_perc}_mean_{dataset_type}.csv'
    elif evaluation_type == 'pg':
        csv_filename = f'./output/evaluation/pg_{dataset_type}.csv'
    
    eval_df.to_csv(csv_filename, index=True)      
    print(f"Results saved to {csv_filename}")


if __name__ == "__main__":
    # Parameters
    dataset_types = ["duke", "mura", "kvasir", "cpchild"]
    init_types = ["scratch", "sup", "dino", "mae"]
    methods = ["gradcam", "attention", "chefer"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Generate interpretability maps for the dataset and save them
    for dataset_type in dataset_types:
        for init_type in init_types:
            for method in methods:
                generate_interpretability(dataset_type, init_type, method, device)

    # Placeholder for visualization (commented out as no implementation yet)

    # Evaluate interpretability maps. In this case, only dataset type is used as a parameter
    # As the evaluation results will be generated per dataset.
    for dataset_type in dataset_types:
        generate_evaluation(dataset_type, 'iou', 5)
        generate_evaluation(dataset_type, 'iou', 10)
        generate_evaluation(dataset_type, 'iou', 20)
        generate_evaluation(dataset_type, 'iou', 40)
        generate_evaluation(dataset_type, 'pg')

