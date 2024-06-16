import os
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import timm
import numpy as np

from pytorch_grad_cam import GradCAM
from pytorch_grad_cam.utils.model_targets import ClassifierOutputTarget
from pytorch_grad_cam.utils.reshape_transforms import vit_reshape_transform

def get_test_augmentation(dataset) -> transforms.Compose:
    
    # ImageNet mean & std
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225])

    # Difference in resizing (resize vs center crop)
    if dataset in ['duke', 'cpchild']:
        aug = transforms.Compose([
            transforms.Resize(([224, 224])),
            transforms.ToTensor(),
            normalize
            ])        

    elif dataset in ['mura', 'kvasir']:
        aug = transforms.Compose([
            transforms.CenterCrop(([224, 224])),
            transforms.ToTensor(),
            normalize
            ])

    return aug


def load_vit(model_name):
    if model_name not in ['dino', 'mae', 'sup', 'scratch']:
        assert False, 'Wrong model name'
    current_path = os.path.dirname(os.path.abspath(__file__))
    path = current_path + '/'
    model_locs = {
        'mae': '../weights/mae_finetuned_vit_base.pth',
        'dino': '../weights/dino_vitbase16_pretrain.pth',
        'sup': 'n/a',
        'scratch': 'n/a',}

    model_path = path + model_locs[model_name]
    if model_name == 'dino':
        from models.dino_vit import dino_vit_base
        model = dino_vit_base()
        model_ckpt = torch.load(model_path)
        model.load_state_dict(model_ckpt, strict=True)
        model.head = nn.Linear(in_features=768, out_features=2, bias=True)
    elif model_name == 'mae':
        from models.mae_vit import mae_vit_base_patch16
        model = mae_vit_base_patch16()
        model_ckpt = torch.load(model_path)['model']
        model_ckpt['norm.weight'] = model_ckpt['fc_norm.weight']
        model_ckpt['norm.bias'] = model_ckpt['fc_norm.bias']
        del model_ckpt['fc_norm.weight']
        del model_ckpt['fc_norm.bias']
        model.load_state_dict(model_ckpt)
        model.head = nn.Linear(in_features=768, out_features=2, bias=True)
    elif model_name == 'sup':
        import timm
        model = timm.create_model('vit_base_patch16_224', pretrained=True)
        model.head = nn.Linear(in_features=768, out_features=2, bias=True)
    elif model_name == 'scratch':
        import timm
        model = timm.create_model('vit_base_patch16_224', pretrained=False)
        model.head = nn.Linear(in_features=768, out_features=2, bias=True)

    return model


def load_model(dataset_type, init_type, method, src = '../weights'):
    """
    Load a pretrained model from disk, corresponding to the given pretraining method and dataset name
    """
    if method == "chefer":
        from baselines.ViT.ViT_LRP import vit_base_patch16_224 as vit_LRP
        model = vit_LRP()
    elif method == "gradcam":
        model = load_vit(init_type)
    elif method == "attention":
        model = timm.create_model('vit_base_patch16_224', pretrained=False)
        model.head = nn.Linear(in_features=768, out_features=2, bias=True)       

    if dataset_type not in ['duke', 'mura', 'cpchild', 'kvasir']:
        assert False, 'Wrong dataset name.'
    
    base_path = src
    weight_name = f'{dataset_type}_{init_type}.pth'
    weight_path = os.path.join(base_path, weight_name)

    checkpoint = torch.load(weight_path, map_location = 'cpu')
    model.load_state_dict(checkpoint)
    
    return model


def get_chefer_attribution(attribution_generator, original_image, device, class_index=None):
    transformer_attribution = attribution_generator.generate_LRP(original_image.unsqueeze(0).cuda(device), method="transformer_attribution", index=class_index).detach()
    transformer_attribution = transformer_attribution.reshape(1, 1, 14, 14)
    transformer_attribution = torch.nn.functional.interpolate(transformer_attribution, scale_factor=16, mode='bilinear')
    transformer_attribution = transformer_attribution.reshape(224, 224).data.cpu().numpy()
    transformer_attribution = (transformer_attribution - transformer_attribution.min()) / (transformer_attribution.max() - transformer_attribution.min())
    return transformer_attribution


def get_timm_attentions(model, img):
    from torchvision.models.feature_extraction import create_feature_extractor
    block_id = 11
    attn_extractor = create_feature_extractor(
        model, return_nodes=[f'blocks.{block_id}.attn.softmax'])
    with torch.no_grad():
        out = attn_extractor(img)
    attn = out[f'blocks.{block_id}.attn.softmax']
    cls_attn = attn[0, :, 0, 1:]
    cls_attn = torch.mean(cls_attn, dim=0)
    cls_attn = cls_attn.reshape(14, 14)
    return cls_attn


def get_inter_map(model, img, method, device):
    """
    Generate interpretability maps for each method
    """
    if method == "gradcam":
        img = img.unsqueeze(0) 

        # Initialize GradCAM instance
        target_layers = [model.blocks[-1].norm1]
        reshape_transform = vit_reshape_transform

        # In the original implementation there is a parameter "use_cuda", which seems to be deprecated.
        # Currently, CAM instance uses the same CPU/GPU device of the given model.
        cam = GradCAM(model=model, target_layers=target_layers, reshape_transform=reshape_transform)

        targets = [ClassifierOutputTarget(1)]
        
        grayscale_cam = cam(input_tensor=img,
                        targets=targets,
                        eigen_smooth=True,
                        aug_smooth=True)
        inter_out = grayscale_cam[0, :]


    elif method == "chefer":
        from baselines.ViT.ViT_explanation_generator import LRP
        attribution_generator = LRP(model)
        inter_out = get_chefer_attribution(attribution_generator, img, device, class_index=1)
        
    elif method == "attention":
        img = img.unsqueeze(0) 
        attn_out = get_timm_attentions(model, img).to("cpu")
        repeat_factor = 224 // 14        
        upscaled_attn = np.kron(attn_out, np.ones((repeat_factor, repeat_factor)))
        inter_out = (upscaled_attn-np.min(upscaled_attn))/(np.max(upscaled_attn)-np.min(upscaled_attn))

    return inter_out


