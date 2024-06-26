# Attention Maps: A Solution to AI Medical Imaging Interpretability?

This repository contains the implementation of the project _"Are attention maps a solution to the interpretability crisis in AI-assisted medical imaging?"_. In this project, we investigated the reliability of attention maps of the Vision Transformer (ViT) as an interpretability method under the context of medical image diagnosis. This repository provides Python codes to reproduce interpretability results of attention maps and two different methods: [Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam) and the [Chefer method](https://github.com/hila-chefer/Transformer-Explainability).

## Code Usage & Reproduction of the Results

You can run `main.py` to generate interpretability maps and final evaluation results over all datasets and interpretability methods.

The structure of this repository is as follows:
* **Code** - *src/* folder contains necessary python files to generate interpretability maps (including attention maps) and evaluate different interpretability methods.

* **Data** - *data/* folder contains placeholding directories only without any image data. You may follow the instruction [here](https://github.com/ugent-korea/attention_maps/blob/master/reproducibility_tutorial.md) to download medical image datasets that we used.

* **Checkpoints** - *weights/* folder is a placeholding directory as well. `.pth` files of our selected trained models and MAE & DINO pre-trained ViT weights should be put in this directory.


## Checkpoints
| Dataset / Method |           Random | Supervised | DINO | MAE |
|------------------|-----------------:|-----------:|-----:|----:|
| CP-Child         |[Download](Link)|[Download](Link)|[Download](Link)|[Download](Link)|
| DUKE             |[Download](Link)|[Download](Link)|[Download](Link)|[Download](Link)|
| Kvasir           |[Download](Link)|[Download](Link)|[Download](Link)|[Download](Link)|
| MURA             |[Download](Link)|[Download](Link)|[Download](Link)|[Download](Link)|


## References

Chefer, H., Gur, S., & Wolf, L. (2021). Transformer Interpretability Beyond Attention Visualization. 
_Proceedings of the IEEE Computer Society Conference on Computer Vision and Pattern Recognition_, 
782–791. https://doi.org/10.1109/CVPR46437.2021.00084

Dosovitskiy, A., Beyer, L., Kolesnikov, A., Weissenborn, D., Zhai, X., Unterthiner, T., Dehghani, M., 
Minderer, M., Heigold, G., Gelly, S., Uszkoreit, J., & Houlsby, N. (2020). An Image is Worth 16x16 
Words: Transformers for Image Recognition at Scale. _arxiv_.
https://doi.org/10.48550/arXiv.2010.11929

Selvaraju, R. R., Cogswell, M., Das, A., Vedantam, R., Parikh, D., & Batra, D. (2017). Grad-CAM: Visual 
Explanations from Deep Networks via Gradient-Based Localization. _Proceedings of the IEEE 
International Conference on Computer Vision_, 618–626. 
https://doi.org/10.1109/ICCV.2017.74
