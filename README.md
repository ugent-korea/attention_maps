# Attention Maps: A Solution to AI Medical Imaging Interpretability?

This repository contains the implementation of the project _"Are attention maps a solution to the interpretability crisis in AI-assisted medical imaging?"_. In this project, we investigated the reliability of attention maps of the Vision Transformer (ViT) as an interpretability method under the context of medical image diagnosis. This repository provides Python codes to reproduce interpretability results of attention maps and two different methods: [Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam) and the [Chefer method](https://github.com/hila-chefer/Transformer-Explainability).

## Code Usage & Reproduction of the Results

You can run `main.py` to generate interpretability maps and final evaluation results over all datasets and interpretability methods.

The structure of this repository is as follows:
* **Code** - *src/* folder contains necessary python files to generate interpretability maps (including attention maps) and evaluate different interpretability methods.

* **Data** - *data/* folder contains placeholding directories only without any image data. You may follow the instruction [here](https://github.com/ugent-korea/attention_maps/blob/master/reproducibility_tutorial.md) to download medical image datasets that we used.

* **Checkpoints** - *weights/* folder is a placeholding directory as well. `.pth` files of our selected trained models and MAE & DINO pre-trained ViT weights should be put in this directory.

## Training
### Hyperparameter Settings

| Dataset   | Model init. | Optimizer | Learning Rate | Momentum | Weight Decay | Scheduler       | Total Epoch | Saved Epoch | Batch Size |
|-----------|-------------|-----------|---------------|----------|--------------|-----------------|-------------|-------------|------------|
| CP-Child  | Random      | Adam      | 0.00001       | 0.9      | 0.0001       | CosineAnnealing | 20          | 18          | 64         |
|           | Sup         | Adam      | 0.00001       | 0.9      | 0.0001       | CosineAnnealing | 20          | 5           | 64         |
|           | DINO        | Adam      | 0.00001       | 0.9      | 0.0001       | CosineAnnealing | 20          | 12          | 64         |
|           | MAE         | Adam      | 0.00001       | 0.9      | 0.0001       | CosineAnnealing | 20          | 11          | 64         |
| DUKE      | Random      | SGD       | 0.1           | 0.3      | 0            | CosineAnnealing | 200         | 79          | 64         |
|           | Sup         | SGD       | 0.03          | 0.1      | 0            | CosineAnnealing | 20          | 2           | 64         |
|           | DINO        | SGD       | 0.005         | 0.1      | 0            | CosineAnnealing | 15          | 5           | 64         |
|           | MAE         | SGD       | 0.03          | 0.1      | 0            | CosineAnnealing | 15          | 2           | 64         |
| Kvasir    | Random      | SGD       | 0.08          | 0.1      | 0            | CosineAnnealing | 25          | 24          | 32         |
|           | Sup         | SGD       | 0.0018        | 0.1      | 0            | CosineAnnealing | 25          | 8           | 32         |
|           | DINO        | SGD       | 0.0012        | 0.1      | 0.0001       | CosineAnnealing | 25          | 20          | 32         |
|           | MAE         | SGD       | 0.02          | 0.1      | 0.0001       | CosineAnnealing | 25          | 25          | 32         |
| MURA      | Random      | SGD       | 0.001         | 0.9      | 0.0001       | CosineAnnealing | 200         | 180         | 64         |
|           | Sup         | SGD       | 0.001         | 0.9      | 0.0001       | CosineAnnealing | 100         | 9           | 64         |
|           | DINO        | SGD       | 0.001         | 0.9      | 0.0001       | CosineAnnealing | 150         | 43          | 64         |
|           | MAE         | SGD       | 0.001         | 0.9      | 0.0001       | CosineAnnealing | 100         | 35          | 64         |



### Checkpoints
| Dataset / Method |           Random | Supervised | DINO | MAE |
|------------------|-----------------:|-----------:|-----:|----:|
| CP-Child         |[Download](https://drive.google.com/file/d/12yZ1JxoEnSNuXQIfoWvs-DvN9DaI6ZMp/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1junOUWWRRTPtvauJeIxu1dzwjMtYwOSl/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1zL8fJW69Ze4EfvPt58k2idNJ8kLhVR-c/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1eI38mIIHDod2czM4klsIgf_3tyyimWaJ/view?usp=drive_link)|
| DUKE             |[Download](https://drive.google.com/file/d/16--L8NNOH1z_cKdUNza7riAMw8ngoVxv/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1n7K4EWd0ALQ5RE0dSqL-P8zV-hUfrsVG/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1YGj4B-u3ztB6qz3NA--11KT7_hWJkYIo/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1IBt_X3Rv-GybjquOjcQnyvY0R7oZhTtT/view?usp=drive_link)|
| Kvasir           |[Download](https://drive.google.com/file/d/1HtV6cTiSA7I01_fLIthprQ-ViTPvXCb9/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1dPqiPFOmear24XYUPzsIHT52oF65clMs/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1nAWSTKJ05xeMMq2R05LioVcu1WiDmNmw/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1Kc4rEEfT8w5MIFdhmV9YUyJY3zwH-8Yf/view?usp=drive_link)|
| MURA             |[Download](https://drive.google.com/file/d/1uwGYNym6vnQTxDUtdKTyL_KVmwrNCyJo/view?usp=drive_link)|[Download](https://drive.google.com/file/d/16vpyWh9gfj0TwJZBeNWD1ymcyvkFqpCl/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1Pg2ChmMVHUZqZmhpn1_Jrpyt9VBkocNR/view?usp=drive_link)|[Download](https://drive.google.com/file/d/1nSflcnkNG4dJ2dFaYuipoOFYk76kT9uU/view?usp=drive_link)|





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
