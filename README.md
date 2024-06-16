# Attention Maps: A Solution to AI Medical Imaging Interpretability?

This repository contains the implementation of the project _"Are attention maps a solution to the interpretability crisis in AI-assisted medical imaging?"_. In this project, we investigated the reliability of attention maps of the Vision Transformer (ViT) as an interpretability methods under the context of medical image diagnosis. This repository provides Python codes to reproduce interpretability results of attention maps and two different methods: [Grad-CAM](https://github.com/jacobgil/pytorch-grad-cam) and the [Chefer method](https://github.com/hila-chefer/Transformer-Explainability).

## Code Usage & Reproduction of the Results

You can run `main.py` to generate interpretability maps and final evaluation results over all datasets and interpretability methods.

The structure of this repository is as follows:
* **Code** - *src/* folder contains necessary python files to perform the attack and calculate various stats (i.e., correctness and modification)

* **Data** - *data/* folder contains placeholding directories only without any image data. You may follow the instruction here to download medical image datasets that we used

* **Checkpoints** - *weights/* folder is a placeholding directory as well. `.pth` files of our selected trained models and MAE & DINO pre-trained ViT weights should be put in this directory.
