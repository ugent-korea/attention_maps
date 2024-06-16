# How to get and preprocess datasets

## Duke-Breast-Cancer-MRI
You can follow [this tutorial](https://sites.duke.edu/mazurowski/2022/07/13/breast-mri-cancer-detect-tutorial-part1/) written by original paper's author to get Duke-Breast-Cancer-MRI (DBC-MRI) dataset in `.png` format.
In our project, MRI volumes from patient 1 to 900 are used and the number of positive images are reduced according to that of negative images to deal with class imbalance issue.
All the images are resized into 256 by 256 pixels.

## Musculoskeletal radiographs (MURA)
Go to section 'Downloading the Dataset (v1.0)' at [this link](https://stanfordmlgroup.github.io/competitions/mura/). It will redirect you to Stanford AIMI website.
You have to register and log in the website to access the data. After loggin in, register for MURA: MSK X-rays. 
All the images are preprocessed by resizing the shortest side to 256 pixels while maintaining the aspect ratio.

## Kvasir
You can download Kvasir version 2 dataset at [here](https://datasets.simula.no/kvasir/).
All the images are preprocessed by resizing the shortest side to 256 pixels while maintaining the aspect ratio, resulting in two different resolutions: 320 by 256 or 458 by 256 pixels.
Since 458 by 256 images are containing additional black margin (i.e., background), they are cropped so that all the images are 360 by 256 pixels.

## CP-CHILD
You can download the dataset [here](https://figshare.com/articles/dataset/CP-CHILD_zip/12554042). As all the images are already 256 by 256 pixels, no additional processing is required.


