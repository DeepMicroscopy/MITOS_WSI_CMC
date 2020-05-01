![Large-Scale Canine Mammary Carcinoma Data Set for Mitotic Figure Assessment on Whole Slide Images](title_CMC.png)

## MITOS_WSI_CMC data set

This repository contains all code needed to derive the Technical Validation of our paper:
> Aubreville, M., Bertram, C.A., Marzahl, C. Maier, A., & Klopfleisch R. (2020): Dogs as Model for Human Breast Cancer: A Completely Annotated Whole Slide Image Dataset (submitted)

It contains two main parts:

### Data set variant evaluation

This folder contains the evaluation for all variants, i.e. the manually labelled (MEL), the the object-detection augmented manually expert labelled (ODAEL), and the clustering- and object detection augmented manually expert labelled (CODAEL) variant.

Main results of the data set variants based on a one- and two-stage-detector can be found in [Evaluation.ipynb](Evaluation.ipynb).

## Setting up the environment

Besides [https://github.com/fastai/](fast.ai) you can use the following notebook to set up the dataset for you: [Setup.ipynb](Setup.ipynb). The download of the WSI from figshare will take a while. Once everything has been downloaded, you can either use the data loaders provided in this repository, or, if you want to get a visual impression of the dataset, use [our annotation tool SlideRunner](https://github.com/maubreville/SlideRunner). The SlideRunner package (which can be acquired using pip) is also a pre-requisite to run the code.

## Training notebooks

The training process can be seen in the notebooks for the respective dataset variants:

[RetinaNet-CMC-MEL.ipynb](RetinaNet-CMC-MEL.ipynb)

[RetinaNet-CMC-ODAEL.ipynb](RetinaNet-CMC-ODAEL.ipynb)

[RetinaNet-CMC-CODAEL.ipynb](RetinaNet-CMC-CODAEL.ipynb)

