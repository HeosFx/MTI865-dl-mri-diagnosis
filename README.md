# MTI865 - Deep Learning MRI diagnosis

## Introduction
This project originates from a proposal by one of our professors at ÉTS Montréal. It is inspired by the **Automated Cardiac Diagnosis Challenge (ACDC)**, a benchmark competition first introduced to the medical imaging community during the **MICCAI 2017** conference.

The original ACDC challenge focused on developing automated methods for diagnosing cardiac conditions using cardiac MRI (Magnetic Resonance Imaging). The dataset provided for the competition includes short-axis cine MRI scans from multiple patients, along with expert-labeled annotations. These labels cover key cardiac structures—such as the left ventricle, right ventricle, and myocardium—as well as diagnostic information like the presence of cardiomyopathy or other abnormalities.

In essence, the goal of the challenge is to encourage the development of machine learning and deep learning techniques capable of:

- Segmenting cardiac structures in MRI images,
- Extracting meaningful anatomical and functional features, and
- Performing automated cardiac diagnosis based on these findings.

Our project adapts this challenge to a course setting. We work with a subset of the ACDC dataset and aim to design and evaluate models capable of solving similar tasks. The project thus exposes us to real-world medical imaging data, common preprocessing pipelines, segmentation models (like U-Net), and diagnostic prediction methods used in clinical AI research.

Due to time constraints, our session project focused exclusively on the segmentation task. While the original ACDC challenge also includes diagnostic classification based on extracted cardiac features, we limited the scope to developing and evaluating models capable of accurately segmenting the key cardiac structures in MRI images—specifically the left ventricle, right ventricle, and myocardium. This allowed us to concentrate on mastering the preprocessing steps, implementing segmentation architectures such as U-Net, and analyzing performance metrics like Dice scores, all within the timeframe of the course.

## Team
- [@paulmerceur](https://github.com/paulmerceur)
- [@Joaquim2805](https://github.com/Joaquim2805)
- [@HeosFx](https://github.com/HeosFx)

## Repository
This repository is a public copy of the original project, created to respect dataset privacy constraints. It contains all the Python code required to train and evaluate our U-Net segmentation model.

## Results
The results of our experiments and analyses are presented in the report `Projet_MTI865`.
