# HFHS CNN Project Notebook
Author: Sameed Khan, Cleveland Clinic Lerner College of Medicine <br>
Created: 09/25/2023 <br>
Last Updated: 10/07/2023<br>

**Purpose**: A chronological journal of ideas and experiments over the lifetime of this project.
This is provided for completeness and personal note-taking.
It serves as a record of experiments and ideas tried for later replication and reference for what did / did not work.

## Planning
- Several angles of potential data including `stil` and `cine` images
- From radiomics workup and previous manuscript, segmented muscle performed worse for analysis compared to bounding box object detection which grabbed a small central region of muscle
- Most critical feature from that work was whether patients had "large images" - if patients contained an image that had a size greater than 960 x 720, model performance drastically increased and they were very easy to classify
  - This may just be an unknown confounder of the dataset; I did not have enough information to verify
- For CNN project - using a CNN to predict the diabetes class of the patient
  - need to decide what kind of data we are inputting - we have `cine` studies with thousands of frames we could plug into a transformer or LSTM
  - problem from radiomics study is that the data appears *very noisy* for the feature we are trying to find
    - Adding segmentation of entire muscle worsened performance of radiomics instead of augmenting it
  - ~~Following the above finding, it makes sense to limit the input to basically two images per patient, one short-axis bicep stil and another long-axis supra stil~~
  - Now just using long-axis supra **only**
    - If performance >= 0.8 AUROC on CNN, then it could make sense to start including other angles as separate channels
  - In our sample dataset, what kinds of images are present? See `check_images.ipynb`
    - Lots of heterogeneity in terms of the sizing of the images
    - The `dicom` header contains a `RegionLocationMinX0` and corresponding coordinate properties that can be used to crop the dicom image, but this still leaves the burned-in annotation as well as the grayscale bar. Also leaves black bars on the left and right of the image for Phillips DICOM sequences.
    - Defaulted to just using image cropping rules derived by MSU data science students 

## Other Notes
- All US images in `sample_data` that were used for segmentation training are diabetes-positive patients
- Previous radiomics analysis finds that radiomics features correlate highly with BMI, so it is possible that we are just picking up on fat infiltration
  - In the small subset analysis that differentiated very well, all of the patients in the diabetic group were much older, which contributed to an almost 0.1 rise in `AUROC` once you included the `AgeAtTimeofStudy` feature

## Update History
**10/07/2023<br>**
- First draft of training code
- Logs performance metrics `AUROC`, `F1`, `Accuracy`
- Only takes long-axis supraspinatus as input
- Uses [RadImageNet](https://github.com/BMEII-AI/RadImageNet) backbone as opposed to ImageNet
