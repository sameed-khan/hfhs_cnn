# HFHS T2DM CNN Classification
Classifying T2DM status from shoulder ultrasound using `ResNet50` with [RadImageNet](https://github.com/BMEII-AI/RadImageNet) backbone.

## Setup Instructions
1. Navigate inside folder:  
`cd hfhs_cnn`

2. Create new `conda` environment and activate it:<br>
`conda create -n hfhs_cnn python=3.10`<br>
`conda activate hfhs_cnn`

3. Install required packages:<br> 
`pip install -r requirements.txt`<br>
(Note: This installs cpu-only PyTorch since I don't expect the GPU to be up, so training will be *extremely* slow)

4. Run training script like so:<br>
`python train.py complete_filepaths.json`<br>
You can also run `python train.py -h` to see a list of hyperparameters that can be changed, such as `batch_size` (do this if there is an error with too much memory usage; you can lower the `batch_size`) or the number of epochs to train for.