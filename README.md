# EndoHAR:Hierarchical Adaptive Representation of Foundation Models for Endoscopic Monocular Depth Estimation

## overview
![Image](Fig.png)

## Initialization
```
pip install -r requirements.txt
```
Depth anything model can download from [EndoDAC](https://github.com/BeileiCui/EndoDAC). You should create a folder such as ```pretrained model``` and place the downloaded model in it.

## Dataset
You can download the [SCARED dataset](https://endovissub2019-scared.grand-challenge.org/).and split the train/test used in ```splits/endovis``` folder.
### Data structure
Please follow [AF-SfMLearner](https://github.com/ShuweiShao/AF-SfMLearner) to convert videos into images and prepare data structures.

## Training and Evaluation
Need to export ground truth depth
```
python export_gt_depth.py --data_path <your_data_path> --split endovis
```
Need to export ground truth pose, we have prepared test sequence1 , you can also use other sequences
```
python export_gt_pose.py --data_path <your_data_path> --split endovis --sequence sequence<number>
```
### End-to-end training
```
python train_end_to_end.py --data_path <your_data_path> --log_dir './logs'
```
### Evaluation
Evaluate a single depth map
```
python test_simple.py --model_path <your_model_path>  --image_path <your_image_path>
```
The following example command evaluates the epoch 1 weights of the model
```
python evaluate_depth.py --data_path <your_data_path> --load_weights_folder "/hy-tmp/EndoHAR/logs/models/weights_1" --eval_mono
```




