# HAM

### [Dataset Overview](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000)
Another more interesting than digit classification dataset to use to get biology and medicine students more excited about machine learning and image processing.

### Backgrounds and Personal Purpose
* To observe different behaviors of CNN using Grad-CAM as the training progresses when using **pretrained models** and **training from scratches**
* [Detailed exaplnatation on Experiments with PDF](https://github.com/kimcando/HAM/blob/main/assets/HAM_%EA%B9%80%EC%86%8C%EC%97%B0.pdf) in Korean

### Usage
* run
```
python train.py
```

* to find lr
```
python lr_finder_train.py
```

* experiments on partial freeze
```
python partial_freeze_train.py
```

* experiments on GradCAM
```
python gradcam_train.py
```
* scripts examples
  * refer to `full_scripts/`

* arguments settings
  * refer to [arguments.py](https://github.com/kimcando/HAM/blob/main/arguments.py)


### base code reference
* [kaggle notebooks](https://www.kaggle.com/code/xinruizhuang/skin-lesion-classification-acc-90-pytorch)
