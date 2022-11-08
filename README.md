# Hierarchical_CNN
## Reference paper
- [Hierarchical image classification in CNNs](http://cs229.stanford.edu/proj2019spr/report/18.pdf)

## Environment
- torch : 1.8.2+cu111
- torchvision : 0.9.2+cu111
- torchmetrics : 0.10.0
- numpy : 1.21.2
- wandb : 0.13.2
```
pip install -r requirements.txt
```  
<br>

## Dataset
- [Cifar100](https://www.cs.toronto.edu/~kriz/cifar.html)
    - Use Superclass for Coarse label
    - Use Classes for Fine label
```
python preprocess.py --dataset cifar100 --data_dir "data_dir" --types train,test
```

## Architecture
![Architecture](./images/architecture.jpg)
- Backbone : [Wide_ResNet_50](https://arxiv.org/abs/1605.07146)
    - Coarse : Using WRN50 Backbone expcet for `layer4`
    - Fine : Using whole WRN50 Backbone
- Main Goal is training Fine Label Classification
- Concatenating Coarse feature and Fine feature
- Using `1x1conv` layer for matching channel after concatenating
- Simply Using Fc layer for Classifier  
<br>

## Training (in progress)
```
The following hyperparameters are not the best ones.
The model is still training.
```
- Training w/ wandb.sweep
```
train.py --epochs 15 --batch_size 32 --lr 0.008 --backbone wide_resnet_50 --wandb --sweep --sweep_count 5
```
- Training w/o wandb.sweep
```
python train.py --epochs 15 --batch_size 32 --lr 0.008 --backbone wide_resnet_50 --wandb --milestones 7,13
```
- If you want to train only_fine model, just add `--only_fine` above command  
<br>

### Default Hyperparameter Setting
- `optimizer` : `SGD`
    - or you can use `Adam`
- `batch_size` : `32`
- `coarse` : `1`, `fine` : `2`
    - setting different Loss Weight
    - Coarse : Fine = 1 : 2
- `epochs` : `15`
- `lr` : `8e-3`
- `lr_scheduler` : `ReduceLROnPlateau`
    - or you can use `MultiStepLR` with using `milestones` argument

### optimizing model
- Using `wandb.sweep`
- Process
    1. Optimizing "only fine model" using `wandb.sweep`
        - batch_size :32
        - epochs : 15
        - lr : 0.0079
        - milestones : [7, 12]
    2. Traning "coarse+fine model" using above hyper parameter
- Graph only fine vs. coarse+fine model
    ![graph](./images/graph.png)  
<br>

## To do
- Find best parameter for Coarse+fine model
    - coarse and fine loss weight
    - lr
    - etc..
- Change the layer to freeze on the Coarse Backbone
- Apply other dataset with hierarchical label
    - Deep Fashion
    - etc..
- Apply other Backbone
    - [BiT](https://arxiv.org/abs/1912.11370)
- Apply different feature synthesis instead of concatenating