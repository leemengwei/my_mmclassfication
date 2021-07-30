

# train model
python tools/train.py  configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet_pig_stand_or_crouch.py


# run on image 
python demo/image_demo.py ~/leemengwei/dataset/others/pig_stand_crouch_classification/val/7000095.jpg configs/mobilenet_v2/mobilenet_v2_b32x8_imagenet_pig_stand_or_crouch.py ~/useful_models/epoch_20_stand_or_crouch.pth