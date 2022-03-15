python org_train.py --exp_name input_org \
                --model resnet18 \
                --lr 1e-4 \
                --pretrained false

python org_train.py --exp_name input_org \
                --model resnet18 \
                --lr 1e-4  \
                --pretrained true

python org_train.py --exp_name input_org \
                --model resnet18 \
                --lr 1e-4  \
                --pretrained true \
                --freeze true \
                --bn_freeze true

python org_train.py --exp_name input_org \
                --model resnet18 \
                --lr 1e-4  \
                --pretrained true \
                --freeze true \
                --bn_freeze false
