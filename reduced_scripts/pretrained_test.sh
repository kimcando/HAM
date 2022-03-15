python train.py --exp_name cnn_single \
                --model resnet18 \
                --lr 1e-3 \
                --pretrained true \
                --comm_round 500

python train.py --exp_name cnn_single \
                --model resnet18 \
                --lr 1e-3 \
                --pretrained false \
                --comm_round 500

python train.py --exp_name cnn_single \
                --model resnet18 \
                --lr 1e-4 \
                --pretrained true \
                --comm_round 500

python train.py --exp_name cnn_single \
                --model resnet18 \
                --lr 1e-4 \
                --pretrained false \
                --comm_round 500

python train.py --exp_name cnn_single \
                --model resnet18 \
                --lr 5e-5 \
                --pretrained true \
                --comm_round 500

python train.py --exp_name cnn_single \
                --model resnet18 \
                --lr 5e-5 \
                --pretrained false \
                --comm_round 500