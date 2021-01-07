# ===================== DEBUG =====================================

# python train.py --gpus 2 --name debug --model enhance \
    # --input_nc 3 --output_nc 3 --gan_mode lsgan --lr 0.0002 --Pnorm bn \
    # --Pnorm bn --Gnorm "in" --Dnorm none --n_layers_D 4 \
    # --dataroot ../srdata/CelebAMask-HQ --dataset_name celebamask --batch_size 4 \
    # --debug --save_iter_freq 1 --save_latest_freq 1 --visual_freq 1 --print_freq 1 

# ==================== Train ============================================

# python train.py --gpus 2 --name FaceParse_Resnet_v001 --model parse \
    # --input_nc 3 --output_nc 3 --lr 0.0002  --Pnorm bn \
    # --dataroot ../srdata/CelebAMask-HQ --dataset_name celebamask --batch_size 8 \

python train.py --gpus 4 --name FaceEnhance_SPADENet_v001 --model enhance \
    --input_nc 3 --output_nc 3 --gan_mode lsgan --lr 0.0002  \
    --Pnorm bn --Gnorm "in" --Dnorm none --n_layers_D 4  \
    --dataroot ./srdata/CelebAMask-HQ --dataset_name celebamask --batch_size 4 \
    --visual_freq 100 --print_freq 10 \
    --parse_net_weight ./pretrain_models/latest_net_P.pth 

