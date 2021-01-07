# ===================== DEBUG =====================================

# python train.py --gpus 2 --name debug --model enhance \
    # --input_nc 3 --output_nc 3 --gan_mode lsgan --lr 0.0002 --Pnorm bn \
    # --Pnorm bn --Gnorm "in" --Dnorm none --n_layers_D 4 \
    # --dataroot ../srdata/CelebAMask-HQ --dataset_name celebamask --batch_size 4 \
    # --debug --save_iter_freq 1 --save_latest_freq 1 --visual_freq 1 --print_freq 1 

# python train.py --gpus 4 --name debug --model sparnet \
    # --input_nc 3 --output_nc 3 --gan_mode hinge --lr 0.0002  \
    # --Gnorm "in" --Dnorm "in" --n_layers_D 3  \
    # --dataroot ../CelebAMask-HQ --dataset_name celebamask --batch_size 4 \
    # --debug --save_iter_freq 1 --save_latest_freq 1 --visual_freq 1 --print_freq 1 


# ==================== Train ============================================

# python train.py --gpus 2 --name FaceParse_Resnet_v001 --model parse \
    # --input_nc 3 --output_nc 3 --lr 0.0002  --Pnorm bn \
    # --dataroot ../srdata/CelebAMask-HQ --dataset_name celebamask --batch_size 8 \

# python train.py --gpus 4 --name SPARNet_v001 --model sparnet \
    # --input_nc 3 --output_nc 3 --gan_mode hinge --lr 0.0002  \
    # --Gnorm "in" --Dnorm "in" --n_layers_D 4  \
    # --dataroot ../CelebAMask-HQ --dataset_name celebamask --batch_size 4 \
    # --visual_freq 100 --print_freq 10 


# python train.py --gpus 4 --name SPARNet_v002 --model sparnet \
    # --input_nc 3 --output_nc 3 --gan_mode hinge --lr 0.0002  \
    # --Gnorm "in" --Dnorm "in" --n_layers_D 3  \
    # --dataroot ../CelebAMask-HQ --dataset_name celebamask --batch_size 4 \
    # --visual_freq 100 --print_freq 10 --continue_train --load_iter 50000


# python train.py --gpus 4 --name SPARNet_Noss_v001 --model sparnet --not_use_ss \
    # --input_nc 3 --output_nc 3 --gan_mode hinge --lr 0.0002  \
    # --Gnorm "in" --Dnorm "in" --n_layers_D 3  \
    # --dataroot ../CelebAMask-HQ --dataset_name celebamask --batch_size 4 \
    # --visual_freq 100 --print_freq 10 # --continue_train --load_iter 5000

# python train.py --gpus 4 --name SPARNet_HG_v001 --model sparnet --not_use_ss \
    # --input_nc 3 --output_nc 3 --gan_mode hinge --lr 0.0002  \
    # --Gnorm "in" --Dnorm "in" --n_layers_D 3  \
    # --dataroot ../CelebAMask-HQ --dataset_name celebamask --batch_size 4 \
    # --visual_freq 100 --print_freq 10 --continue_train --load_iter 30000

# python train.py --gpus 4 --name SPARNet_SS_v001 --model sparnet \
    # --input_nc 3 --output_nc 3 --gan_mode hinge --lr 0.0002  \
    # --Gnorm "in" --Dnorm "in" --n_layers_D 3  \
    # --dataroot ../CelebAMask-HQ --dataset_name celebamask --batch_size 4 \
    # --visual_freq 100 --print_freq 10 --continue_train --load_iter 5000 

# python train.py --gpus 4 --name SPARNet_HGSS_v001 --model sparnet \
    # --input_nc 3 --output_nc 3 --gan_mode lsgan --lr 0.0002  \
    # --Gnorm "in" --Dnorm "in" --n_layers_D 3  \
    # --dataroot ../CelebAMask-HQ --dataset_name celebamask --batch_size 4 \
    # --visual_freq 100 --print_freq 10 --continue_train --load_iter 5000


python train.py --gpus 2 --name SPARNet_HGSS_v200 --model sparnet --not_use_ss \
    --input_nc 3 --output_nc 3 --gan_mode hinge --lr 0.0002  \
    --Gnorm "in" --Dnorm "in" --n_layers_D 4 --D_num 3  \
    --lambda_pix 100 --lambda_fm 1 --lambda_pcp 0 \
    --dataroot ../FFHQ1024 --dataset_name ffhq --batch_size 2 \
    --visual_freq 100 --print_freq 10 --save_latest_freq 500 --continue_train #--load_iter 5000


# python train.py --gpus 4 --name SPARNet_HG_NOSS_v002 --model sparnet --not_use_ss \
    # --input_nc 3 --output_nc 3 --gan_mode hinge --lr 0.0002  \
    # --Gnorm "in" --Dnorm "in" --n_layers_D 4 --D_num 3  \
    # --dataroot ../ffhq1024 --dataset_name ffhq --batch_size 4 \
    # --visual_freq 100 --print_freq 10 # --continue_train #--load_iter 5000

# python train.py --gpus 4 --name SPARStyle_v001 --model sparstyle --not_use_ss \
    # --input_nc 3 --output_nc 3 --gan_mode hinge --lr 0.0002  \
    # --lambda_pix 10 --lambda_fm 1 --lambda_pcp 1 \
    # --Gnorm "in" --Dnorm "in" --n_layers_D 4 --D_num 3  \
    # --dataroot ../ffhq1024 --nwdn_root ../nwdn_data --dataset_name comb --batch_size 4 \
    # --visual_freq 100 --print_freq 10  --continue_train #--load_iter 5000

# python train.py --gpus 4 --name SPARNet_BNSA_NOSS_v002 --model sparnet --not_use_ss \
    # --input_nc 3 --output_nc 3 --gan_mode hinge --lr 0.0002  \
    # --Gnorm "in" --Dnorm "in" --n_layers_D 4 --D_num 3  \
    # --dataroot ../ffhq1024 --dataset_name ffhq --batch_size 4 \
    # --visual_freq  100 --print_freq 10 --continue_train #--load_iter 5000

