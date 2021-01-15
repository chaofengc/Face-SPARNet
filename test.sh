# export CUDA_VISIBLE_DEVICES=$1
# ================================================================================
# Test SPARNet on Helen test dataset provided by DICNet
# ================================================================================

python test.py --gpus 1 --model sparnet --name SPARNet_S16_V4_Attn2D \
    --load_size 128 --dataset_name single --dataroot test_dirs/Helen_test_DIC/LR \
    --pretrain_model_path ./pretrain_models/SPARNet-V16-S4-epoch20.pth \
    --save_as_dir results_helen/SPARNet_S16_V4_Attn2D/

python test.py --gpus 1 --model sparnet --name SPARNetLight_Attn3D \
    --res_depth 1 --att_name spar3d \
    --load_size 128 --dataset_name single --dataroot test_dirs/Helen_test_DIC/LR/ \
    --pretrain_model_path ./pretrain_models/SPARNetLight_Attn3D-epoch20.pth \
    --save_as_dir results_helen/SPARNetLight_Attn3D/

# ----------------- calculate PSNR/SSIM scores ----------------------------------
python psnr_ssim.py
# ------------------------------------------------------------------------------- 

# ================================================================================
# Test SPARNetHD for aligned images
# ================================================================================

python test.py --gpus 1 --model sparnethd --name SPARNetHD_V4_Attn2D \
    --res_depth 10 --att_name spar --Gnorm 'in' \
    --load_size 512 --dataset_name single --dataroot test_dirs/CelebA-TestN/ \
    --pretrain_model_path ./pretrain_models/SPARNetHD_V4_Attn2D_net_H-epoch10.pth \
    --save_as_dir results_CelebA-TestN/SPARNetHD_V4_Attn2D/

python test.py --gpus 1 --model sparnethd --name SPARNetHD_V4_Attn3D \
    --res_depth 10 --att_name spar3d --Gnorm 'in' \
    --load_size 512 --dataset_name single --dataroot test_dirs/CelebA-TestN/ \
    --pretrain_model_path ./pretrain_models/SPARNetHD_V4_Attn3D_net_H-epoch10.pth \
    --save_as_dir results_CelebA-TestN/SPARNetHD_V4_Attn3D/

# ----------------- calculate FID scores ----------------------------------
python -m pytorch_fid results_CelebA-TestN/SPARNetHD_V4_Attn2D/ test_dirs/CelebAHQ-Test-HR 
python -m pytorch_fid results_CelebA-TestN/SPARNetHD_V4_Attn3D/ test_dirs/CelebAHQ-Test-HR 
# ------------------------------------------------------------------------------- 

# ================================================================================
# Test SPARNetHD on single images
# ================================================================================

python test_enhance_single_unalign.py --gpus 1 --model sparnethd --name SPARNetHD_V4_Attn2D \
    --res_depth 10 --att_name spar --Gnorm 'in' \
    --pretrain_model_path ./pretrain_models/SPARNetHD_V4_Attn2D_net_H-epoch10.pth \
    --test_img_path ./test_images/test_hzgg.jpg --results_dir test_hzgg_results

python test_enhance_single_unalign.py --gpus 1 --model sparnethd --name SPARNetHD_V4_Attn3D \
    --res_depth 10 --att_name spar3d --Gnorm 'in' \
    --pretrain_model_path ./pretrain_models/SPARNetHD_V4_Attn3D_net_H-epoch10.pth \
    --test_img_path ./test_images/test_hzgg.jpg --results_dir test_hzgg_results
