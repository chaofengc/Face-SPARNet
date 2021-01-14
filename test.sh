export CUDA_VISIBLE_DEVICES=0
# ================================================================================
# Test SPARNet on Helen test dataset provided by DICNet
# ================================================================================

# python test.py --gpus 1 --model sparnet --name SPARNet_S16_V4_Attn2D \
    # --load_size 128 --dataset_name single --dataroot ../Helen_test_DIC/LR \
    # --pretrain_model_path ./pretrain_models/SPARNet-V16-S4-epoch20.pth \
    # --save_as_dir results_helen/SPARNet_S16_V4_Attn2D/

# python test.py --gpus 1 --model sparnet --name SPARNetLight_Attn3D \
    # --res_depth 1 --att_name spar3d \
    # --load_size 128 --dataset_name single --dataroot ../Helen_test_DIC/LR/ \
    # --pretrain_model_path ./pretrain_models/SPARNetLight_Attn3D-epoch20.pth \
    # --save_as_dir results_helen/SPARNetLight_Attn3D/

# ----------------- calculate PSNR/SSIM scores ----------------------------------
# python psnr_ssim.py
# ------------------------------------------------------------------------------- 

# ================================================================================
# Test SPARNetHD for aligned images
# ================================================================================

# ================================================================================
# Test SPARNetHD on single images
# ================================================================================

# python test_enhance_single_unalign.py --gpus 1 --model sparnethd --name SPARNetHD_V4_Attn2D \
    # --res_depth 10 --att_name spar --Gnorm 'in' \
    # --pretrain_model_path ./check_points/SPARNetHD_V4_Attn2D/latest_net_H.pth \
    # --test_img_path ./test_images/test_hzgg.jpg --results_dir test_hzgg_results

python test_enhance_single_unalign.py --gpus 1 --model sparnethd --name SPARNetHD_V4_Attn3D \
    --res_depth 10 --att_name spar3d --Gnorm 'in' \
    --pretrain_model_path ./check_points/SPARNetHD_V4_Attn3D/latest_net_H.pth \
    --test_img_path ./test_images/test_hzgg.jpg --results_dir test_hzgg_results
