export CUDA_VISIBLE_DEVICES=2 && \
PYTHONPATH="/HDD_data_storage_2u_1/jinruiyang/shared_space/code/TinyMIM/Segmentation/TinyMIM":$PYTHONPATH \
python -m torch.distributed.launch --nproc_per_node=1 --master_port=29501 \
    train.py \
    ../configs/mae/upernet_mae_base_12_512_slide_160k_ade20k.py \
    --launcher pytorch --seed 0 --work-dir ./ckpt/ \
    --options model.pretrained="/HDD_data_storage_2u_1/jinruiyang/shared_space/code/TinyMIM/TinyMIM-PT-B.pth"
