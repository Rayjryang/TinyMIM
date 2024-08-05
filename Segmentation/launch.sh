export CUDA_VISIBLE_DEVICES=2
cd TinyMIM
bash tools/dist_train.sh \
configs/mae/upernet_mae_base_12_512_slide_160k_ade20k.py 1 --seed 0 --work-dir ./ckpt/ \
--options model.pretrained="../../TinyMIM-PT-B.pth"