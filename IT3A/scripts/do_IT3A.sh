

# # difftpt
python ./tpt_classification.py \
        --test_sets R --arch ViT-B/16 --b 8 --ctx_init a_photo_of_a --gpu 1 --tpt \
        --aug_mode difftpt --selection_cosine 0.8 --selection_selfentro 0.3 --lr 5e-3 --tta_steps 4 --num_prompts 8\
        --diff_root /data1/stuyuany/data_tpt/generation/imagenet-r_1k

