

# # difftpt
# python ./tpt_classification.py \
#         --test_sets R --arch RN50 --b 64 --ctx_init a_photo_of_a --gpu 1 --tpt \
#         --aug_mode difftpt --selection_cosine 0.8 --selection_selfentro 0.3 --lr 5e-3 --tta_steps 4 \
#         --diff_root your_output_data_path/imagenet-r_1k/

# # difftpt_coop
# python ./tpt_classification.py \
#         --test_sets R --arch RN50 --b 64 --ctx_init a_photo_of_a --gpu 1 --tpt \
#         --aug_mode difftpt --selection_cosine 0.8 --selection_selfentro 0.3 --lr 5e-3 --tta_steps 4 \
#         --diff_root your_output_data_path/imagenet-r_1k/ \
#         --load /path/to/pretrained/coop/weight.pth

# # difftpt_cocoop
# python ./tpt_classification.py \
#         --test_sets R --arch RN50 --b 64 --ctx_init a_photo_of_a --gpu 1 --tpt \
#         --aug_mode difftpt --selection_cosine 0.8 --selection_selfentro 0.3 --lr 5e-3 --tta_steps 4 \
#         --diff_root your_output_data_path/imagenet-r_1k/ \
#         --load /path/to/pretrained/cocoop/weight.pth \
#         --cocoop



# # tpt
# python ./tpt_classification.py \
#         --test_sets R --arch RN50 --b 64 --ctx_init a_photo_of_a --gpu 1 --tpt \
#         --aug_mode tpt --selection_cosine 0.8 --selection_selfentro 0.3 --lr 5e-3 --tta_steps 4 \
#         --diff_root your_output_data_path/imagenet-r_1k/

# # tpt_coop
# python ./tpt_classification.py \
#         --test_sets R --arch RN50 --b 64 --ctx_init a_photo_of_a --gpu 1 --tpt \
#         --aug_mode tpt --selection_cosine 0.8 --selection_selfentro 0.3 --lr 5e-3 --tta_steps 4 \
#         --diff_root your_output_data_path/imagenet-r_1k/ \
#         --load /path/to/pretrained/coop/weight.pth

# # tpt_cocoop
# python ./tpt_classification.py \
#         --test_sets R --arch RN50 --b 64 --ctx_init a_photo_of_a --gpu 1 --tpt \
#         --aug_mode tpt --selection_cosine 0.8 --selection_selfentro 0.3 --lr 5e-3 --tta_steps 4 \
#         --diff_root your_output_data_path/imagenet-r_1k/ \
#         --load /path/to/pretrained/cocoop/weight.pth \
#         --cocoop