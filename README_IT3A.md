# 【IJCV 2025】 Diffusion-Enhanced Test-time Adaptation with Text and Image Augmentation

This repo is the official implementation of [IT3A](https://arxiv.org/abs/2412.09706).

## Abstract
xxxx



## Get started

### Installation

* We adopt [generative-robustness](https://github.com/Hritikbansal/generative-robustness?utm_source=catalyzex.com) enviroment as our dependency.
```bash
# Clone this repo
git clone https://github.com/chunmeifeng/DiffTPT.git
cd DiffTPT

# Create a conda enviroment
1. conda env create -f environment.yml
2. conda activate difftpt
3. pip install torch==1.13.0+cu117 torchvision==0.14.0+cu117 torchaudio==0.13.0+cu117 -f https://download.pytorch.org/whl/torch_stable.html
4. accelerate config
- This machine
- multi-GPU
- (How many machines) 1
- (optimize with dynamo) NO
- (Deepspeed) NO
- (FullyShardedParallel) NO
- (MegatronLM) NO
- (Num of GPUs) 5
- (device ids) 0,1,2,3,4
- (np/fp16/bp16) no
```

### Datasets
We evaluate our method in two Scenarios:
1. S<sub>1</sub>: Natural Distribution Shifts
    * ImageNet
    * ImageNet-V2
    * ImageNet-A
    * ImageNet-R
    * ImageNet-Sketch
2. S<sub>2</sub>: Cross-Datasets Generalization
    * Flower102
    * OxfordPets
    * SUN397
    * DTD
    * Food101
    * StanfordCars
    * Aircraft
    * UCF101
    * EuroSAT
    * Caltech101

* Please refer to [CoOp/CoCoOp](https://github.com/KaiyangZhou/CoOp) and [TPT](https://github.com/azshue/TPT) for more details on data.


### How to Run
1. Stable Diffusion based data augmentation
```bash
# for ImageNet-R
accelerate launch --num_cpu_threads_per_process 8 -m image_generator_R --data_dir your_data_set_path/imagenet-r --save_image_gen your_output_data_path/imagenet-r_1k
```
* Please refer to [generative-robustness](https://github.com/Hritikbansal/generative-robustness?utm_source=catalyzex.com) for more details.

2. DiffTPT
```bash
bash scripts/do_tpt_difftpt.sh
```

## Citation

```  
@inproceedings{feng2023diverse,
  title={Diverse data augmentation with diffusions for effective test-time prompt tuning},
  author={Feng, Chun-Mei and Yu, Kai and Liu, Yong and Khan, Salman and Zuo, Wangmeng},
  booktitle={Proceedings of the IEEE/CVF International Conference on Computer Vision},
  pages={2704--2714},
  year={2023}
}
```

## Acknowledgements

We extend our appreciation to the developers of the [CoOp/CoCoOp](https://github.com/KaiyangZhou/CoOp) and [TPT](https://github.com/azshue/TPT) project for sharing their open-source implementation and providing guidance on preparing the data.

