import os
import torch
import argparse
from PIL import Image
from torchvision import transforms
from accelerate import Accelerator
from diffusers import StableDiffusionImageVariationPipeline
from torch.utils.data import Dataset
import random
import shutil


parser = argparse.ArgumentParser()
parser.add_argument("--batch_size", type = int, default = 5)
parser.add_argument("--data_dir", type = str)
parser.add_argument("--save_image_gen", type = str)
parser.add_argument("--dfu_times", type = int, default = 65)
args = parser.parse_args()


accelerator = Accelerator()
os.makedirs(args.save_image_gen, exist_ok = True)


class Dataset_ImageNetR(Dataset):
    def __init__(self, root, transform):
        self.root = root
        self.transform = transform
        self.folders = os.listdir(self.root)
        self.folders.sort()
        self.images = []
        for folder in self.folders:
            if not os.path.isdir(os.path.join(self.root, folder)):
                continue
            class_images = os.listdir(os.path.join(self.root, folder))
            class_images = list(map(lambda x: os.path.join(folder, x), class_images))
            random.shuffle(class_images)
            class_image = class_images[0:5]
            self.images  = self.images + class_image

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        image = self.transform(Image.open(os.path.join(self.root, self.images[idx])).convert('RGB'))
        return self.images[idx], image
    

def generate_images(pipe, dataloader, args):
    pipe, dataloader = accelerator.prepare(pipe, dataloader)
    pipe.safety_checker = lambda images, clip_input: (images, False)
    pipe = pipe.to(accelerator.device)
    with torch.no_grad():
        for count, (image_locations, original_images) in enumerate(dataloader):
            print(f'{count} / {len(dataloader)}, {image_locations[0]}.')

            for image_lo in image_locations:
                os.makedirs(os.path.join(args.save_image_gen, os.path.dirname(image_lo)), exist_ok = True)
                source_path = os.path.join(args.data_dir, image_lo)
                dist_path = os.path.join(args.save_image_gen, image_lo)
                
                if not os.path.exists(dist_path):
                    shutil.copyfile(source_path, dist_path)
                    with open(os.path.join(args.save_image_gen, 'selected_data_list.txt'), 'a+') as f:
                        f.write(dist_path+'\n')

            for time_ in range(args.dfu_times):
                images = pipe(original_images, guidance_scale = 3).images
                for index in range(len(images)):
                    # print(image_locations[index].split('.')[0]+'_'+str(126+time_)+'.'+image_locations[index].split('.')[1])
                    images[index].save(os.path.join(args.save_image_gen, image_locations[index].split('.')[0]+'_'+str(time_)+'.'+image_locations[index].split('.')[1]))


def main():
    model_name_path = "lambdalabs/sd-image-variations-diffusers"
    pipe = StableDiffusionImageVariationPipeline.from_pretrained(model_name_path, revision = "v2.0")
    
    tform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Resize(
            (224, 224),
            interpolation=transforms.InterpolationMode.BICUBIC,
            antialias=False,
            ),
        transforms.Normalize(
        [0.48145466, 0.4578275, 0.40821073],
        [0.26862954, 0.26130258, 0.27577711]),
    ])

    dataset = Dataset_ImageNetR(args.data_dir, tform)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    generate_images(pipe, dataloader, args)



if __name__ == "__main__":
    main()