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

imagenet_r_fold = ['n01443537', 'n01484850', 'n01494475', 'n01498041', 'n01514859', 'n01518878', 'n01531178', 'n01534433', 'n01614925', 'n01616318', 'n01630670', 'n01632777', 'n01644373', 'n01677366', 'n01694178', 'n01748264', 'n01770393', 'n01774750', 'n01784675', 'n01806143', 'n01820546', 'n01833805', 'n01843383', 'n01847000', 'n01855672', 'n01860187', 'n01882714', 'n01910747', 'n01944390', 'n01983481', 'n01986214', 'n02007558', 'n02009912', 'n02051845', 'n02056570', 'n02066245', 'n02071294', 'n02077923', 'n02085620', 'n02086240', 'n02088094', 'n02088238', 'n02088364', 'n02088466', 'n02091032', 'n02091134', 'n02092339', 'n02094433', 'n02096585', 'n02097298', 'n02098286', 'n02099601', 'n02099712', 'n02102318', 'n02106030', 'n02106166', 'n02106550', 'n02106662', 'n02108089', 'n02108915', 'n02109525', 'n02110185', 'n02110341', 'n02110958', 'n02112018', 'n02112137', 'n02113023', 'n02113624', 'n02113799', 'n02114367', 'n02117135', 'n02119022', 'n02123045', 'n02128385', 'n02128757', 'n02129165', 'n02129604', 'n02130308', 'n02134084', 'n02138441', 'n02165456', 'n02190166', 'n02206856', 'n02219486', 'n02226429', 'n02233338', 'n02236044', 'n02268443', 'n02279972', 'n02317335', 'n02325366', 'n02346627', 'n02356798', 'n02363005', 'n02364673', 'n02391049', 'n02395406', 'n02398521', 'n02410509', 'n02423022', 'n02437616', 'n02445715', 'n02447366', 'n02480495', 'n02480855', 'n02481823', 'n02483362', 'n02486410', 'n02510455', 'n02526121', 'n02607072', 'n02655020', 'n02672831', 'n02701002', 'n02749479', 'n02769748', 'n02793495', 'n02797295', 'n02802426', 'n02808440', 'n02814860', 'n02823750', 'n02841315', 'n02843684', 'n02883205', 'n02906734', 'n02909870', 'n02939185', 'n02948072', 'n02950826', 'n02951358', 'n02966193', 'n02980441', 'n02992529', 'n03124170', 'n03272010', 'n03345487', 'n03372029', 'n03424325', 'n03452741', 'n03467068', 'n03481172', 'n03494278', 'n03495258', 'n03498962', 'n03594945', 'n03602883', 'n03630383', 'n03649909', 'n03676483', 'n03710193', 'n03773504', 'n03775071', 'n03888257', 'n03930630', 'n03947888', 'n04086273', 'n04118538', 'n04133789', 'n04141076', 'n04146614', 'n04147183', 'n04192698', 'n04254680', 'n04266014', 'n04275548', 'n04310018', 'n04325704', 'n04347754', 'n04389033', 'n04409515', 'n04465501', 'n04487394', 'n04522168', 'n04536866', 'n04552348', 'n04591713', 'n07614500', 'n07693725', 'n07695742', 'n07697313', 'n07697537', 'n07714571', 'n07714990', 'n07718472', 'n07720875', 'n07734744', 'n07742313', 'n07745940', 'n07749582', 'n07753275', 'n07753592', 'n07768694', 'n07873807', 'n07880968', 'n07920052', 'n09472597', 'n09835506', 'n10565667', 'n12267677']

class Dataset_ImageNetR(Dataset):
    def __init__(self, root, transform, filter_list=None):
        self.root = root
        self.transform = transform
        self.folders = os.listdir(self.root)
        self.folders.sort()
        self.images = []
        if filter_list is not None:
            self.labels = []
        for folder in self.folders:
            if not os.path.isdir(os.path.join(self.root, folder)):
                continue
            class_images = os.listdir(os.path.join(self.root, folder))
            class_images = list(map(lambda x: os.path.join(folder, x), class_images))
            random.shuffle(class_images)
            if filter_list is not None:
                val_list = []
                with open(filter_list, 'r') as f:
                    for line in f.readlines():
                        read_img = line.strip()
                        val_list.append(read_img)
                class_image = [img for img in class_images if os.path.join(root, img) not in val_list]
                labels = [imagenet_r_fold.index(img.split('/')[-2]) for img in class_image]
                self.labels = self.labels + labels
            else:
                class_image = class_images[0:5]
            self.images  = self.images + class_image

    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, idx):
        if hasattr(self, 'labels'):
            image = self.transform(Image.open(os.path.join(self.root, self.images[idx])).convert('RGB'))
            return image, self.labels[idx]
        else:
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


def main(args):
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
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type = int, default = 5)
    parser.add_argument("--data_dir", type = str)
    parser.add_argument("--save_image_gen", type = str)
    parser.add_argument("--dfu_times", type = int, default = 65)
    args = parser.parse_args()


    accelerator = Accelerator()
    os.makedirs(args.save_image_gen, exist_ok = True)
    main(args)