from PIL import Image
import requests

import torch
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from models.network_swin2sr import Swin2SR as net


def main():
    url = 'https://github.com/mv-lab/swin2sr/blob/main/testsets/real-inputs/shanghai.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw)

    transforms = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pixel_values = transforms(image).unsqueeze(0)

    # inference
    model = net(upscale=2, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    param_key_g = "params"               
    model.eval()
    pretrained_model = torch.load("/content/drive/MyDrive/Swin2SR/Original checkpoints/Swin2SR_ClassicalSR_X2_64.pth", map_location="cpu")
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)

    with torch.no_grad():
        output = model(pixel_values)

    print("Shape of output:", output.shape)
    print("First values of output:", output[0, 0, :3, :3])


if __name__ == '__main__':
    main()