import argparse

from PIL import Image
import requests

import torch
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

from models.network_swin2sr import Swin2SR as net


def main(model_name):
    url = 'https://github.com/mv-lab/swin2sr/blob/main/testsets/real-inputs/shanghai.jpg?raw=true'
    image = Image.open(requests.get(url, stream=True).raw)

    transforms = Compose([
        Resize((256, 256)),
        ToTensor(),
        Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])

    pixel_values = transforms(image).unsqueeze(0)

    # inference: 
    if model_name == "Swin2SR_ClassicalSR_X2_64":
        model = net(upscale=2, in_chans=3, img_size=64, window_size=8,
                            img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                            mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = "params"               
    elif model_name == "Swin2SR_ClassicalSR_X4_64":
        model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                            img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                            mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
        param_key_g = "params"               
    elif model_name in ["Swin2SR_CompressedSR_X4_48", "Swin2SR_CompressedSR_X4_DIV2K_Test", "Swin2SR_CompressedSR_X4_DIV2K_Valid"]:
        # scale = 4, img_size = 48
        model = net(upscale=4, in_chans=3, img_size=48, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle_aux', resi_connection='1conv')
        param_key_g = 'params'   
    elif model_name == "Swin2SR_Jpeg_dynamic":
        model = net(upscale=1, in_chans=1, img_size=126, window_size=7,
                    img_range=255., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='', resi_connection='1conv')
        param_key_g = 'params'
    elif model_name == "Swin2SR_Lightweight_X2_64":
        # scale = 2, img_size = 64
        model = net(upscale=2, in_chans=3, img_size=64, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6], embed_dim=60, num_heads=[6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffledirect', resi_connection='1conv')
        param_key_g = 'params'   
    elif model_name in ["Swin2SR_RealworldSR_X4_64_BSRGAN_PSNR", "Swin2SR_RealworldSR_X4_RealSRSet"]:
        # scale = 4
        model = net(upscale=4, in_chans=3, img_size=64, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='nearest+conv', resi_connection='1conv')
        param_key_g = 'params_ema'

    # load weights
    model.eval()
    state_dict = f"/content/drive/MyDrive/Swin2SR/Original checkpoints/{model_name}.pth"
    pretrained_model = torch.load(state_dict, map_location="cpu")
    model.load_state_dict(pretrained_model[param_key_g] if param_key_g in pretrained_model.keys() else pretrained_model, strict=True)
    
    # forward pass
    with torch.no_grad():
        output = model(pixel_values)

    print("Shape of output:", output.shape)
    print("First values of output:", output[0, 0, :3, :3])


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model_name",
        default="Swin2SR_ClassicalSR_X2_64",
        type=str,
        help="Name of the model on which you'd like to do a forward pass.",
    )
    args = parser.parse_args()
    main(args.model_name)