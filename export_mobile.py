import torch
from models.network_swinir import SwinIR as net
from PIL import Image
from torchvision import transforms
from torch.utils.mobile_optimizer import optimize_for_mobile

if __name__ == '__main__':
    up_scale=2
    training_patch_size = 48
    model = net(upscale=up_scale, in_chans=3, img_size=training_patch_size, window_size=8,
                        img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                        mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')
    example_img = Image.open("testsets/Set5/LR_bicubic/X2/babyx2.png")
    convert_tensor = transforms.ToTensor()
    example_input = convert_tensor(example_img)
    example_input = example_input.unsqueeze(0)
    model.eval()
    scripted_module = torch.jit.trace(model, example_input)
    torch.jit.save(scripted_module, "jit_saved_module.pt")
    opt_model = optimize_for_mobile(scripted_module)
    opt_model._save_for_lite_interpreter("lite_module.ptl")
