import argparse
import torch
from torchvision import transforms
from PIL import Image
import cv2
import numpy as np
from torch.jit.mobile import _load_for_lite_interpreter
from models.network_swinir import SwinIR as net



def mobile_infer():

    scale = 8 # TODO(programe this)
    window_size = 8 # TODO(program this)
    parser = argparse.ArgumentParser()
    parser.add_argument('--image', type=str, help='path to input image')
    parser.add_argument('--model', type=str, default='./lite_module.ptl', help='model used for inference')
    parser.add_argument('--out', type=str, default='./out.bmp', help='path to save output image')
    args = parser.parse_args()
    img_path = args.image
    print(f'input image path: {img_path}')


    # input_img = Image.open(img_path)
    img_lq = cv2.imread(img_path, cv2.IMREAD_COLOR).astype(
            np.float32) / 255.
    img_lq = np.transpose(img_lq if img_lq.shape[2] == 1 else img_lq[:, :, [2, 1, 0]], (2, 0, 1))
    img_lq = torch.from_numpy(img_lq).float().unsqueeze(0).to('cpu')
    print("size: ", img_lq.size())


    training_patch_size = 48
    model = net(upscale=scale, in_chans=3, img_size=training_patch_size, window_size=8,
                    img_range=1., depths=[6, 6, 6, 6, 6, 6], embed_dim=180, num_heads=[6, 6, 6, 6, 6, 6],
                    mlp_ratio=2, upsampler='pixelshuffle', resi_connection='1conv')

    with torch.no_grad():
        _, _, h_old, w_old = img_lq.size()

        # convert_tensor = transforms.ToTensor()
        # input_tensor = convert_tensor(input_img).unsqueeze(0)
        h_pad = (h_old // window_size + 1) * window_size - h_old
        w_pad = (w_old // window_size + 1) * window_size - w_old
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [2])], 2)[:, :, :h_old + h_pad, :]
        img_lq = torch.cat([img_lq, torch.flip(img_lq, [3])], 3)[:, :, :, :w_old + w_pad]
        model_path = args.model
        print(f'model path: {model_path}')
        model = _load_for_lite_interpreter(model_path)
        output = model(img_lq)
        output = output[..., :h_old * scale, :w_old * scale]






    output2 = output.data.squeeze().float().cpu().clamp_(0, 1).numpy()
    if output2.ndim == 3:
        output2 = np.transpose(output2[[2, 1, 0], :, :], (1, 2, 0))  # CHW-RGB to HCW-BGR
    # print('output size: ', output2.size())
    output2 *=  255
    output2 = output2.round().astype(np.uint8)

    output_path = args.out
    if cv2.imwrite(output_path, output2):
        print(f'output image path: {output_path}')
    else:
        raise "Failed to output image"

if __name__ == '__main__':
    mobile_infer()
