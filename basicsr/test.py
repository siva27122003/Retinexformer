import logging
import torch
from os import path as osp
from PIL import Image
import numpy as np

from basicsr.models import create_model
from basicsr.train import parse_options
from basicsr.utils import (get_env_info, get_root_logger, get_time_str,
                           make_exp_dirs, tensor2img)


def load_image(image_path):
    """Load a single image as a tensor."""
    img = Image.open(image_path).convert('RGB')
    img = np.array(img).astype(np.float32) / 255.0  # Normalize to [0, 1]
    img = torch.from_numpy(np.transpose(img, (2, 0, 1))).unsqueeze(0)  # CHW format
    return img


def save_image(tensor, save_path):
    """Save a tensor as a PNG image."""
    img_np = tensor2img(tensor)  # Convert tensor to image format
    Image.fromarray(img_np).save(save_path)


def main():
    # Parse options, set distributed setting, set random seed
    opt = parse_options(is_train=False)

    torch.backends.cudnn.benchmark = True

    # Make directories and initialize loggers
    make_exp_dirs(opt)
    log_file = osp.join(opt['path']['log'],
                        f"test_{opt['name']}_{get_time_str()}.log")
    logger = get_root_logger(
        logger_name='basicsr', log_level=logging.INFO, log_file=log_file)
    logger.info(get_env_info())
    logger.info(dict2str(opt))

    # Load a single PNG image
    image_path = '/content/Retinexformer/basicsr/data/meta_info/lowtree.jpg'  # Replace with your PNG image path
    img = load_image(image_path)

    # Create model
    model = create_model(opt)

    # Run inference
    model.feed_data(data={'lq': img})  # Assuming the input key is 'lq'
    model.test()

    # Get the output image
    output_img = model.get_current_visuals()['result']  # Assuming 'result' is the output key

    # Save the output image
    save_image(output_img, 'output/result.png')  # Specify the output path
    logger.info(f"Image saved at 'output/result.png'")


if __name__ == '__main__':
    main()
