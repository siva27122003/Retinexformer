from PIL import Image
import torch
from torchvision import transforms
import basicsr.models.create_model as create_model

def single_image_inference(opt, image_path):
    # Load the image
    img = Image.open(image_path)
    
    # Preprocess the image
    transform = transforms.Compose([
        transforms.Resize((opt['datasets']['train']['gt_size'], opt['datasets']['train']['gt_size'])),
        transforms.ToTensor()
    ])
    img_tensor = transform(img).unsqueeze(0)  # Add batch dimension

    # Load the model
    model = create_model(opt)
    model.feed_test_data({'lq': img_tensor})

    # Run the model for a single forward pass
    model.test()

    # Get the output
    output = model.get_current_visuals()['result']
    output_image = transforms.ToPILImage()(output.squeeze(0))  # Remove batch dimension

    return output_image
