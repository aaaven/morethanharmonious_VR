from diffusers import StableDiffusionXLInpaintPipeline
import torch
from PIL import Image
import torch
import numpy as np
import time

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(device)
torch.set_grad_enabled(False)
if torch.cuda.is_available():
    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True

pipe = StableDiffusionXLInpaintPipeline.from_pretrained("sdxl-turbo")
pipe = pipe.to(device)

prompts = [
    "Landfill. Pollution is the introduction of harmful materials into the environment. Landfills collect garbage and other land pollution in a central location. Many places are running out of space for landfills",
    "City Light Pollution. Boats, buildings, street lights, and even fireworks contribute to the light pollution in Victoria Harbor, Hong Kong. Light pollution can be detrimental to the health of people and animals in the area",
    "Human induced oil spills devastating wildlife and resulting in animals becoming sick or dying ",
    "Human induced disasters, such as oil spills, can be devastating for all forms of wildlife. Often times resulting in animals becoming sick or dying.",
    "Wildfires scorch the land in Malibu Creek State Park. As the wind picks up, the fire begins to spread faster",
    "The tallest towers of Shanghai, China, rise above the haze. Shanghai's smog is a mixture of pollution from coal, the primary source of energy for most homes and businesses in the region, as well as emissions from vehicles"
]

square_ground = Image.open('../../test/image/background.png')
mask_img = Image.open('../../test/image/mask.jpg')
x, y = 268, 516
strength = 0.75
guidance_scale = 1.5
num_inference_steps = 6

import random
for i in range(20):
    time0 = time.time()
    img2img_result = pipe(prompt=prompts[random.randint(0, 5)], image=square_ground, mask_image=mask_img, num_inference_steps=num_inference_steps, strength=strength, guidance_scale=guidance_scale, output_type='pt', return_dict=False)
    torch.cuda.synchronize()
    print("cost: ", time.time() - time0)
    torch.cuda.empty_cache()
Image.fromarray((img2img_result[0][0] * 255).cpu().numpy().transpose(1, 2, 0).astype(np.uint8)).save('../../test/image/res.png')