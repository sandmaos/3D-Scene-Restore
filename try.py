import torchvision
from PIL import Image
from mmflow.apis import init_model, inference_model
from mmflow.datasets import visualize_flow, write_flow
import mmcv

# Specify the path to model config and checkpoint file
config_file = './flownet2css_8x1_sfine_flyingthings3d_subset_384x768.py'
checkpoint_file = 'flownet2css_8x1_sfine_flyingthings3d_subset_384x768.pth'

# build the model from a config file and a checkpoint file
model = init_model(config_file, checkpoint_file, device='cuda:0')
img1='./demo/image000.png'
image = Image.open(img1)
w1=torchvision.transforms.functional.affine(image,translate=(10,5),angle=0,scale=1,shear=0)
w1.save("./new_image.png")
# test image pair, and save the results

img2='./new_image.png'
result = inference_model(model, img1, img2)
print(result[0][0][0])
print(result[0][0][1])
# save the optical flow file
write_flow(result, flow_file='flow.flo')
# save the visualized flow map
flow_map = visualize_flow(result, save_file='flow_map.png')