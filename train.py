import numpy as np
import torch
from mmflow.apis import init_model, inference_model
from mmflow.datasets import visualize_flow, write_flow
from models import Net,Block
from data_loader import prepare_data
from torch.autograd import Variable


'''
config_file = 'flownet2css_8x1_sfine_flyingthings3d_subset_384x768.py'
checkpoint_file = 'flownet2css_8x1_sfine_flyingthings3d_subset_384x768.pth'
'''

def saveModel(model):
   path = "./myModel.pth"
   torch.save(model.state_dict(), path)
def test(net):
   total_loss=0
   config_file = './flownet2css_8x1_sfine_flyingthings3d_subset_384x768.py'
   checkpoint_file = 'flownet2css_8x1_sfine_flyingthings3d_subset_384x768.pth'


   basedir = "./nerf_llff_data/fern"
   dataset, translations, transformations = prepare_data(basedir)

   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

   # Convert model parameters and buffers to CPU or Cuda

   pwc_model = init_model(config_file, checkpoint_file, device='cuda:0')
   for name, parameter in pwc_model.named_parameters():
      parameter.requires_grad = False
   net.to(device)
   net.eval()

   for image_pair, t1, t2 in zip(dataset, transformations, translations):
      rotation = torch.from_numpy(np.asarray(t1)).float()
      translation = torch.from_numpy(np.asarray(t2)).float()

      rotation = Variable(rotation.to(device))
      translation = Variable(translation.to(device))

      optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
      optimizer.zero_grad()

      optical_flow = inference_model(pwc_model, image_pair[0], image_pair[1]);
      optical_flow = torch.from_numpy(np.asarray([optical_flow])).float()
      optical_flow = optical_flow.permute(0, 3, 1, 2)
      optical_flow = Variable(optical_flow.to(device))
      '''
      pred_rotation
      '''
      pred_translation , pred_rotation= net(optical_flow)
      print(pred_translation,translation)
      print(pred_rotation, rotation)
      print("--")
      loss_fn = torch.nn.MSELoss(reduction='sum')

      loss1 = loss_fn(rotation, pred_rotation)

      print(translation,pred_translation)

      loss2 = loss_fn(translation, pred_translation)
      loss=loss2+loss1
      total_loss+=loss.item()
      print("loss ", loss2)
   print(total_loss/len(dataset))

def train(num_epochs, net):
   best_accuracy = 0.0
   # Specify the path to model config and checkpoint file
   config_file = './flownet2css_8x1_sfine_flyingthings3d_subset_384x768.py'
   checkpoint_file = 'flownet2css_8x1_sfine_flyingthings3d_subset_384x768.pth'


   basedir = "./nerf_llff_data/fern"
   dataset, translations, transformations = prepare_data(basedir)


   device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

   # Convert model parameters and buffers to CPU or Cuda
   '''
   net.to(device)
   net.load_state_dict(torch.load("myModel.pth"))
   '''


   pwc_model = init_model(config_file, checkpoint_file, device='cuda:0')
   for name, parameter in pwc_model.named_parameters():
      parameter.requires_grad = False
   for epoch in range(num_epochs):
      running_loss = 0.0
      best_running_loss=10000

      for image_pair, t1, t2 in zip(dataset, transformations, translations ):
         rotation=torch.from_numpy(np.asarray(t1)).float()
         translation=torch.from_numpy(np.asarray(t2)).float()

         rotation= Variable(rotation.to(device))
         translation = Variable(translation.to(device))

         optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9)
         optimizer.zero_grad()

         optical_flow=inference_model(pwc_model, image_pair[0], image_pair[1]);
         optical_flow=torch.from_numpy(np.asarray([optical_flow])).float()
         optical_flow=optical_flow.permute(0,3,1,2)
         if image_pair[0] == image_pair[1]:
            optical_flow=torch.zeros(1, 2, 378 ,504)
         optical_flow=Variable(optical_flow.to(device))



         pred_translation, pred_rotation= net(optical_flow)

         loss_fn = torch.nn.MSELoss(reduction='sum')

         loss1 = loss_fn(rotation, pred_rotation)


         loss2=  loss_fn(translation, pred_translation)

         loss=loss2+loss1
         # backpropagate the loss
         loss.backward()
         # adjust parameters based on the calculated gradients
         optimizer.step()

         running_loss += loss.item()  # extract the loss value

      print('[%d] loss: %.7f' %
            (epoch + 1, running_loss / len(dataset)))

      if running_loss < best_running_loss:
         saveModel(net)
         best_running_loss = running_loss

      # zero the loss
      running_loss = 0.0
'''
model=Net(Block)
model.load_state_dict(torch.load("myModel.pth"))
test(model)
'''

net = Net(Block)
pretrained_dict = torch.load("myModel.pth")
model_dict = net.state_dict()
# 1. filter out unnecessary keys
pretrained_dict = {k: v for k, v in pretrained_dict.items() if k in model_dict}
# 2. overwrite entries in the existing state dict
model_dict.update(pretrained_dict)
net.load_state_dict(model_dict)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
net.to(device)
train(50, net)
