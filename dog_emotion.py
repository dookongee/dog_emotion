import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
#from streamlit_drawable_canvas import st_canvas
#from ultralytics import YOLO
from torchvision import models
from torchvision.models import ResNet34_Weights
from torchvision.models import EfficientNet_B0_Weights
from torchvision.models import DenseNet121_Weights
import torch.nn as nn
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from PIL import Image
import albumentations as A
from albumentations.pytorch import ToTensorV2
import cv2
import os

st.title("Dog_Prediction")   

@st.cache_resource(hash_funcs={torch.nn.Module: lambda _: None})
def densenet(weight, device):
  model=models.densenet121(weights=DenseNet121_Weights.DEFAULT)
  for param in model.parameters():
    param.requires_grad=False
  for param in model.features.denseblock4.parameters():
    param.requires_grad=True
  model.classifier = nn.Linear(model.classifier.in_features,1)
  for param in model.classifier.parameters():
    param.requires_grad=True
  checkpoint = torch.load(weight,map_location=device,weights_only=True)
  model.load_state_dict(checkpoint['model_state_dict'])
  model.to(device)
  model.eval()
  return model

@st.cache_resource(hash_funcs={torch.nn.Module: lambda _: None})
def efficientnet(weight, device):
  model=models.efficientnet_b0(weights=EfficientNet_B0_Weights.DEFAULT)
  for param in model.parameters():
    param.requires_grad = False
  for param in model.features[5:].parameters():
    param.requires_grad = True
  model.classifier[1] = nn.Linear(model.classifier[1].in_features, 1)
  for param in model.classifier.parameters():
    param.requires_grad = True
  checkpoint = torch.load(weight,map_location=device,weights_only=True)
  model.load_state_dict(checkpoint['model_state_dict'])
  model.to(device)
  model.eval()
  return model

@st.cache_resource(hash_funcs={torch.nn.Module: lambda _: None})
def resnet(weight, device):
  model = models.resnet34(weights=ResNet34_Weights.DEFAULT)
  for param in model.parameters():
    param.requires_grad = False
  for param in model.layer4.parameters():
    param.requires_grad = True
  model.fc = nn.Linear(model.fc.in_features, 1)
  for param in model.fc.parameters():
    param.requires_grad = True
  checkpoint = torch.load(weight,map_location=device,weights_only=True)
  model.load_state_dict(checkpoint['model_state_dict'])
  model.to(device)
  model.eval()
  return model

@st.cache_resource(hash_funcs={torch.nn.Module: lambda _: None})
def resnet34(device):
  model_res=models.resnet34(weights=ResNet34_Weights.DEFAULT)
  model_res.to(device)
  model_res.eval()
  return model_res

def cls_result(model, img_path, transform):
  img = np.array(img_path)
  augmented=transform(image=img)
  img=augmented['image']

  with torch.no_grad():
    image = img.unsqueeze(0).to(device)
    output = model(image)
    prob = torch.sigmoid(output)  # Sigmoid 활성화 함수 적용
    predicted = (prob > 0.5).long()
    prob=round(prob.item(),3)*100
    if prob<50:prob=100-prob
  return predicted.item(), prob

def predict_dog(image_path, model, transform, dog_class_id):
  img = np.array(img_path)
  augmented=transform(image=img)
  img=augmented['image']

  with torch.no_grad():
    image = img.unsqueeze(0).to(device)
    outputs = model(image)

  probabilities = torch.softmax(outputs, dim=1)
  dog_prob = probabilities[:, dog_class_ids].sum()

  non_dog_prob = 1 - dog_prob
  return "Dog" if dog_prob > non_dog_prob else "Not Dog"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

de_ada=densenet('de_ada_b_epoch_7.pth',device)
de_ada_re=densenet('de_ada_re_b_epoch_8.pth',device)
de_sgd=densenet('de_sgd_b.pth',device)
eff_ada=efficientnet('eff_ada_b_epoch_5.pth',device)
eff_ada_re=efficientnet('eff_ada_re_b_epoch_6.pth',device)
res_ada_re=resnet('res34_ada_re_b_epoch_8.pth',device)
res_sgd=resnet('res34_sgd_b_epoch_8.pth',device)
resnet34=resnet34(device)

model_list=[de_ada,de_ada_re,de_sgd,eff_ada,eff_ada_re,res_ada_re,res_sgd]

test_transform = A.Compose([
  A.Resize(224, 224),
  A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225), max_pixel_value=255.0),
  ToTensorV2()
])
pred_list = []
prob_list = []

uploaded_file = st.file_uploader("강아지 이미지를 업로드하세요", type=['jpg', 'jpeg', 'png'])

dog_class_ids = [i for i in range(151, 276)] 

if uploaded_file is not None:
  image = Image.open(uploaded_file).convert('RGB')
  img_path=image
  st.image(img_path, caption='업로드한 이미지')

  result=predict_dog(img_path, resnet34, test_transform, dog_class_ids)
  if result=='Dog':

    for i in model_list:
      pred, prob=cls_result(i,img_path,test_transform)
      pred_list.append(pred)
      prob_list.append(prob)

    if pred_list.count(0)>pred_list.count(1):
      prediction='happy'
      probability=round(sum([prob_list[i] for i, value in enumerate(pred_list) if value==0])/pred_list.count(0),2)
    else:
      prediction='not_happy'
      probability=round(sum([prob_list[i] for i, value in enumerate(pred_list) if value==1])/pred_list.count(1),2)

    st.write(prediction, probability)
    st.write(pred_list)
    st.write(prob_list)
  else:
    st.write("강아지를 검출할 수 없습니다.")
    st.write("다른 강아지 사진을 업로드해 주세요!")