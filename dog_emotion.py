import streamlit as st
import pandas as pd
import numpy as np
from PIL import Image
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
import requests
import json 
from streamlit_lottie import st_lottie

st.set_page_config(layout="wide")

# Lottie 애니메이션 로드 함수
def load_lottie_from_file(filepath: str):
    with open(filepath, "r") as f:
        return json.load(f)

# 로컬 JSON 파일 경로
json_file_path = "dog.json"

# JSON 파일 로드
lottie_animation = load_lottie_from_file(json_file_path)

# HTML 삽입
st.components.v1.html(f"""
    <div class="lottie-container" style="width: 100%; height: 300px; position: relative; overflow: hidden;">
        <div id="lottie-animation" class="lottie-animation" style="position: absolute; top: 50%; left: -300px; transform: translateY(-50%); animation: moveLottie 10s linear infinite; width: 300px; height: 300px;"></div>
    </div>
    <style>
    @keyframes moveLottie {{
        0% {{ left: -300px; }}
        100% {{ left: 100%; }}
    }}
    </style>
    <script src="https://cdnjs.cloudflare.com/ajax/libs/lottie-web/5.7.4/lottie.min.js"></script>
    <script>
    var animationData = {json.dumps(lottie_animation)};
    var animation = lottie.loadAnimation({{
        container: document.getElementById('lottie-animation'),
        renderer: 'svg',
        loop: true,
        autoplay: true,
        animationData: animationData
    }});
    </script>
""", height=300)

st.markdown(
    """
    <style>

        /* 눈송이 애니메이션 */
        @keyframes snow {
            0% { transform: translateY(0); opacity: 1; }
            100% { transform: translateY(100vh); opacity: 0; }
        }
        .snowflake {
            position: fixed;
            top: -10px;
            font-size: 2rem;
            color: white;
            text-shadow: 0 0 10px #FFF, 0 0 20px #AAF;
            animation: snow 10s linear infinite;
        }

        /* 각 눈송이에 랜덤 위치와 속도 설정 */
        .snowflake:nth-child(1) { left: 10%; animation-duration: 15s; }
        .snowflake:nth-child(2) { left: 20%; animation-duration: 10s; }
        .snowflake:nth-child(3) { left: 30%; animation-duration: 12s; }
        .snowflake:nth-child(4) { left: 40%; animation-duration: 14s; }
        .snowflake:nth-child(5) { left: 50%; animation-duration: 9s; }
        .snowflake:nth-child(6) { left: 60%; animation-duration: 13s; }
        .snowflake:nth-child(7) { left: 70%; animation-duration: 11s; }
        .snowflake:nth-child(8) { left: 80%; animation-duration: 16s; }
        .snowflake:nth-child(9) { left: 90%; animation-duration: 8s; }

        .stTitle {
            font-size: 4rem;
            color: #FF0000;
            text-shadow: 3px 3px 10px #FFFFFF, 0 0 20px #FFD700;
            animation: glow 2s infinite alternate;
        }

        @keyframes glow {
            from { text-shadow: 3px 3px 10px #FFFFFF, 0 0 20px #FFD700; }
            to { text-shadow: 3px 3px 20px #FF4500, 0 0 40px #FF6347; }
        }
        .stFileUploader {
            background: linear-gradient(135deg, #FF0000, #FFD700);
            border: 3px solid #FFFFFF;
            padding: 20px;
            border-radius: 15px;
            box-shadow: 0px 0px 20px #FFFFFF;
            font-size: 1.2rem;
            color: white;
            text-align: center;
        }

        /* 눈송이를 추가 */
        .snowflake-container {
            position: fixed;
            width: 100%;
            height: 100%;
            pointer-events: none;
            z-index: 1000;
        }
    </style>

    <div class="snowflake-container">
        <div class="snowflake">❄️</div>
        <div class="snowflake">☃️</div>
        <div class="snowflake">🎄</div>
        <div class="snowflake">🎅</div>
        <div class="snowflake">❄️</div>
        <div class="snowflake">☃️</div>
        <div class="snowflake">🎄</div>
        <div class="snowflake">🎅</div>
        <div class="snowflake">❄️</div>
    </div>
    """,
    unsafe_allow_html=True
)

st.markdown(
    """
    <h1 style="text-align: center; font-size: 4rem; color: #FFFFFF; text-shadow: 3px 3px 10px #FF0000, 0 0 20px #FFD700;">너의 강아지의 기분은?</h1>
    """,
    unsafe_allow_html=True
)   

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
    prob = torch.sigmoid(output)  
    predicted = (prob > 0.5).long()
    prob=round(prob.item(),3)
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
 

  for i in model_list:
    pred, prob=cls_result(i,img_path,test_transform)
    pred_list.append(pred)
    prob_list.append(prob)

  weighted_prob = np.average(prob_list, axis=0)
  prob1 = weighted_prob
  prob0 = 1 - weighted_prob

  if prob1 > prob0:
    prediction = 'not happy'
    probability = round(prob1 * 100,2)
  else:
    prediction = 'happy'
    probability = round(prob0 * 100,2)

  if result=='Dog':
    st.markdown(
              f"""
              <div style="padding: 20px; border: 3px solid #FFFFFF; background: linear-gradient(135deg, #FF0000, #FFD700); color: white; border-radius: 15px; box-shadow: 0px 0px 20px #FFFFFF; font-size: 1.2rem; text-align: center;">
              <h2 style="text-align: center;">🎁 예측 결과: <span style='color: #FFFFFF;'>{prediction}</span></h2>
              <p style="text-align: center;">확률: <b>{probability}%</b></p>
              </div>
              """,
              unsafe_allow_html=True
    )

  else:
    st.markdown(
            f"""
            <div style="padding: 20px; border: 3px solid #FFFFFF; background: linear-gradient(135deg, #FF0000, #FFD700); color: white; border-radius: 15px; box-shadow: 0px 0px 20px #FFFFFF; font-size: 1.2rem; text-align: center;">
            <h2 style="text-align: center;">🎁 예측 결과: <span style='color: #FFFFFF;'>{prediction}</span></h2>
            <p style="text-align: center;">확률: <b>{probability}%</b></p>
            <p style="text-align: center;">이미지에 강아지가 없는 것 같아요: <b></p>
            <p style="text-align: center;">강아지 다운 사진을 다시 업로드해 주세요!: <b></p>
            </div>
            """,
            unsafe_allow_html=True
    )
  
