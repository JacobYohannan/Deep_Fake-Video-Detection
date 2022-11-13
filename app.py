from flask import Flask, render_template, redirect, request, url_for, send_file
from flask import jsonify, json
from werkzeug.utils import secure_filename

# Interaction with the OS
import os
os.environ['KMP_DUPLICATE_LIB_OK']='True'

# Used for DL applications, computer vision related processes
import torch
import torchvision

# For image preprocessing
from torchvision import transforms

# Combines dataset & sampler to provide iterable over the dataset
from torch.utils.data import DataLoader
from torch.utils.data.dataset import Dataset

import numpy as np
import cv2

# To recognise face from extracted frames
import face_recognition
from tqdm.autonotebook import tqdm
# Autograd: PyTorch package for differentiation of all operations on Tensors
# Variable are wrappers around Tensors that allow easy automatic differentiation
from torch.autograd import Variable

import time

import sys

# 'nn' Help us in creating & training of neural network
from torch import nn
import glob
# Contains definition for models for addressing different tasks i.e. image classification, object detection e.t.c.
from torchvision import models

from skimage import img_as_ubyte
import warnings
warnings.filterwarnings("ignore")

UPLOAD_FOLDER = 'Uploaded_Files'
video_path = ""

detectOutput = []

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Creating Model Architecture

class Model(nn.Module):
  def __init__(self, num_classes, latent_dim= 2048, lstm_layers=1, hidden_dim=2048, bidirectional=False):
    super(Model, self).__init__()

    # returns a model pretrained on ImageNet dataset
    model = models.resnext50_32x4d(pretrained= True)

    # Sequential allows us to compose modules nn together
    self.model = nn.Sequential(*list(model.children())[:-2])

    # RNN to an input sequence
    self.lstm = nn.LSTM(latent_dim, hidden_dim, lstm_layers, bidirectional)

    # Activation function
    self.relu = nn.LeakyReLU()

    # Dropping out units (hidden & visible) from NN, to avoid overfitting
    self.dp = nn.Dropout(0.4)

    # A module that creates single layer feed forward network with n inputs and m outputs
    self.linear1 = nn.Linear(2048, num_classes)

    # Applies 2D average adaptive pooling over an input signal composed of several input planes
    self.avgpool = nn.AdaptiveAvgPool2d(1)



  def forward(self, x):
    batch_size, seq_length, c, h, w = x.shape

    # new view of array with same data
    x = x.view(batch_size*seq_length, c, h, w)

    fmap = self.model(x)
    x = self.avgpool(fmap)
    x = x.view(batch_size, seq_length, 2048)
    x_lstm,_ = self.lstm(x, None)
    return fmap, self.dp(self.linear1(x_lstm[:,-1,:]))




im_size = 112

# std is used in conjunction with mean to summarize continuous data
mean = [0.485, 0.456, 0.406]

# provides the measure of dispersion of image grey level intensities
std = [0.229, 0.224, 0.225]

# Often used as the last layer of a nn to produce the final output
sm = nn.Softmax()

# Normalising our dataset using mean and std
inv_normalize = transforms.Normalize(mean=-1*np.divide(mean, std), std=np.divide([1,1,1], std))

# For image manipulation
def im_convert(tensor):
  image = tensor.to("cpu").clone().detach()
  image = image.squeeze()
  image = inv_normalize(image)
  image = image.numpy()
  image = image.transpose(1,2,0)
  image = image.clip(0,1)
  cv2.imwrite('./2.png', image*255)
  return image

# For prediction of output  
def predict(model, img, path='./'):
  # use this command for gpu    
  # fmap, logits = model(img.to('cuda'))
  fmap, logits = model(img.to())
  params = list(model.parameters())
  weight_softmax = model.linear1.weight.detach().cpu().numpy()
  logits = sm(logits)
  _, prediction = torch.max(logits, 1)
  confidence = logits[:, int(prediction.item())].item()*100
  print('confidence of prediction: ', logits[:, int(prediction.item())].item()*100)
  return [int(prediction.item()), confidence]


# To validate the dataset
class validation_dataset(Dataset):
  def __init__(self, video_names, sequence_length = 60, transform=None):
    self.video_names = video_names
    self.transform = transform
    self.count = sequence_length

  # To get number of videos
  def __len__(self):
    return len(self.video_names)

  # To get number of frames
  def __getitem__(self, idx):
    video_path = self.video_names[idx]
    vidname = video_path.split("/")[1].split(".")[0]
    frames = []
    a = int(100 / self.count)
    first_frame = np.random.randint(0,a)
    count=0
    for i, frame in enumerate(self.frame_extract(video_path)):
      
      faces = face_recognition.face_locations(frame)
      
      try:
        top,right,bottom,left = faces[0]
        frame = frame[top:bottom, left:right, :]
        cv2.imwrite("frame/frame_"+vidname+ str(count)+ ".jpg", frame)
        count+=1
        print(count)
        
      except:
        pass
      frames.append(self.transform(frame))
      if(len(frames) == self.count):
        break
    frames = torch.stack(frames)
    frames = frames[:self.count]
    
    return frames.unsqueeze(0)

  # To extract number of frames
  def frame_extract(self, path):
    vidObj = cv2.VideoCapture(path) 
    success = 1
    success, image = vidObj.read()
    count=0
    while success:
      #cv2.imwrite("frame/frame_"+ str(count)+ ".jpg", image)
      # cv2.imwrite(f"Estates-Starterface/face/frame_"+str(count)+".jpg", image)

      success, image = vidObj.read()
      count += 1
      if success:
          yield image

def detectFakeVideo(videoPath,count):
    im_size = 112
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    train_transforms = transforms.Compose([
                                        transforms.ToPILImage(),
                                        transforms.Resize((im_size,im_size)),
                                        transforms.ToTensor(),
                                        transforms.Normalize(mean,std)])
    path_to_videos= [videoPath]

    video_dataset = validation_dataset(path_to_videos,sequence_length = count,transform = train_transforms)
    # use this command for gpu
    #model = Model(2).cuda()
    model = Model(2)
    path_to_model = 'model/df_model.pt'
    model.load_state_dict(torch.load(path_to_model, map_location=torch.device('cpu')))
    model.eval()
    for i in range(0,len(path_to_videos)):
        print(path_to_videos[i])
        prediction = predict(model,video_dataset[i],'./')
        if prediction[0] == 1:
            print("REAL")
        else:
            print("FAKE")
    return prediction
    



@app.route('/', methods=['POST', 'GET'])
def homepage():
  if request.method == 'GET':
	  return render_template('index.html')
  #return render_template('index.html')

@app.route('/index.html', methods=['POST', 'GET'])
def home():
  if request.method == 'GET':
	  return render_template('index.html')


@app.route('/img')
def get_image():
  if request.args.get('file') == '':
    return ''
  
  filename = request.args.get('file')

  return send_file("D:/Estates-Starter/frame/" + filename,mimetype="image/jpg")  

@app.route('/detect.html', methods=['POST', 'GET'])
def DetectPage():
    
    if request.method == 'GET':
        return render_template('detect.html')
    if request.method == 'POST':

      
        frames_folder = os.listdir('frame')
        for images in frames_folder:
          if images.endswith(".jpg"):
            os.remove(os.path.join('frame',images))

        frames_folder = os.listdir('Uploaded_Files')
        for images in frames_folder:
          if images.endswith(".mp4"):
            os.remove(os.path.join('Uploaded_Files',images))


        video = request.files['video']
        count = request.form['count']
        count= int(count)
        print(video.filename)
        

        video_filename = secure_filename(video.filename)
        video.save(os.path.join(app.config['UPLOAD_FOLDER'], video_filename))
        video_path = "Uploaded_Files/" + video_filename
        prediction = detectFakeVideo(video_path,count)

        print(prediction)

        result = "REAL"
        if prediction[0] == 0:
              result = "FAKE"
              
        confidence = prediction[1]
        confidence = round(confidence,2)
        

        frames_folder = os.listdir('frame')
       
        frames = []
        for images in frames_folder:
          if images.endswith(".jpg"):
            frames.append(images)

        #os.remove(video_path);
        return jsonify({"result" : result,"confidence" : confidence, "frames" : frames })


if __name__ == '__main__':
    app.debug=True
    app.run()