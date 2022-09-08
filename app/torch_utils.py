import io
import torch 
import torch.nn as nn 
import torchvision.transforms as transforms 
from PIL import Image

model = torch.load('model.pt',map_location=torch.device('cpu'))

def transform_image(image_bytes):
    transform = transforms.Compose([transforms.Resize([512,512]),
                                      transforms.ToTensor(),
                                      transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])

    image = Image.open(io.BytesIO(image_bytes))
    return transform(image).unsqueeze(0)

def get_prediction(image):
    image = image.expand((1,3,512,512))
    model.eval()
    output = model(image)
    predictions = output.argmax(dim=1).cpu().detach().tolist()
    #print('Predicted Class: ' + str(predictions[0]))
    return predictions[0]