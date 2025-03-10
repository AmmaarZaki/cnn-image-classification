import numpy as np
import matplotlib.pyplot as plt
import streamlit as st
import torch
import torchvision.transforms as transforms
import torch.nn as nn
import torch.nn.functional as F
import torch.serialization
from PIL import Image

class NeuralNet(nn.Module):

    def __init__(self):
        super().__init__()

        self.conv1 = nn.Conv2d(3, 12, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(12, 24, 5)
        self.fc1 = nn.Linear(24 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward (self, x):
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = torch.flatten(x, 1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)

        return x

torch.serialization.add_safe_globals([NeuralNet])

def main():
    st.title('CNN Image Classifier')
    st.write('Upload an image in these classes and let the machine predicts what they are:')
    st.write('These are the classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck')
    st.write('Image file must be jpg, jpeg or png')

    image_file = st.file_uploader('Please upload an image:', type=['jpg', 'jpeg', 'png'])

    if image_file:
        image = Image.open(image_file)
        st.image(image, use_container_width=True)

        transform = transforms.Compose([
            transforms.Resize((32, 32)),
            transforms.ToTensor(),
            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        ])

        image_tensor = transform(image).unsqueeze(0)

        state_dict = torch.load('./frontend/CNN-Image-Classification-Trained.pth', map_location=torch.device('cpu'))
        model = NeuralNet()
        model.load_state_dict(state_dict)       
        model.eval()
        
        with torch.no_grad():
            output = model(image_tensor)
            probabilities = torch.nn.functional.softmax(output, dim=1)[0]
            predictions = probabilities.numpy()

        classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck']

        fig, ax = plt.subplots()
        y_position = np.arange(len(classes))

        ax.barh(y_position, output[0], align='center')
        ax.set_yticks(y_position)
        ax.set_yticklabels(classes)
        ax.invert_yaxis()

        for i, v in enumerate(output[0]):
            ax.text(v, i, f" {v:.2f}", va='center')

        ax.set_xlim(-20, 20)
        ax.grid(axis='x')
        ax.set_xlabel('Prediction')
        ax.set_title('CNN Image Classifier')

        st.pyplot(fig)

        predicted_class = classes[np.argmax(output)]
        confidence = np.max(predictions) * 100
        st.write(f"Prediction: {predicted_class} with {confidence:.1f}% confidence")

    else:
        st.text('Please upload an image.')

if __name__ == "__main__":
    main()