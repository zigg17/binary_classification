import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import torchvision.models as models
import torch
from torch import nn
from torchvision import transforms
import certifi
import ssl
ssl._create_default_https_context = ssl._create_unverified_context


# Define the application window
root = tk.Tk()
root.title("Image Classifier")

# Assuming 'YourModelClass' is the class of your model
model = models.resnet34(weights='IMAGENET1K_V1')
model.conv1 = nn.Conv2d(1, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
num_ftrs = model.fc.in_features
num_classes = 1
model.fc = nn.Linear(num_ftrs, num_classes)
state_dict = torch.load('catdog1.pth', map_location=torch.device('cpu'))  # Load the state dictionary
model.load_state_dict(state_dict)  # Load the state dictionary into the model
model.eval()  # Set the model to evaluation mode


# Define the image transformation
transform = transforms.Compose([
    transforms.Resize((224, 224)),  # Replace 'size' with the size used in training
    transforms.ToTensor(),
    transforms.Grayscale(num_output_channels=1)
    # Add any other transforms you used during training
])

def open_image():
    file_path = filedialog.askopenfilename()
    if file_path:
        img_original = Image.open(file_path)
        img_original = img_original.convert('RGB')

        # Resize the image for display purposes
        display_size = (200, 200)  # Set a display size that fits your application
        img_display = img_original.resize(display_size)

        # Display the resized image
        tk_img = ImageTk.PhotoImage(img_display)
        image_label.config(image=tk_img)
        image_label.image = tk_img

        # Process the original image for prediction
        img_transformed = transform(img_original)
        img_transformed = img_transformed.unsqueeze(0)

        with torch.inference_mode():
            prediction = torch.round(torch.sigmoid(model(img_transformed).squeeze()))
        
        if prediction == 1:
            prediction = 'dog'
        else:
            prediction = 'cat'

        # Display the prediction
        prediction_label.config(text=f'Prediction: {prediction}')


# Create a button to open the image
open_button = tk.Button(root, text="Open Image", command=open_image)
open_button.pack()

# Label to display the selected image (optional)
image_label = tk.Label(root)
image_label.pack()

# Label to display the prediction
prediction_label = tk.Label(root, text="Prediction: None")
prediction_label.pack()

# Start the application
root.mainloop()

