import streamlit as st
import torch
from torchvision.datasets import ImageFolder
from torchvision.transforms import transforms
from PIL import Image
from src.models import get_resnet18_pretrained
import pandas as pd 


MODEL_PATH = r"C:\Users\Lenovo\Desktop\AI_Project\checkpoints\resnet18_pretrained_sgd.pth"
TEST_DATA_PATH = r"C:\Users\Lenovo\Desktop\AI_Project\dataset\TEST"  # Test dataset path
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#labels = pd.read_csv(r'C:\Users\Lenovo\Desktop\AI_Project\dataset\labels.csv')

# Get ImageFolder-based class mappings
@st.cache_data
def get_imagefolder_mappings(dataset_path):
    """
    Retrieve class-to-index mapping used by ImageFolder.
    """
    dataset = ImageFolder(root=dataset_path)
    #print(dataset.class_to_idx.items())
    return {v: k for k, v in dataset.class_to_idx.items()}  # Reverse mapping (index -> class name)


@st.cache_resource
def load_model(model_path, num_classes):
    """
    Load the trained model.
    """
    model = get_resnet18_pretrained(num_classes=num_classes)
    
    # Load checkpoint
    checkpoint = torch.load(model_path, map_location=DEVICE)

    # Extract model weights from the checkpoint
    model.load_state_dict(checkpoint['state_dict'])  # Load the model weights
    
    model.to(DEVICE)
    model.eval()  # Set the model to evaluation mode
    
    return model


# Preprocess uploaded images
def preprocess_image(image):
    """
    Preprocess the uploaded image for model inference.
    """
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5], std=[0.5])  # Match training normalization
    ])
    return transform(image).unsqueeze(0)  # Add batch dimension


# Predict the class of an image
def predict(model, image_tensor, class_mappings):
    """
    Predict the class label for the given image tensor.
    """
    with torch.no_grad():
        image_tensor = image_tensor.to(DEVICE)
        outputs = model(image_tensor)
        _, predicted_label = outputs.max(1)  # Get class index with the highest score
    return class_mappings[predicted_label.item()]


# Streamlit UI
st.title("Traffic Sign Classification App")
st.write("Upload an image, and the model will predict its traffic sign class.")

# Get class mappings and load the model
class_mappings = get_imagefolder_mappings(TEST_DATA_PATH)
model = load_model(MODEL_PATH, len(class_mappings))


uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    try:
        
        image = Image.open(uploaded_file).convert("RGB")
        image_tensor = preprocess_image(image)

        
        st.image(image, caption="Uploaded Image", width=200)  
        #st.write("Classifying...")

       
        prediction = predict(model, image_tensor, class_mappings)
        #label = labels.loc[labels['ClassId'] == int(prediction), 'Name'].values[0]
        
        st.write(f"**Prediction:** {prediction}")

    except Exception as e:
        st.error(f"An error occurred: {e}")
