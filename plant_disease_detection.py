import os
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.models as models
from torchvision.datasets import ImageFolder
from torch.utils.data import DataLoader, random_split
import gradio as gr
from PIL import Image
from torch.optim.lr_scheduler import ReduceLROnPlateau
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Disease information database

DISEASE_INFO = {
    "Apple___Apple_scab": {
        "disease_name": "Apple Scab",
        "description": "A fungal disease that causes dark, scaly lesions on leaves and fruit.",
        "symptoms": [
            "Dark olive-green spots on leaves",
            "Dark, scaly lesions on fruit",
            "Twisted or distorted leaves",
            "Premature leaf drop"
        ],
        "treatment": [
            "Apply appropriate fungicides early in the growing season.",
            "Remove and dispose of infected plant material (leaves, fruits, and stems).",
            "Improve air circulation by pruning nearby vegetation.",
            "Practice crop rotation to prevent fungal spore buildup."
        ],
        "prevention": [
            "Space plants adequately for good air circulation.",
            "Sanitize tools and equipment after use.",
            "Apply preventive fungicides during early growth stages.",
            "Maintain proper soil drainage."
        ]
    },
    "Apple___Black_rot": {
        "disease_name": "Black Rot",
        "description": "A fungal disease affecting apples, causing rotting of fruit and leaf spots.",
        "symptoms": [
            "Purple spots on leaves.",
            "Rotting fruit with dark rings.",
            "Cankers on branches.",
            "Premature leaf drop."
        ],
        "treatment": [
            "Remove and destroy infected plant material, including leaves, branches, and fruits.",
            "Apply fungicides during early stages of infection.",
            "Prune out dead or diseased wood to prevent spread.",
            "Practice crop rotation and proper field sanitation."
        ],
        "prevention": [
            "Prune plants during dry weather to prevent fungal spread.",
            "Sanitize pruning tools after every use.",
            "Remove any mummified fruit or debris around the plant.",
            "Maintain optimal plant spacing to allow air circulation."
        ]
    },
    "Corn___Common_rust": {
        "disease_name": "Common Rust",
        "description": "A fungal disease causing rust-like pustules on corn leaves.",
        "symptoms": [
            "Rust-colored pustules on leaves.",
            "Yellowing and drying of infected areas.",
            "Reduced plant vigor.",
            "Yield reduction in severe cases."
        ],
        "treatment": [
            "Apply fungicides effective against rust fungi early in the infection.",
            "Remove and destroy infected plant material immediately.",
            "Rotate crops to reduce fungal spore buildup in the soil.",
            "Plant disease-resistant corn varieties."
        ],
        "prevention": [
            "Monitor crops regularly during the growing season.",
            "Plant rust-resistant corn varieties.",
            "Avoid overhead irrigation to reduce leaf wetness.",
            "Practice crop rotation to minimize fungal spores."
        ]
    },
    "Corn___healthy": {
        "disease_name": "Healthy Corn Plant",
        "description": "The plant shows no signs of disease.",
        "symptoms": ["No visible symptoms of disease."],
        "treatment": ["No treatment needed."],
        "prevention": [
            "Monitor plants regularly to catch early signs of disease.",
            "Maintain proper irrigation schedules.",
            "Use disease-free seeds for planting.",
            "Practice good field sanitation and crop rotation."
        ]
    },
    "Apple___healthy": {
        "disease_name": "Healthy Apple Plant",
        "description": "The plant shows no signs of disease.",
        "symptoms": ["No visible symptoms of disease."],
        "treatment": ["No treatment needed."],
        "prevention": [
            "Monitor plants regularly to catch early signs of disease.",
            "Prune and maintain trees regularly.",
            "Use high-quality disease-free seedlings.",
            "Apply preventive fungicides before rainy seasons."
        ]
    }
}


class PlantDiseaseModel(nn.Module):
    def __init__(self, num_classes):
        super(PlantDiseaseModel, self).__init__()
        self.model = models.efficientnet_b0(weights=models.EfficientNet_B0_Weights.IMAGENET1K_V1)
        n_features = self.model.classifier[1].in_features
        self.model.classifier[1] = nn.Linear(n_features, num_classes)

    def forward(self, x):
        return self.model(x)


class Config:
    data_dir = "dataset/PlantVillage"
    batch_size = 20
    num_epochs = 8
    learning_rate = 0.001
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    image_size = 224
    train_split = 0.8
    val_split = 0.1
    test_split = 0.1
    model_path = 'plant_disease_model.pth'


def get_transforms():
    train_transform = transforms.Compose([
        transforms.Resize((Config.image_size, Config.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(20),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    val_transform = transforms.Compose([
        transforms.Resize((Config.image_size, Config.image_size)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
    ])

    return train_transform, val_transform


def train_model():
    """Train and save the plant disease model."""
    train_transform, val_transform = get_transforms()
    dataset = ImageFolder(Config.data_dir, transform=train_transform)
    num_classes = len(dataset.classes)

    train_size = int(Config.train_split * len(dataset))
    val_size = int(Config.val_split * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = random_split(
        dataset, [train_size, val_size, test_size]
    )
    train_loader = DataLoader(train_dataset, batch_size=Config.batch_size, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=Config.batch_size, shuffle=False)

    model = PlantDiseaseModel(num_classes=num_classes).to(Config.device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=Config.learning_rate)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', patience=2, verbose=True)

    for epoch in range(Config.num_epochs):
        model.train()
        total_loss = 0.0

        for inputs, labels in train_loader:
            inputs, labels = inputs.to(Config.device), labels.to(Config.device)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        val_loss = 0.0
        model.eval()
        with torch.no_grad():
            for inputs, labels in val_loader:
                inputs, labels = inputs.to(Config.device), labels.to(Config.device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                val_loss += loss.item()

        logger.info(f"Epoch {epoch + 1}/{Config.num_epochs}, Loss: {total_loss:.4f}, Val Loss: {val_loss:.4f}")
        scheduler.step(val_loss)

    logger.info("Training complete. Saving model...")
    torch.save({
        'model_state_dict': model.state_dict(),
        'class_names': dataset.classes
    }, Config.model_path)
    logger.info(f"Model saved to {Config.model_path}")


def predict_disease(image):
    try:
        # Load the trained model and class names
        checkpoint = torch.load(Config.model_path, map_location=Config.device)
        class_names = checkpoint['class_names']

        model = PlantDiseaseModel(num_classes=len(class_names))
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(Config.device)
        model.eval()

        # Preprocess the input image
        transform = get_transforms()[1]
        image = image.convert("RGB")
        image_tensor = transform(image).unsqueeze(0).to(Config.device)

        # Make predictions
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        predicted_class = class_names[predicted.item()]
        confidence_value = confidence.item() * 100

        # Guess the fruit name
        fruit_name = predicted_class.split("___")[0]

        # Fetch disease information
        disease_info = DISEASE_INFO.get(predicted_class, None)
        if disease_info:
            disease_name = disease_info["disease_name"]
            description = disease_info["description"]
            symptoms = "\n".join(disease_info["symptoms"])
            treatment = "\n".join(disease_info["treatment"])
            prevention = "\n".join(disease_info["prevention"])

            response = (
                f"Prediction: {disease_name} (Confidence: {confidence_value:.2f}%)\n"
                f"Fruit: {fruit_name}\n\n"
                f"Description: {description}\n\n"
                f"Symptoms:\n{symptoms}\n\n"
                f"Treatment:\n{treatment}\n\n"
                f"Prevention:\n{prevention}"
            )
        else:
            response = (
                f"Prediction: {predicted_class} (Confidence: {confidence_value:.2f}%)\n"
                f"Fruit: {fruit_name}\n\n"
                f"Information not available for this disease."
            )

        return response

    except Exception as e:
        logger.error(f"Prediction error: {str(e)}")
        return f"Error during prediction: {str(e)}"


def create_gradio_interface():
    try:
        interface = gr.Interface(
            fn=predict_disease,
            inputs=gr.Image(type="pil", label="Upload Plant Image"),
            outputs="text",
            title="Plant Disease Diagnostic Tool",
            description="Upload an image of a plant leaf to detect diseases."
        )
        logger.info("Gradio interface created successfully.")
        return interface
    except Exception as e:
        logger.error(f"Error creating Gradio interface: {str(e)}")
        raise


def main():
    try:
        if not os.path.exists(Config.model_path):
            logger.info("Training new model...")
            train_model()

        logger.info("Starting Gradio interface...")
        interface = create_gradio_interface()
        interface.launch(server_name="0.0.0.0", server_port=7860, share=True, debug=True)
    except Exception as e:
        logger.error(f"Application error: {str(e)}")
        raise


if __name__ == "__main__":
    main()
