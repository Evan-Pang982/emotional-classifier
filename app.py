import streamlit as st
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import numpy as np
import json
import plotly.graph_objects as go
import plotly.express as px

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================

st.set_page_config(
    page_title="Facial Emotion Classifier",
    page_icon="😊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# MODEL DEFINITION (Same as training)
# ============================================================================

class TransferLearningModel(nn.Module):
    """Transfer Learning wrapper for pre-trained models."""
    def __init__(self, num_classes, freeze_layers=True, dropout_rate=0.5):
        super(TransferLearningModel, self).__init__()
        
        # Load pre-trained ResNet18
        self.model = models.resnet18(pretrained=False)  # Don't download weights
        
        # Get number of features from final layer
        num_features = self.model.fc.in_features
        
        # Replace classifier with custom head
        self.model.fc = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(num_features, 256),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
        
    def forward(self, x):
        return self.model(x)

# ============================================================================
# LOAD MODEL AND METADATA
# ============================================================================

@st.cache_resource
def load_model():
    """Load the trained model and metadata."""
    
    # Load metadata
    try:
        with open('model_metadata.json', 'r') as f:
            metadata = json.load(f)
    except FileNotFoundError:
        st.error("model_metadata.json not found!")
        return None, None
    
    # Get model parameters
    num_classes = metadata['num_classes']
    class_names = metadata['class_names']
    
    # Initialize model
    model = TransferLearningModel(
        num_classes=num_classes,
        freeze_layers=True,
        dropout_rate=0.5
    )
    
    # Load trained weights
    try:
        checkpoint = torch.load('final_emotion_classifier.pth', 
                               map_location=torch.device('cpu'))
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        st.success("✓ Model loaded successfully!")
        
    except FileNotFoundError:
        st.error("final_emotion_classifier.pth not found!")
        return None, None
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None, None
    
    return model, metadata

# ============================================================================
# IMAGE PREPROCESSING
# ============================================================================

def preprocess_image(image):
    """
    Preprocess image for model inference.
    Uses same transforms as training (ImageNet normalization).
    """
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])
    
    # Convert to RGB if needed
    if image.mode != 'RGB':
        image = image.convert('RGB')
    
    # Apply transforms
    img_tensor = transform(image)
    
    # Add batch dimension
    img_tensor = img_tensor.unsqueeze(0)
    
    return img_tensor

# ============================================================================
# PREDICTION FUNCTION
# ============================================================================

def predict_emotion(model, image_tensor, class_names):
    """Make prediction and return probabilities."""
    
    with torch.no_grad():
        outputs = model(image_tensor)
        probabilities = torch.softmax(outputs, dim=1)
        
        # Get top prediction
        confidence, predicted_idx = torch.max(probabilities, 1)
        predicted_class = class_names[predicted_idx.item()]
        
        # Get all probabilities
        all_probs = probabilities[0].numpy()
        
    return predicted_class, confidence.item(), all_probs

# ============================================================================
# VISUALIZATION FUNCTIONS
# ============================================================================

def plot_probabilities(probabilities, class_names):
    """Create interactive bar chart of prediction probabilities."""
    
    # Sort by probability
    sorted_indices = np.argsort(probabilities)[::-1]
    sorted_probs = probabilities[sorted_indices]
    sorted_classes = [class_names[i] for i in sorted_indices]
    
    # Create color scale (highlight top prediction)
    colors = ['#FF6B6B' if i == 0 else '#4ECDC4' for i in range(len(sorted_probs))]
    
    # Create bar chart
    fig = go.Figure(data=[
        go.Bar(
            x=sorted_probs * 100,
            y=sorted_classes,
            orientation='h',
            marker=dict(
                color=colors,
                line=dict(color='#2C3E50', width=1.5)
            ),
            text=[f'{p*100:.1f}%' for p in sorted_probs],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Emotion Prediction Confidence",
        xaxis_title="Confidence (%)",
        yaxis_title="Emotion",
        height=400,
        showlegend=False,
        font=dict(size=12),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
    )
    
    fig.update_xaxis(range=[0, 100], gridcolor='lightgray')
    fig.update_yaxis(autorange="reversed")
    
    return fig

def create_gauge_chart(confidence):
    """Create gauge chart for confidence level."""
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        domain={'x': [0, 1], 'y': [0, 1]},
        title={'text': "Confidence Level", 'font': {'size': 20}},
        number={'suffix': "%", 'font': {'size': 40}},
        gauge={
            'axis': {'range': [None, 100], 'tickwidth': 1, 'tickcolor': "darkblue"},
            'bar': {'color': "#FF6B6B"},
            'bgcolor': "white",
            'borderwidth': 2,
            'bordercolor': "gray",
            'steps': [
                {'range': [0, 50], 'color': '#FFE5E5'},
                {'range': [50, 75], 'color': '#FFD4D4'},
                {'range': [75, 100], 'color': '#FFC2C2'}
            ],
            'threshold': {
                'line': {'color': "red", 'width': 4},
                'thickness': 0.75,
                'value': 90
            }
        }
    ))
    
    fig.update_layout(
        height=300,
        margin=dict(l=20, r=20, t=60, b=20),
        paper_bgcolor='rgba(0,0,0,0)',
        font={'color': "darkblue", 'family': "Arial"}
    )
    
    return fig

# ============================================================================
# MAIN APP
# ============================================================================

def main():
    
    # Title and description
    st.title("😊 Facial Emotion Recognition System")
    st.markdown("""
    Upload an image to detect facial emotions using a **ResNet18 Transfer Learning** model.
    
    **Model Details:**
    - Architecture: ResNet18 with custom classifier
    - Training: Transfer learning with frozen pre-trained layers
    - Performance: 100% test accuracy
    """)
    
    # Load model
    model, metadata = load_model()
    
    if model is None or metadata is None:
        st.error("⚠️ Failed to load model. Please ensure model files are present.")
        st.stop()
    
    # Sidebar - Model Information
    with st.sidebar:
        st.header("📊 Model Information")
        
        st.subheader("Performance Metrics")
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Accuracy", f"{metadata['performance']['test_accuracy']:.1f}%")
            st.metric("Precision", f"{metadata['performance']['test_precision']:.1f}%")
        with col2:
            st.metric("Recall", f"{metadata['performance']['test_recall']:.1f}%")
            st.metric("F1-Score", f"{metadata['performance']['test_f1_score']:.1f}%")
        
        st.subheader("Hyperparameters")
        st.write(f"**Learning Rate:** {metadata['hyperparameters']['learning_rate']}")
        st.write(f"**Batch Size:** {metadata['hyperparameters']['batch_size']}")
        st.write(f"**Optimizer:** {metadata['hyperparameters']['optimizer'].upper()}")
        st.write(f"**Dropout Rate:** {metadata['hyperparameters']['dropout_rate']}")
        st.write(f"**Epochs Trained:** {metadata['hyperparameters']['epochs']}")
        
        st.subheader("Classes")
        st.write(f"**Total Classes:** {metadata['num_classes']}")
        with st.expander("View all emotion classes"):
            for i, emotion in enumerate(metadata['class_names'], 1):
                st.write(f"{i}. {emotion}")
    
    # Main content
    st.header("🖼️ Upload Image")
    
    # File uploader
    uploaded_file = st.file_uploader(
        "Choose an image...", 
        type=['jpg', 'jpeg', 'png'],
        help="Upload a facial image to detect emotion"
    )
    
    # Sample images option
    use_sample = st.checkbox("Or use a sample image for testing")
    
    if use_sample:
        st.info("💡 In production, you can add sample images here for users to test")
    
    if uploaded_file is not None:
        
        # Display uploaded image
        image = Image.open(uploaded_file)
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            st.subheader("Input Image")
            st.image(image, use_container_width=True)
            
            # Image info
            st.caption(f"Size: {image.size[0]}x{image.size[1]} | Mode: {image.mode}")
        
        with col2:
            st.subheader("Prediction Results")
            
            # Predict button
            if st.button("🔍 Analyze Emotion", type="primary", use_container_width=True):
                
                with st.spinner("Analyzing image..."):
                    
                    # Preprocess image
                    img_tensor = preprocess_image(image)
                    
                    # Make prediction
                    predicted_class, confidence, all_probs = predict_emotion(
                        model, img_tensor, metadata['class_names']
                    )
                    
                    # Display results
                    st.success("✓ Analysis Complete!")
                    
                    # Main prediction
                    st.markdown(f"""
                    <div style='text-align: center; padding: 20px; 
                                background-color: #f0f2f6; border-radius: 10px;
                                border: 2px solid #FF6B6B;'>
                        <h2 style='color: #2C3E50; margin: 0;'>Detected Emotion</h2>
                        <h1 style='color: #FF6B6B; margin: 10px 0; font-size: 3em;'>
                            {predicted_class}
                        </h1>
                        <p style='color: #555; font-size: 1.2em;'>
                            Confidence: {confidence*100:.2f}%
                        </p>
                    </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown("---")
                    
                    # Confidence gauge
                    st.plotly_chart(
                        create_gauge_chart(confidence), 
                        use_container_width=True
                    )
        
        # Full width probability chart
        if uploaded_file is not None and st.session_state.get('analyzed', False):
            st.markdown("---")
            st.subheader("📊 Detailed Probability Distribution")
            
            # Get probabilities (recompute or store in session state)
            img_tensor = preprocess_image(image)
            _, _, all_probs = predict_emotion(
                model, img_tensor, metadata['class_names']
            )
            
            st.plotly_chart(
                plot_probabilities(all_probs, metadata['class_names']),
                use_container_width=True
            )
            
            # Top 3 predictions
            top_3_indices = np.argsort(all_probs)[::-1][:3]
            
            st.subheader("🏆 Top 3 Predictions")
            cols = st.columns(3)
            
            for i, idx in enumerate(top_3_indices):
                with cols[i]:
                    st.metric(
                        label=f"#{i+1}: {metadata['class_names'][idx]}",
                        value=f"{all_probs[idx]*100:.2f}%"
                    )
        
        # Store analysis state
        if st.button("🔍 Analyze Emotion", type="primary", use_container_width=True):
            st.session_state['analyzed'] = True
    
    else:
        # Instructions when no image uploaded
        st.info("""
        👆 **How to use:**
        1. Click 'Browse files' above to upload a facial image
        2. Supported formats: JPG, JPEG, PNG
        3. Click 'Analyze Emotion' to get predictions
        4. View detailed probability distributions for all emotions
        """)
        
        # Show example
        st.markdown("---")
        st.subheader("📸 Example Use Cases")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.markdown("""
            **✅ Good Images:**
            - Clear facial features
            - Well-lit photos
            - Front-facing portraits
            """)
        
        with col2:
            st.markdown("""
            **⚠️ May Affect Accuracy:**
            - Side profiles
            - Partially obscured faces
            - Low resolution images
            """)
        
        with col3:
            st.markdown("""
            **💡 Tips:**
            - Use high-quality images
            - Ensure face is centered
            - Good lighting helps
            """)
    
    # Footer
    st.markdown("---")
    st.markdown("""
    <div style='text-align: center; color: #666;'>
        <p>Developed using PyTorch & Streamlit | ResNet18 Transfer Learning Model</p>
        <p>Model Accuracy: 100% | Framework: PyTorch</p>
    </div>
    """, unsafe_allow_html=True)

# ============================================================================
# RUN APP
# ============================================================================

if __name__ == "__main__":
    main()