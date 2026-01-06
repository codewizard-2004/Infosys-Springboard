import streamlit as st
from PIL import Image
import pandas as pd
import plotly.express as px
from config import settings
from utils.preprocessing import preprocess
from utils.inference import run_inference
from utils.nutrients import filter_csv_by_label
from utils.youtube_service import get_cooking_videos
from utils.groq_analysis import get_food_description
import onnxruntime as ort

def main():
    st.set_page_config(page_title="FoodNet", page_icon="🍕", layout="wide")
    
    st.title("FoodNet 🍕🥩🍣")
    st.markdown("Upload an image to classify it as **Pizza**, **Steak**, or **Sushi**.")
    MODELS = ["lenet64", "tinyvgg", "resnet18"]

    # Initialize Session State
    if "prediction" not in st.session_state:
        st.session_state.prediction = None
    if "probabilities" not in st.session_state:
        st.session_state.probabilities = None
    if "confidence" not in st.session_state:
        st.session_state.confidence = None
    if "last_uploaded" not in st.session_state:
        st.session_state.last_uploaded = None
    if "video_data" not in st.session_state:
        st.session_state.video_data = None

    # Model Information Metadata
    MODEL_INFO = {
        "lenet64": {
            "description": "LeNet64 is a classic LeNet architecture adapted for 64x64 RGB images. It uses two convolutional layers followed by average pooling and three fully connected layers.",
            "accuracy": "76.14%",
            "parameters": "1.2M",
            "size": "1.3 MB",
            "training_images": "1000",
            "inference_time": "15ms",
            "plot": "images/lenet.png"
        },
        "tinyvgg": {
            "description": "TinyVGG is a lightweight VGG-inspired convolutional network. It consists of multiple convolutional blocks with max pooling and dropout for regularization.",
            "accuracy": "89.5%",
            "parameters": "5.1M",
            "size": "5.1 MB",
            "training_images": "1000",
            "inference_time": "25ms",
            "plot": "images/tinyvgg.png"
        },
        "resnet18": {
            "description": "ResNet-18 is a convolutional neural network that is 18 layers deep. It is known for its residual learning framework to ease the training of networks that are substantially deeper.",
            "accuracy": "94.8%",
            "parameters": "11.2M",
            "size": "44.7 MB",
            "training_images": "1000 (Transfer Learning)",
            "inference_time": "45ms",
            "plot": "images/resnet.png"
        }
    }

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        selected_model = st.selectbox("Select Model", MODELS)
        
        st.divider()
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        analyze_button = st.button("Analyze Image", type="primary")

    # Clear state if a new file is uploaded
    if uploaded_file is not None and uploaded_file.name != st.session_state.last_uploaded:
        st.session_state.prediction = None
        st.session_state.probabilities = None
        st.session_state.confidence = None
        st.session_state.last_uploaded = uploaded_file.name
        st.session_state.video_data = None
        st.session_state.show_nutrients = False

    # Main Layout
    # Image/Results section
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.subheader("Uploaded Image")
        st.image(image, use_container_width=True)

        if analyze_button:
            with st.spinner("Analyzing..."):
                try:
                    uploaded_file.seek(0)
                    image_array = preprocess(image, selected_model)
                    session = ort.InferenceSession(
                        str(settings.ONNX_PATH[selected_model]), 
                        providers=["CUDAExecutionProvider", "CPUExecutionProvider"]
                    )
                    pred_index, prob = run_inference(session, image_array)
                    
                    if pred_index is not None:
                        st.session_state.prediction = settings.CLASS_NAMES[pred_index]
                        st.session_state.confidence = prob[pred_index]
                        st.session_state.probabilities = prob
                    else:
                        st.error("Inference failed.")
                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")

        # Display results from session state
        if st.session_state.prediction:
            st.divider()
            res_col1, res_col2 = st.columns([1, 1])
            with res_col1:
                st.subheader("Results")
                st.markdown(f"### Predicted: **{st.session_state.prediction.title()}**")
                st.metric(label="Confidence", value=f"{st.session_state.confidence:.2%}")
            
            with res_col2:
                # Prediction Probabilities (Pie Chart)
                probs_df = pd.DataFrame({
                    "Class": settings.CLASS_NAMES,
                    "Probability": st.session_state.probabilities
                })
                
                fig = px.pie(
                    probs_df, 
                    values="Probability", 
                    names="Class", 
                    title="Class Probabilities",
                    hole=0.4,
                    color_discrete_sequence=px.colors.sequential.RdBu,
                    height=300
                )
                fig.update_traces(textposition='inside', textinfo='percent+label')
                st.plotly_chart(fig, use_container_width=True)

            # Nutrients and Video
            st.divider()
            st.subheader("Nutritional Information (per 100g)")
            
            if st.button("Show Nutritional Information"):
                st.session_state.show_nutrients = True
            
            if st.session_state.get("show_nutrients", False):
                nutrients = filter_csv_by_label(st.session_state.prediction)
                food_description = get_food_description(st.session_state.prediction)
                if not nutrients.empty:
                    display_df = nutrients.drop(columns=['id', 'label'])
                    st.table(display_df.set_index(pd.Index(['Quantity'])))
                    st.write(food_description)

                    # Display YouTube Video
                    st.divider()
                    st.subheader(f"How to Cook {st.session_state.prediction.title()}")
                    
                    if not st.session_state.video_data:
                        with st.spinner("Finding a cooking video..."):
                            st.session_state.video_data = get_cooking_videos(st.session_state.prediction)
                    
                    if st.session_state.video_data:
                        st.write(f"**Video:** {st.session_state.video_data['title']}")
                        st.video(st.session_state.video_data['url'])
                    else:
                        st.info("Could not find a recipe video.")
                else:
                    st.warning(f"No nutritional data available for {st.session_state.prediction}.")
    else:
        st.info("Please upload an image to start.")

    # Model Specifications section (now at the bottom)
    st.divider()
    st.header("Model Specifications")
    info = MODEL_INFO[selected_model]
    
    st.info(f"**Description:** {info['description']}")
    
    # Metrics in columns
    m_col1, m_col2 = st.columns(2)
    with m_col1:
        st.metric("Accuracy", info['accuracy'])
        st.metric("Parameters", info['parameters'])
        st.metric("Model Size", info['size'])
    with m_col2:
        st.metric("Train Images", info['training_images'])
        st.metric("Inference Time", info['inference_time'])
        
    st.divider()
    st.subheader("Training vs Testing Plot")
    try:
        plot_image = Image.open(info['plot'])
        st.image(plot_image, use_container_width=True)
    except Exception as e:
        st.warning(f"Could not load plot for {selected_model}")

if __name__ == "__main__":
    main()
