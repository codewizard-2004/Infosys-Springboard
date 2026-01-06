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

    # Sidebar
    with st.sidebar:
        st.header("Configuration")
        selected_model = st.selectbox("Select Model", MODELS)
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
    col1, col2 = st.columns([1, 1])

    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        with col1:
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
            with col2:
                st.subheader("Results")
                st.markdown(f"### Predicted: **{st.session_state.prediction.title()}**")
                st.metric(label="Confidence", value=f"{st.session_state.confidence:.2%}")

            # Elements moved outside the column view
            st.divider()
            
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
                color_discrete_sequence=px.colors.sequential.RdBu
            )
            fig.update_traces(textposition='inside', textinfo='percent+label')
            st.plotly_chart(fig, use_container_width=True)

            # Display Nutritional Information
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
        with col1:
            st.info("Please upload an image to start.")

if __name__ == "__main__":
    main()
