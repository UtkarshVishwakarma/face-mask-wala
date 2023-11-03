from keras.models import load_model
from mtcnn.mtcnn import MTCNN
import streamlit as st
from PIL import Image
import os
import cv2

detector = MTCNN()

model = load_model('model.h5')

st.title("Face Mask Detector 🎭😷")

uploaded_image = st.file_uploader("Upload an image", type=['png', 'jpg', 'jpeg'])

st.warning("⚠ Only! Select or Drop image of A single human")

# Directory where you want to save the uploaded images
output_directory = 'uploads'  

# Function to save the uploaded image
def save_image(uploaded_image):
    waste = os.listdir('uploads')[0]
    os.remove(os.path.join('uploads', waste))
    if not os.path.exists(output_directory):
        os.makedirs(output_directory)

    # Convert the uploaded image to a PIL Image
    image = Image.open(uploaded_image)

    # Construct the output image path with a unique name
    image_name = f"uploaded_{uploaded_image.name}"
    output_image_path = os.path.join(output_directory, image_name)

    # Save the image to the output directory
    image.save(output_image_path)
    return output_image_path

try:
    if uploaded_image is not None:
    # Display the uploaded image
        st.image(uploaded_image, caption="Uploaded Image", use_column_width=True)

        # Save the uploaded image and display a success message
        saved_image_path = save_image(uploaded_image)

        img = cv2.imread(saved_image_path)

        faces = detector.detect_faces(img)

        x, y, width, height = faces[0]['box']

        img_final = img[x:x+width, y:y+height]
        img_final = cv2.resize(img_final, (224, 224))

        b_img = img_final.reshape(1, 224, 224, 3)

        pred = model.predict(b_img)

        if pred[0][0] < 0.50:
            pred_final = 'No Mask ❌'
            st.error(pred_final)

        else:
            pred_final = 'Mask ✅'
            st.success(pred_final) 

except:
    st.header("Image Not Valid 🚫")


    


# Note: In a real-world application, you may want to handle errors and provide better user feedback.

