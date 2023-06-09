import gradio as gr
import requests
import os
from dotenv import load_dotenv

load_dotenv()
# Set URL
# Run: 
# docker run -p 8000:80 --name cls-serve hasibzunair/classification_model_serving
#REST_API_URL = os.getenv('MODEL_API_HOST') + os.getenv('IMAGE_CLASSIFICATION_ENDPOINT')
print(os.getenv('MODEL_API_HOST'))
print(os.getenv('IMAGE_CLASSIFICATION_ENDPOINT'))
REST_API_URL =  os.getenv('MODEL_API_HOST') + '/api/image-classification/predict'

# Inference!
def inference(image_path):
    print("inference")
    print(image_path)
    # Load the input image and construct the payload for the request
    image = open(image_path, "rb").read()
    payload = {"image": image}
    print(REST_API_URL)
    # Submit the request
    r = requests.post(REST_API_URL, files=payload).json()

    # Ensure the request was sucessful, format output for visualization
    output = {}
    if r["success"]:
        # Loop over the predictions and display them
        for (i, result) in enumerate(r["predictions"]):
            output[result["label"]] = result["probability"]
            print("{}. {}: {:.4f}".format(i + 1, result["label"],
                result["probability"]))
    else:
        print("Request failed")
    return output

# Define ins outs placeholders
inputs = gr.inputs.Image(type='filepath')
outputs = gr.outputs.Label(type="confidences",num_top_classes=5)

# Define style
title = "Image Recognition App"
description = "To use it, simply upload your image, or click one of the examples images to load them."

# Run inference
frontend = gr.Interface(inference, 
            inputs, 
            outputs, 
            examples=["test1.jpeg", "test2.jpeg"], 
            title=title, 
            description=description, 
            analytics_enabled=False)


# Launch app and set PORT
try:
    frontend.launch(server_name="0.0.0.0", server_port=7860)
except KeyboardInterrupt:
    frontend.close()
except Exception as e:
    print(e)
    frontend.close()



