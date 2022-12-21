# Point E

This is a Streamlit application that generates 3D point clouds based on a text prompt. The application utilizes two trained models, a base model and an upsample model, to generate the point clouds. The base model generates a low resolution point cloud and the upsample model refines it to a higher resolution. 

## How to Use

To use the application, simply enter a text prompt in the designated input field and click the 'Generate' button. The application will then generate a 3D point cloud based on the prompt and display it in the browser.

## Dependencies

The following libraries are required to run the application:

- Streamlit
- PyTorch
- tqdm

## Local Development

To run the application locally, clone the repository and install the required dependencies. Then, run the following command in the root directory:
```
streamlit run streamlit_app.py
```

This will start the application and open it in your default web browser.

## Credit

The models used in this application were trained by OpenAI and can be found [here](https://github.com/openai/point-e).

