import streamlit as st

import torch
from tqdm.auto import tqdm

from point_e.diffusion.configs import DIFFUSION_CONFIGS, diffusion_from_config
from point_e.diffusion.sampler import PointCloudSampler
from point_e.models.download import load_checkpoint
from point_e.models.configs import MODEL_CONFIGS, model_from_config
from point_e.util.plotting import plot_point_cloud

import time

LOG_FILE_NAME = "log.txt"

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# Set a prompt to condition on.
# prompt = 'a red motorcycle'
prompt = st.text_input('Prompt', 'a red motorcycle')

if not st.button('Generate'):
    st.stop()

with open(LOG_FILE_NAME, 'a') as f:
    f.write(f'{time.time()}, {prompt}\n')

print('creating base model...')
with st.spinner('Creating base model...'):
    base_name = 'base40M-textvec'
    base_model = model_from_config(MODEL_CONFIGS[base_name], device)
    base_model.eval()
    base_diffusion = diffusion_from_config(DIFFUSION_CONFIGS[base_name])

print('creating upsample model...')
with st.spinner('Creating upsample model...'):
    upsampler_model = model_from_config(MODEL_CONFIGS['upsample'], device)
    upsampler_model.eval()
    upsampler_diffusion = diffusion_from_config(DIFFUSION_CONFIGS['upsample'])

print('downloading base checkpoint...')
with st.spinner('Downloading base checkpoint...'):
    base_model.load_state_dict(load_checkpoint(base_name, device))

print('downloading upsampler checkpoint...')
with st.spinner('Downloading upsampler checkpoint...'):
    upsampler_model.load_state_dict(load_checkpoint('upsample', device))


sampler = PointCloudSampler(
    device=device,
    models=[base_model, upsampler_model],
    diffusions=[base_diffusion, upsampler_diffusion],
    num_points=[1024, 4096 - 1024],
    aux_channels=['R', 'G', 'B'],
    guidance_scale=[3.0, 0.0],
    model_kwargs_key_filter=('texts', ''), # Do not condition the upsampler at all
)



# Produce a sample from the model.
start_time = time.time()
with st.spinner('Generating point cloud...'):
    samples = None
    step_num = 0
    step_num_text_field = st.empty()
    step_num_text_field.text('Step number: 0')
    for x in tqdm(sampler.sample_batch_progressive(batch_size=1, model_kwargs=dict(texts=[prompt]))):
        samples = x
        step_num_text_field.text(f'Step {step_num}')
        step_num += 1

pc = sampler.output_to_point_clouds(samples)[0]

st.write(plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75))))

fig = plot_point_cloud(pc, grid_size=3, fixed_bounds=((-0.75, -0.75, -0.75),(0.75, 0.75, 0.75)))
