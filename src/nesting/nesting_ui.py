import streamlit as st
import json, time, os
from services import dxf_io, nesting, preview

st.header('WJP Nesting Test UI')

cfg = {
    'min_gap_mm': 3.0, 'kerf_mm': 1.1, 'edge_margin_mm': 10.0,
    'rotation_step_deg': 15, 'max_rotations': 12, 'retry_rounds': 3,
    'cost_per_meter': 825
}

uploaded_file = st.file_uploader('Upload DXF file', type=['dxf'])
handles = st.text_area('Enter DXF Handles (comma separated)')
frame_w = st.number_input('Frame Width (mm)', 100.0, 5000.0, 1000.0)
frame_h = st.number_input('Frame Height (mm)', 100.0, 5000.0, 1000.0)

if st.button('Run Nesting') and uploaded_file and handles:
    handles_list = [h.strip() for h in handles.split(',') if h.strip()]
    tmp_path = '/mnt/data/tmp_input.dxf'
    with open(tmp_path, 'wb') as f:
        f.write(uploaded_file.read())

    objs = dxf_io.read_objects(tmp_path, handles_list)
    frame = {'width': frame_w, 'height': frame_h, 'margin': cfg['edge_margin_mm']}
    result = nesting.nest(objs, frame, cfg)

    ts = str(int(time.time()))
    out_dir = f'/mnt/data/nesting_run_{ts}'
    os.makedirs(out_dir, exist_ok=True)
    out_dxf = f'{out_dir}/nested_output.dxf'
    dxf_io.write_nested(tmp_path, result['placed'], out_dxf)
    with open(f'{out_dir}/metrics.json', 'w') as f:
        json.dump(result['metrics'], f, indent=2)

    st.json(result['metrics'])
    img_path = preview.render(result, frame_w, frame_h)
    st.image(img_path, caption='Nested Layout')
    st.download_button('Download Nested DXF', data=open(out_dxf, 'rb').read(), file_name='nested_output.dxf')
