import requests
import json
import sseclient
import os
import numpy as np
import nibabel as nib

os.makedirs('tmp_data', exist_ok=True)
shape = (240, 240, 155)
files = {}
for mod in ['flair', 't1', 't1ce', 't2']:
    path = f'tmp_data/{mod}.nii.gz'
    if not os.path.exists(path):
        nib.save(nib.Nifti1Image(np.zeros(shape, dtype=np.float32), np.eye(4)), path)
    files[mod] = open(path, 'rb')

res = requests.post('http://localhost:8000/predict', files=files).json()
print("Job:", res)
if 'job_id' in res:
    response = requests.get(f"http://localhost:8000/progress/{res['job_id']}", stream=True)
    client = sseclient.SSEClient(response)
    for event in client.events():
        data = json.loads(event.data)
        print(data)
        if data['status'] in ['done', 'error']:
            break
