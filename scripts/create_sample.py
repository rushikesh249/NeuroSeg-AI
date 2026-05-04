import nibabel as nib
import numpy as np
import os

def create_dummy_nifti(filename, shape=(128, 128, 128)):
    # Create random data to simulate an MRI volume
    data = np.random.rand(*shape).astype(np.float32)
    # Add some "structure" so it's not just noise in the viewer
    xx, yy, zz = np.ogrid[:shape[0], :shape[1], :shape[2]]
    center = (shape[0]//2, shape[1]//2, shape[2]//2)
    distance = (xx - center[0])**2 + (yy - center[1])**2 + (zz - center[2])**2
    data[distance < (shape[0]//3)**2] += 0.5
    
    affine = np.eye(4)
    img = nib.Nifti1Image(data, affine)
    nib.save(img, filename)
    print(f"✅ Created {filename}")

def main():
    sample_dir = "sample_data"
    os.makedirs(sample_dir, exist_ok=True)
    
    modalities = ["flair", "t1", "t1ce", "t2"]
    for mod in modalities:
        create_dummy_nifti(os.path.join(sample_dir, f"sample_{mod}.nii.gz"))
    
    print("\n🚀 Done! You can now drag and drop the files from the 'sample_data' folder into your UI.")

if __name__ == "__main__":
    main()
