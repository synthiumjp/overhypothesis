import torch
import torch_directml

# 1. Check if the 'translator' (DirectML) is working
available = torch_directml.is_available()
print(f'Is DirectML available? {available}')

if available:
    # 2. Assign the AMD GPU to a variable called 'device'
    device = torch_directml.device()
    
    # 3. Create a small piece of data and move it to the GPU
    x = torch.tensor([1.0, 2.0, 3.0]).to(device)
    
    print(f'Success! Data is currently on: {x.device}')
else:
    print('Hardware not found. We may need to check your AMD drivers.')
