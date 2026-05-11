import sys
from PIL import Image
import torch
import torchvision.transforms as transforms
import os


output_path = "./output/"

os.makedirs(output_path, exist_ok=True)

if len(sys.argv) != 3:
    print("Usage: python main.py <image_path> <model_path>")
    sys.exit(1)

# Get image path from command line
image_path = sys.argv[1]

# Get model path from command line
model_path = sys.argv[2]

if torch.cuda.is_available():
    device_name = "cuda"
elif torch.backends.mps.is_available():
    device_name = "mps"
else:
    device_name = "cpu"

device = torch.device(device_name)
print(f"Code runs in {device}")


try:
    G = torch.jit.load(model_path, map_location=device)
    G.eval()
except Exception as e:
    print(f"An error has ocurred when loading the model {e}")
    sys.exit(1)

# Same transformations as when training
transform = transforms.Compose([
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

try:
    #Load input image
    input_image = Image.open(image_path).convert("RGB")
except Exception as e:
    print(f"An error has ocurred when loading the model {e}")
    sys.exit(1)

input_tensor = transform(input_image)

input_tensor = input_tensor.unsqueeze(0).to(device)

# Generate output
with torch.no_grad():
    generated_tensor = G(input_tensor)

# Remove batch dimension
generated_tensor = generated_tensor.squeeze(0).cpu()

# Denormalize from [-1,1] -> [0,1]
generated_tensor = generated_tensor * 0.5 + 0.5

# Clamp values just in case
generated_tensor = torch.clamp(generated_tensor, 0, 1)

# Convert to image
generated_image = transforms.ToPILImage()(generated_tensor)

# Save output
generated_image.save(output_path + "generated.png")

print("Generated image saved!")
