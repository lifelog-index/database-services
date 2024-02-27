import sys 
import clip
# try to get argv[1] as model_name else use default

model_name = "ViT-L/14@336px"
model_name = sys.argv[1] if len(sys.argv) > 1 else model_name

_, _ = clip.load(model_name, device='cpu')