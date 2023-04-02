import torch
import torchvision.models as models
from ultralytics import YOLO
from speedster import optimize_model, save_model

class YOLOWrapper(torch.nn.Module):
    def __init__(self, yolo_model):
        super().__init__()
        self.model = yolo_model.model
    
    def forward(self, x, *args, **kwargs):
        res = self.model(x)
        return res[0], *tuple(res[1])

#1 Provide input model and data (we support PyTorch Dataloaders and custom input, see the docs to learn more)
yolo = YOLO('/Users/brianchen/Desktop/Detector/runs/detect/train7/weights/best.pt')
model_wrapper = YOLOWrapper(yolo)

# Provide some input data for the model    
input_data = [((torch.randn(1, 3, 640, 640), ), torch.tensor([0])) for i in range(100)]

# Run Speedster optimization
optimized_model = optimize_model(
  model_wrapper, input_data=input_data, metric_drop_ths=0.1, store_latencies=True, device="gpu"
)

x = torch.randn(1, 3, 640, 640)

## Warmup the model
## This step is necessary before the latency computation of the 
## optimized model in order to get reliable results.
for _ in range(10):
  optimized_model(x)

res = optimized_model(x)
print(res)
#3 Save the optimized model
save_model(optimized_model, "/Users/brianchen/Desktop/Detector/runs/detect/train7/weights")