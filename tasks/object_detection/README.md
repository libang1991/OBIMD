# OBIMD Object Detection Training

We use the Ultralytics YOLO framework to train a model on the OBIMD dataset.
The training is based on the official `yolov11l.pt` pre-trained weights.

```python
from ultralytics import YOLO

model = YOLO('yolov11l.pt')
model.train(data='path/to/obimd.yaml', epochs=100)
```

---

and all code we put to [train file](./train.py)
