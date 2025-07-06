from ultralytics import YOLO
import os
from datetime import datetime

os.environ["MKL_THREADING_LAYER"] = "GNU"

# 实验配置（每次实验前修改这里）
experiment_config = {
    "type": "label",  # 实验类型：zitou/moban_zitou/sub_zitou
    "save_interval": 40,  # 保存间隔（epoch数）
    "project": "oracle_moben"  # 项目名称
}

# 生成带时间戳的实验名称
experiment_name = f"{experiment_config['type']}_{datetime.now().strftime('%Y%m%d%H%M')}"

# 加载模型
model = YOLO("/workspace/obc_check/obc_check/yolo11l.pt")

# 训练配置
train_results = model.train(
    data="/workspace/obc_check/obc_check/datasets/dataset.yaml",
    epochs=200,
    imgsz=1024,
    batch=8,
    device=(4, 5),
    patience=200,  # 使用内置早停
    project=experiment_config["project"],
    name=experiment_name,  # 唯一实验名称
    save_period=experiment_config["save_interval"],  # 自动保存间隔
    exist_ok=False  # 禁止覆盖已有实验
)

# 手动添加定期保存（双保险）
class CustomSaveCallback:
    def __init__(self, save_interval):
        self.save_interval = save_interval
        
    def on_train_epoch_end(self, trainer):
        if (trainer.epoch + 1) % self.save_interval == 0:
            filename = f"{experiment_name}_epoch{trainer.epoch+1}.pt"
            save_path = os.path.join(trainer.save_dir, filename)
            trainer.model.save(save_path)

# 添加回调
model.add_callback("on_train_epoch_end", CustomSaveCallback(experiment_config["save_interval"]))

# Evaluate model performance on the validation set
metrics = model.val()