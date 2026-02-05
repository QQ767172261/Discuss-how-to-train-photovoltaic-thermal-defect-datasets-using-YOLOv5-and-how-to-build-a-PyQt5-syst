![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/9a749f36ec8d4f83b0f20e2eb277e620.png)
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/8571a35f9cbd45bab4fbcac19a2d20dc.png)
光伏热红外缺陷数据集 1W+带标注xml 格式（yolo格式）  类别为两种 0为单点热斑；1为局部热斑![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/41dfa67209ec4dc6b3c3e5b7b613635b.png)
好的，让我们一步步来处理这个问题。首先，我们将讨论如何使用YOLOv5训练光伏热红外缺陷数据集，然后讨论如何使用这套数据集构建一个PyQt5系统。

![### 1. 数据准备](https://i-blog.csdnimg.cn/direct/fbc97f751bf74767a9f8422149313cdc.png)


#### 1.1. 组织数据
确保你的数据按照以下结构组织：
```
dataset/
│
├── images/
│   ├── train/
│   ├── val/
│   └── test/
│
├── labels/
│   ├── train/
│   ├── val/
│   └── test/
│
└── data.yaml
```
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/4de1975c852144f997d85cd6d9613c74.png)

#### 1.2. 创建数据配置文件
创建一个`data.yaml`文件来配置数据集：
```yaml
train: ./dataset/images/train
val: ./dataset/images/val
test: ./dataset/images/test

nc: 2  # 类别数量（单点热斑和局部热斑）
names: ['single_hotspot', 'local_hotspot']  # 类别名称

# 下载数据集
download: ''
```

### 2. 转换标注格式
你需要将XML格式的标注转换为YOLO格式。可以使用Python脚本来完成这个任务。

#### 2.1. 转换标注文件
```python
import os
import xml.etree.ElementTree as ET
import glob

def convert_xml_to_yolo(xml_file, img_width, img_height):
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    bboxes = []
    for obj in root.findall('object'):
        bbox = obj.find('bndbox')
        x_min = int(bbox.find('xmin').text)
        y_min = int(bbox.find('ymin').text)
        x_max = int(bbox.find('xmax').text)
        y_max = int(bbox.find('ymax').text)
        
        x_center = (x_min + x_max) / 2.0
        y_center = (y_min + y_max) / 2.0
        width = x_max - x_min
        height = y_max - y_min
        
        x_center /= img_width
        y_center /= img_height
        width /= img_width
        height /= img_height
        
        label = int(obj.find('name').text)
        bboxes.append([label, x_center, y_center, width, height])
    
    return bboxes

def convert_dataset(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    xml_files = glob.glob(os.path.join(input_dir, '*.xml'))
    for xml_file in xml_files:
        img_file = xml_file.replace('.xml', '.jpg')
        img = cv2.imread(img_file)
        img_height, img_width, _ = img.shape
        
        bboxes = convert_xml_to_yolo(xml_file, img_width, img_height)
        
        label_file = xml_file.replace('.xml', '.txt')
        label_file = label_file.replace('images', 'labels')
        label_file = label_file.replace('train', 'train')
        
        with open(label_file, 'w') as f:
            for bbox in bboxes:
                f.write(' '.join([str(x) for x in bbox]) + '\n')

# 示例用法
convert_dataset('./dataset/images/train', './dataset/labels/train')
```

### 3. 安装YOLOv5
确保你已经克隆并安装了YOLOv5仓库：
```bash
git clone https://github.com/ultralytics/yolov5.git
cd yolov5
pip install -r requirements.txt
```

### 4. 训练模型

#### 4.1. 配置训练
使用YOLOv5进行目标检测。以下是一个示例命令：
```bash
python train.py --data ./dataset/data.yaml --img 640 --batch 16 --epochs 100 --name yolov5_custom --weights yolov5s.pt
```

### 5. 评估模型

#### 5.1. 运行评估
在验证集上评估训练好的模型：
```bash
python val.py --data ./dataset/data.yaml --weights runs/train/yolov5_custom/weights/best.pt
```

#### 5.2. 可视化结果
你可以使用`val`命令的`--save`标志来可视化结果：
```bash
python val.py --data ./dataset/data.yaml --weights runs/train/yolov5_custom/weights/best.pt --save
```

### 6. 使用PyQt5构建系统

#### 6.1. 安装PyQt5
```bash
pip install PyQt5
```

#### 6.2. 创建PyQt5应用
创建一个简单的PyQt5应用来展示检测结果。

```python
import sys
from PyQt5.QtWidgets import QApplication, QWidget, QVBoxLayout, QLabel, QPushButton, QFileDialog
from PyQt5.QtGui import QPixmap
from PyQt5.QtCore import Qt
import cv2
import numpy as np

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()

    def initUI(self):
        self.setWindowTitle('光伏热红外缺陷检测')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        self.load_button = QPushButton('加载图片', self)
        self.load_button.clicked.connect(self.load_image)
        layout.addWidget(self.load_button)

        self.setLayout(layout)

    def load_image(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            file_name = file_dialog.selectedFiles()[0]
            self.image_label.setPixmap(QPixmap(file_name))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
```

#### 6.3. 集成YOLOv5检测
```python
import torch
from yolov5.utils.general import non_max_suppression

class MainWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.model = torch.hub.load('ultralytics/yolov5', 'custom', path='runs/train/yolov5_custom/weights/best.pt')

    def initUI(self):
        self.setWindowTitle('光伏热红外缺陷检测')
        self.setGeometry(100, 100, 800, 600)

        layout = QVBoxLayout()

        self.image_label = QLabel(self)
        layout.addWidget(self.image_label)

        self.load_button = QPushButton('加载图片', self)
        self.load_button.clicked.connect(self.load_image)
        layout.addWidget(self.load_button)

        self.setLayout(layout)

    def load_image(self):
        file_dialog = QFileDialog()
        file_dialog.setFileMode(QFileDialog.ExistingFile)
        if file_dialog.exec_():
            file_name = file_dialog.selectedFiles()[0]
            self.image_label.setPixmap(QPixmap(file_name))
            self.detect_defects(file_name)

    def detect_defects(self, file_name):
        img = cv2.imread(file_name)
        results = self.model(img)
        detections = results.pandas().xyxy[0]
        for _, row in detections.iterrows():
            x1, y1, x2, y2 = int(row['xmin']), int(row['ymin']), int(row['xmax']), int(row['ymax'])
            cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)
            cv2.putText(img, row['name'], (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        height, width, channel = img.shape
        bytes_per_line = 3 * width
        q_img = QImage(img.data, width, height, bytes_per_line, QImage.Format_RGB888)
        self.image_label.setPixmap(QPixmap.fromImage(q_img))

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
```

### 7. 其他建议

- **数据增强**：使用数据增强技术来提高模型的鲁棒性。
- **模型集成**：集成多个模型以提高性能。
- **模型量化**：量化模型以适应边缘设备。


