# VisionAPI - Real-Time Object Detection

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-latest-orange.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)

Production-ready object detection API powered by YOLOv8 and FastAPI.

## 🚀 Features

- **Real-time object detection** using YOLOv8
- **RESTful API** with FastAPI
- **80+ object classes** detection (COCO dataset)
- **Adjustable confidence** and IOU thresholds
- **JSON and annotated image** outputs
- **Docker containerized**
- **Comprehensive testing**
- **Interactive API documentation**
- **Beautiful web interface**

## 📋 Requirements

- Python 3.10+
- Docker Desktop (for containerized deployment)

## 🛠️ Installation

### Option 1: Local Development

1. **Clone the repository**
```bash
git clone https://github.com/KhaNerd1/vision-api.git
cd vision-api
```

2. **Create virtual environment**
```bash
python -m venv venv
venv\Scripts\activate  # Windows
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Run the API**
```bash
uvicorn app.main:app --reload
```

The API will be available at `http://127.0.0.1:8000`

### Option 2: Docker

1. **Build and run**
```bash
docker-compose up --build
```

The API will be available at `http://127.0.0.1:8000`

## 🎯 Usage

### API Endpoints

- **GET /** - API information
- **GET /health** - Health check
- **POST /api/v1/detect** - Detect objects (returns JSON)
- **POST /api/v1/detect/annotated** - Get annotated image
- **GET /api/v1/model/info** - Model information

### Interactive Documentation

Visit `http://127.0.0.1:8000/docs` for interactive API documentation.

### Web Interface

Open `frontend/index.html` in your browser for a user-friendly interface.

### Example with cURL

**Detect objects:**
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/detect" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" \
  -F "confidence=0.5"
```

**Get annotated image:**
```bash
curl -X POST "http://127.0.0.1:8000/api/v1/detect/annotated" \
  -H "Content-Type: multipart/form-data" \
  -F "file=@image.jpg" \
  -o output.jpg
```

### Example with Python
```python
import requests

url = "http://127.0.0.1:8000/api/v1/detect"

with open("image.jpg", "rb") as f:
    files = {"file": f}
    params = {"confidence": 0.5}
    response = requests.post(url, files=files, params=params)
    
detections = response.json()
print(f"Found {detections['count']} objects")
for det in detections['detections']:
    print(f"- {det['class_name']}: {det['confidence']:.2f}")
```

## 🧪 Testing

Run tests:
```bash
pytest tests/ -v
```

Run with coverage:
```bash
pytest tests/ --cov=app --cov-report=html
```

## 📁 Project Structure
```
vision-api/
├── app/
│   ├── models/
│   │   └── detector.py       # YOLO detector class
│   ├── schemas/
│   │   └── __init__.py       # Pydantic models
│   ├── services/
│   ├── utils/
│   └── main.py               # FastAPI application
├── frontend/
│   └── index.html            # Web interface
├── tests/
│   └── test_api.py           # API tests
├── uploads/                   # Temporary uploads
├── outputs/                   # Processed images
├── docker-compose.yml
├── Dockerfile
├── requirements.txt
└── README.md
```

## 🎨 Detected Object Classes

The model can detect 80 object classes including:
- People, vehicles, animals
- Household items, furniture
- Food, electronics
- Sports equipment
- And more!

See full list at `/api/v1/model/info`

## 🚀 Deployment

### Deploy to Render.com

1. Push to GitHub
2. Connect Render to your repository
3. Create new Web Service
4. Set build command: `docker build -t vision-api .`
5. Set start command: `docker run -p 8000:8000 vision-api`

### Deploy to AWS ECS

See [deployment guide](docs/deployment.md) (coming soon)

## 📊 Performance

- **Average inference time**: ~100-300ms per image (CPU)
- **Throughput**: ~3-10 requests/second
- **Model size**: 6MB (YOLOv8n)

## 🔧 Configuration

Adjust detection parameters:
- `confidence`: Minimum confidence threshold (0.1-1.0)
- `iou_threshold`: IOU threshold for NMS (0.1-1.0)

## 🤝 Contributing

Contributions are welcome! Please:
1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📝 License

This project is licensed under the MIT License.

## 👤 Author

**Your Name**
- GitHub: [@KhaNerd1](https://github.com/KhaNerd1)

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [FastAPI](https://fastapi.tiangolo.com/)
- COCO dataset

## 📧 Contact

For questions or feedback, please open an issue on GitHub.

---

⭐ If you find this project useful, please consider giving it a star!