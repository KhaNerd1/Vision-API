# VisionAPI - Real-Time Object Detection

![Python](https://img.shields.io/badge/python-3.10-blue.svg)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green.svg)
![YOLOv8](https://img.shields.io/badge/YOLOv8-latest-orange.svg)
![Docker](https://img.shields.io/badge/docker-ready-blue.svg)

Production-ready object detection API powered by YOLOv8 and FastAPI.

## 🌐 Live Demo

- **🎨 Web Interface**: [https://khanerd1.github.io/Vision-API/](https://khanerd1.github.io/Vision-API/)
- **📡 API Endpoint**: [https://vision-api-960h.onrender.com](https://vision-api-960h.onrender.com)
- **📚 Interactive API Docs**: [https://vision-api-960h.onrender.com/docs](https://vision-api-960h.onrender.com/docs)
- **🏥 Health Check**: [https://vision-api-960h.onrender.com/health](https://vision-api-960h.onrender.com/health)

> **⚠️ Note**: First request may take 30-60 seconds as the free tier instance spins up from sleep mode. Subsequent requests are fast!

### Quick Test
Try it now! Go to the [Web Interface](https://khanerd1.github.io/Vision-API/) and upload any image to see real-time object detection in action.

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


## 📊 Performance

- **Average inference time**: ~100-300ms per image (CPU)
- **Throughput**: ~3-10 requests/second
- **Model size**: 6MB (YOLOv8n)

## 🔧 Configuration

Adjust detection parameters:
- `confidence`: Minimum confidence threshold (0.1-1.0)
- `iou_threshold`: IOU threshold for NMS (0.1-1.0)


## 📝 License

This project is licensed under the MIT License.

## 👤 Author

**Your Name**
- GitHub: [@KhaNerd1](https://github.com/KhaNerd1)

## 🙏 Acknowledgments

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [FastAPI](https://fastapi.tiangolo.com/)
- COCO dataset

⭐ If you find this project useful, please consider giving it a star!
