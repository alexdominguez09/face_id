# Face ID - Real-time Face Recognition System

![Python](https://img.shields.io/badge/Python-3.9%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Platform](https://img.shields.io/badge/Platform-Linux-lightgrey)

A high-accuracy real-time face recognition system for Linux that detects faces in crowds, assigns unique persistent IDs, and recognizes them across multiple video sessions.

## 🎯 Features

- **Real-time Face Detection**: Detect multiple faces in crowded scenes using MTCNN
- **High-Accuracy Recognition**: State-of-the-art ArcFace embeddings via InsightFace
- **Persistent Face IDs**: Assign and maintain unique identifiers for each person
- **GPU Acceleration**: CUDA support for NVIDIA GPUs
- **Dual Interface**: Both CLI and web-based management
- **Live Monitoring**: Real-time video streaming with face overlay
- **Face Gallery**: Search and manage enrolled faces

## 🏗️ Architecture

```
+-----------------------------+
|      Web Interface          |
|  (FastAPI + HTML/JS)        |
+-------------+---------------+
              |
+-------------v---------------+
|      CLI Interface          |
|  (Command-line operations)  |
+-------------+---------------+
              |
+-------------v---------------+
|   Face Recognition Engine   |
|  [Detector] -> [Recognizer] |
|    (MTCNN)    (InsightFace) |
+-------------+---------------+
              |
+-------------v---------------+
|      Storage Layer          |
|    (SQLite Database)        |
+-----------------------------+
```

## 📁 Project Structure

```
face_id/
├── face_recognition/          # Core recognition engine
│   ├── __init__.py
│   ├── detector.py           # MTCNN face detection
│   ├── recognizer.py         # InsightFace recognition
│   ├── tracker.py            # Face tracking
│   ├── database.py           # SQLite operations
│   ├── utils.py              # Helper functions
│   └── config.py             # Configuration
├── cli/                      # Command-line interface
│   ├── __init__.py
│   └── main.py               # CLI commands
├── web/                      # Web interface
│   ├── __init__.py
│   ├── app.py                # FastAPI application
│   ├── routes/               # API endpoints
│   │   ├── faces.py
│   │   ├── video.py
│   │   └── api.py
│   └── static/               # Frontend assets
│       ├── index.html
│       ├── styles.css
│       └── app.js
├── models/                   # ML models
├── data/                     # Database and embeddings
├── tests/                    # Unit tests
├── requirements.txt          # Python dependencies
├── setup.py                  # Package setup
└── README.md                 # This file
```

## 🚀 Quick Start

### Prerequisites

- Linux OS (Ubuntu 20.04+ recommended)
- Python 3.9 or higher
- NVIDIA GPU with CUDA 11.0+ (optional but recommended)
- 8GB+ RAM

### Installation

```bash
# Clone the repository
git clone https://github.com/alexdominguez09/face_id.git
cd face_id

# Create virtual environment
python3 -m venv venv
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt

# Download models (automated)
python -m face_recognition.utils.download_models

# Initialize database
python -m face_recognition.database init
```

### Usage

#### CLI Interface

```bash
# Start real-time face detection
python -m cli.main start --camera 0

# Add a face manually
python -m cli.main add-face --name "John Doe" --image path/to/photo.jpg

# List all enrolled faces
python -m cli.main list-faces

# Process a video file
python -m cli.main process-video --input video.mp4 --output result.mp4
```

#### Web Interface

```bash
# Start the web server
python -m web.app

# Open browser to http://localhost:8000
```

## 🛠️ Technology Stack

- **Detection**: [MTCNN](https://github.com/ipazc/mtcnn)
- **Recognition**: [InsightFace](https://github.com/deepinsight/insightface)
- **Video Processing**: [OpenCV](https://opencv.org/)
- **Web Framework**: [FastAPI](https://fastapi.tiangolo.com/)
- **Database**: SQLite

## 📊 Performance Targets

- Detection Accuracy: >95% on clear faces
- Recognition Accuracy: >98% on known faces
- Processing Speed: 15-30 FPS (detection), 5-10 FPS (recognition)
- Latency: <200ms for face identification

## 🔒 Security & Privacy

- Face embeddings encrypted at rest
- GDPR-compliant data handling
- Secure web interface with authentication
- Regular security updates

## 📋 Development Roadmap

- [x] Project structure and documentation
- [ ] Phase 1: Core Engine implementation
- [ ] Phase 2: Recognition Pipeline
- [ ] Phase 3: CLI Interface
- [ ] Phase 4: Web Interface
- [ ] Phase 5: Optimization & Testing
- [ ] Phase 6: Documentation & Deployment

## 🤝 Contributing

Contributions are welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Commit your changes
4. Push to the branch
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License.

## 🙏 Acknowledgments

- [InsightFace](https://github.com/deepinsight/insightface) team
- [MTCNN](https://github.com/ipazc/mtcnn) authors
- [OpenCV](https://opencv.org/) community

---

**Note**: This system is for educational and legitimate use only. Always comply with local privacy laws and regulations.
