
# SETUP INSTRUCTIONS

## 1. Environment Setup
```bash
# Create virtual environment
python -m venv presentation_coach_env

# Activate virtual environment
# On Windows:
presentation_coach_env\Scripts\activate
# On macOS/Linux:
source presentation_coach_env/bin/activate

# Install dependencies
pip install -r requirements.txt

# Install additional system dependencies (if needed)
# On Ubuntu/Debian:
sudo apt-get update
sudo apt-get install portaudio19-dev python3-pyaudio
sudo apt-get install ffmpeg

# On macOS:
brew install portaudio
brew install ffmpeg
```

## 2. API Keys Configuration
```python
# Create a .env file in your project root
OPENAI_API_KEY=your_openai_api_key_here
SECRET_KEY=your_secret_key_here
DATABASE_URL=sqlite:///presentation_coach.db

# Update the Flask app to use environment variables:
import os
from dotenv import load_dotenv
load_dotenv()

openai.api_key = os.getenv('OPENAI_API_KEY')
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'default-secret-key')
```

## 3. Database Setup
```bash
# Initialize the database
python -c "from app import app, db; app.app_context().push(); db.create_all()"
```

## 4. Frontend Integration Requirements

### JavaScript WebRTC Setup for Video/Audio Streaming:
```javascript
// Frontend code to capture and send video/audio streams
const startVideoStream = async () => {
    const stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: true
    });
    
    // Capture video frames
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    setInterval(() => {
        ctx.drawImage(video, 0, 0);
        const imageData = canvas.toDataURL('image/jpeg');
        
        fetch('/api/analyze_frame', {
            method: 'POST',
            headers: {'Content-Type': 'application/json'},
            body: JSON.stringify({
                image: imageData,
                session_id: currentSessionId
            })
        });
    }, 1000); // Analyze every second
};
```

## 5. Production Deployment Setup

### Using Gunicorn:
```bash
# Install gunicorn (already in requirements.txt)
pip install gunicorn

# Run with gunicorn
gunicorn --workers 4 --bind 0.0.0.0:5000 app:app
```

### Docker Setup (Optional):
```dockerfile
FROM python:3.9-slim

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

# Install system dependencies
RUN apt-get update && apt-get install -y \
    portaudio19-dev \
    python3-pyaudio \
    ffmpeg \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

COPY . .
EXPOSE 5000
CMD ["gunicorn", "--workers", "4", "--bind", "0.0.0.0:5000", "app:app"]
```

## 6. File Structure
```
presentation_coach/
│
├── app.py                 # Main Flask application
├── requirements.txt       # Python dependencies
├── .env                  # Environment variables
├── uploads/              # Directory for uploaded files
├── static/               # Static files (CSS, JS)
├── templates/            # HTML templates
│   └── index.html        # Main frontend
├── models/               # AI model files (if needed)
└── tests/                # Unit tests
```

## 7. Additional Configuration for Production

### SSL/HTTPS Setup:
```python
# For HTTPS in production
if __name__ == '__main__':
    app.run(debug=False, host='0.0.0.0', port=5000, ssl_context='adhoc')
```

### Nginx Configuration:
```nginx
server {
    listen 80;
    server_name your-domain.com;
    
    location / {
        proxy_pass http://localhost:5000;
        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;
    }
}
```

## 8. Performance Optimizations

### Redis for Caching (Optional):
```bash
pip install redis flask-caching
```

```python
from flask_caching import Cache

cache = Cache(app, config={'CACHE_TYPE': 'redis'})

# Cache expensive operations
@cache.cached(timeout=300)
def expensive_analysis_function():
    # Your analysis code here
    pass
```

## 9. Testing
```bash
# Install testing dependencies
pip install pytest pytest-flask

# Run tests
python -m pytest tests/
```

## 10. Monitoring and Logging
```python
import logging
from logging.handlers import RotatingFileHandler

if not app.debug:
    file_handler = RotatingFileHandler('logs/app.log', maxBytes=10240, backupCount=10)
    file_handler.setFormatter(logging.Formatter(
        '%(asctime)s %(levelname)s: %(message)s [in %(pathname)s:%(lineno)d]'
    ))
    file_handler.setLevel(logging.INFO)
    app.logger.addHandler(file_handler)
```

docker build -t presentation-coach .
docker run -p 5000:5000 --env-file .env presentation-coach