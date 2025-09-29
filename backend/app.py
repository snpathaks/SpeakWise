from flask import Flask, request, jsonify, render_template
from flask_sqlalchemy import SQLAlchemy
from flask_cors import CORS
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import os
import json
import base64
import io
from PIL import Image
from datetime import datetime
import numpy as np

# Import our custom modules
from models import db, User, Script, PracticeSession, FeedbackReport
from video_analyzer import VideoAnalyzer
from audio_analyzer import AudioAnalyzer
from content_analyzer import ContentAnalyzer

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'your-secret-key-here')
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///presentation_coach.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Initialize extensions
db.init_app(app)
CORS(app)

# Create upload directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize analyzers
video_analyzer = VideoAnalyzer()
audio_analyzer = AudioAnalyzer()
content_analyzer = ContentAnalyzer()

# Routes
@app.route('/')
def index():
    return jsonify({
        'message': 'AI Presentation Coach Backend',
        'status': 'running',
        'endpoints': {
            'upload_script': '/api/upload_script',
            'start_session': '/api/start_session',
            'analyze_frame': '/api/analyze_frame',
            'analyze_audio': '/api/analyze_audio',
            'end_session': '/api/end_session',
            'user_progress': '/api/user_progress/<user_id>',
            'user_scripts': '/api/scripts/<user_id>'
        }
    })

@app.route('/api/upload_script', methods=['POST'])
def upload_script():
    """Upload and store presentation script"""
    try:
        user_id = request.form.get('user_id', 1)  # Default user for demo
        title = request.form.get('title')
        category = request.form.get('category')
        content = request.form.get('content')
        
        if not title:
            return jsonify({'success': False, 'error': 'Title is required'}), 400
        
        # Handle file upload if provided
        filename = None
        if 'file' in request.files:
            file = request.files['file']
            if file.filename != '':
                filename = secure_filename(file.filename)
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                
                # Read content from file if no content provided
                if not content:
                    try:
                        with open(file_path, 'r', encoding='utf-8') as f:
                            content = f.read()
                    except UnicodeDecodeError:
                        try:
                            with open(file_path, 'r', encoding='latin-1') as f:
                                content = f.read()
                        except Exception as e:
                            return jsonify({'success': False, 'error': f'Could not read file: {str(e)}'}), 400
        
        if not content:
            return jsonify({'success': False, 'error': 'Content is required'}), 400
        
        # Save script to database
        script = Script(
            user_id=user_id,
            title=title,
            content=content,
            category=category or 'general',
            filename=filename
        )
        db.session.add(script)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'script_id': script.id,
            'message': 'Script uploaded successfully'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/start_session', methods=['POST'])
def start_session():
    """Initialize a new practice session"""
    try:
        data = request.json
        user_id = data.get('user_id', 1)
        script_id = data.get('script_id')
        
        if not script_id:
            return jsonify({'success': False, 'error': 'Script ID is required'}), 400
        
        # Verify script exists
        script = Script.query.get(script_id)
        if not script:
            return jsonify({'success': False, 'error': 'Script not found'}), 404
        
        session = PracticeSession(
            user_id=user_id,
            script_id=script_id,
            session_date=datetime.utcnow()
        )
        db.session.add(session)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'session_id': session.id,
            'script_content': script.content,
            'message': 'Practice session started'
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze_frame', methods=['POST'])
def analyze_frame():
    """Analyze video frame for real-time feedback"""
    try:
        data = request.json
        image_data = data.get('image')
        session_id = data.get('session_id')
        
        if not image_data:
            return jsonify({'success': False, 'error': 'Image data is required'}), 400
        
        # Decode base64 image
        try:
            image_bytes = base64.b64decode(image_data.split(',')[1])
            image = Image.open(io.BytesIO(image_bytes))
            frame = np.array(image)
            
            # Convert RGB to BGR for OpenCV
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                frame = frame[:, :, ::-1]  # RGB to BGR
        except Exception as e:
            return jsonify({'success': False, 'error': f'Invalid image data: {str(e)}'}), 400
        
        # Analyze frame
        analysis_results = video_analyzer.analyze_frame(frame)
        
        return jsonify({
            'success': True,
            'analysis': analysis_results
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/analyze_audio', methods=['POST'])
def analyze_audio():
    """Analyze audio chunk for real-time feedback"""
    try:
        if 'audio' not in request.files:
            return jsonify({'success': False, 'error': 'Audio file is required'}), 400
        
        audio_file = request.files['audio']
        session_id = request.form.get('session_id')
        
        # Save temporary audio file
        temp_path = os.path.join(app.config['UPLOAD_FOLDER'], f'temp_audio_{session_id}.wav')
        audio_file.save(temp_path)
        
        # Analyze audio
        analysis_results = audio_analyzer.analyze_audio_file(temp_path)
        
        # Clean up temp file
        try:
            os.remove(temp_path)
        except:
            pass  # File might already be deleted
        
        return jsonify({
            'success': True,
            'analysis': analysis_results
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/end_session', methods=['POST'])
def end_session():
    """End practice session and generate final report"""
    try:
        data = request.json
        session_id = data.get('session_id')
        duration = data.get('duration', 0)
        analysis_data = data.get('analysis_data', {})
        
        if not session_id:
            return jsonify({'success': False, 'error': 'Session ID is required'}), 400
        
        # Get session and script
        session = PracticeSession.query.get(session_id)
        if not session:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        script = Script.query.get(session.script_id)
        
        # Calculate overall scores
        scores = {
            'confidence_score': analysis_data.get('confidence_score', 70),
            'clarity_score': analysis_data.get('clarity_score', 75),
            'pace_score': analysis_data.get('pace_score', 80),
            'gesture_score': analysis_data.get('gesture_score', 85),
            'eye_contact_score': analysis_data.get('eye_contact_score', 78)
        }
        
        overall_score = sum(scores.values()) / len(scores)
        
        # Update session in database
        session.duration = duration
        session.overall_score = overall_score
        session.confidence_score = scores['confidence_score']
        session.clarity_score = scores['clarity_score']
        session.pace_score = scores['pace_score']
        session.gesture_score = scores['gesture_score']
        session.eye_contact_score = scores['eye_contact_score']
        session.filler_word_count = analysis_data.get('filler_word_count', 5)
        session.feedback_data = json.dumps(analysis_data)
        
        db.session.commit()
        
        # Generate AI feedback
        transcription = analysis_data.get('transcription', '')
        content_feedback = content_analyzer.analyze_content(script.content, transcription)
        
        # Save feedback report
        feedback = FeedbackReport(
            session_id=session_id,
            feedback_type='comprehensive',
            feedback_text=json.dumps(content_feedback),
            improvement_suggestions=json.dumps(content_feedback.get('improvements', []))
        )
        db.session.add(feedback)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'session_id': session_id,
            'overall_score': overall_score,
            'detailed_scores': scores,
            'filler_words': session.filler_word_count,
            'duration': session.duration,
            'ai_feedback': content_feedback
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/user_progress/<int:user_id>')
def user_progress(user_id):
    """Get user's practice progress over time"""
    try:
        sessions = PracticeSession.query.filter_by(user_id=user_id).order_by(
            PracticeSession.session_date.desc()
        ).limit(10).all()
        
        progress_data = []
        for session in sessions:
            progress_data.append({
                'session_id': session.id,
                'date': session.session_date.isoformat(),
                'overall_score': session.overall_score,
                'confidence_score': session.confidence_score,
                'clarity_score': session.clarity_score,
                'pace_score': session.pace_score,
                'gesture_score': session.gesture_score,
                'eye_contact_score': session.eye_contact_score,
                'filler_word_count': session.filler_word_count,
                'duration': session.duration
            })
        
        return jsonify({
            'success': True,
            'progress': progress_data
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/scripts/<int:user_id>')
def get_user_scripts(user_id):
    """Get all scripts for a user"""
    try:
        scripts = Script.query.filter_by(user_id=user_id).order_by(
            Script.uploaded_at.desc()
        ).all()
        
        script_list = []
        for script in scripts:
            script_list.append({
                'id': script.id,
                'title': script.title,
                'category': script.category,
                'uploaded_at': script.uploaded_at.isoformat(),
                'content_preview': script.content[:200] + '...' if len(script.content) > 200 else script.content,
                'word_count': len(script.content.split())
            })
        
        return jsonify({
            'success': True,
            'scripts': script_list
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

@app.route('/api/session_details/<int:session_id>')
def session_details(session_id):
    """Get detailed feedback for a specific session"""
    try:
        session = PracticeSession.query.get(session_id)
        if not session:
            return jsonify({'success': False, 'error': 'Session not found'}), 404
        
        feedback = FeedbackReport.query.filter_by(session_id=session_id).first()
        
        return jsonify({
            'success': True,
            'session': {
                'id': session.id,
                'date': session.session_date.isoformat(),
                'duration': session.duration,
                'overall_score': session.overall_score,
                'scores': {
                    'confidence': session.confidence_score,
                    'clarity': session.clarity_score,
                    'pace': session.pace_score,
                    'gestures': session.gesture_score,
                    'eye_contact': session.eye_contact_score
                },
                'filler_words': session.filler_word_count,
                'feedback_data': json.loads(session.feedback_data) if session.feedback_data else {},
                'ai_feedback': json.loads(feedback.feedback_text) if feedback else {}
            }
        })
    
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500

# Error handlers
@app.errorhandler(404)
def not_found(error):
    return jsonify({'success': False, 'error': 'Endpoint not found'}), 404

@app.errorhandler(500)
def internal_error(error):
    return jsonify({'success': False, 'error': 'Internal server error'}), 500

@app.errorhandler(413)
def too_large(error):
    return jsonify({'success': False, 'error': 'File too large'}), 413

# Initialize database
def create_tables():
    with app.app_context():
        db.create_all()
        
        # Create default user if none exists
        if not User.query.first():
            default_user = User(username='demo_user', email='demo@example.com')
            db.session.add(default_user)
            db.session.commit()
            print("Created default user")

if __name__ == '__main__':
    create_tables()
    print("Starting AI Presentation Coach Backend...")
    print("Available endpoints:")
    print("  GET  / - API status")
    print("  POST /api/upload_script - Upload presentation script")
    print("  POST /api/start_session - Start practice session")
    print("  POST /api/analyze_frame - Analyze video frame")
    print("  POST /api/analyze_audio - Analyze audio")
    print("  POST /api/end_session - End session and get report")
    print("  GET  /api/user_progress/<user_id> - Get user progress")
    print("  GET  /api/scripts/<user_id> - Get user scripts")
    print("  GET  /api/session_details/<session_id> - Get session details")
    
    app.run(debug=True, host='0.0.0.0', port=5000)