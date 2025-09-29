from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
from sqlalchemy import Text, Float, Integer, String, DateTime, ForeignKey
from sqlalchemy.orm import relationship

# Initialize SQLAlchemy
db = SQLAlchemy()

class User(db.Model):
    """User model for storing user information"""
    __tablename__ = 'users'
    
    id = db.Column(Integer, primary_key=True)
    username = db.Column(String(80), unique=True, nullable=False, index=True)
    email = db.Column(String(120), unique=True, nullable=False, index=True)
    created_at = db.Column(DateTime, default=datetime.utcnow, nullable=False)
    last_login = db.Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    scripts = relationship('Script', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    sessions = relationship('PracticeSession', backref='user', lazy='dynamic', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<User {self.username}>'
    
    def to_dict(self):
        """Convert user object to dictionary"""
        return {
            'id': self.id,
            'username': self.username,
            'email': self.email,
            'created_at': self.created_at.isoformat() if self.created_at else None,
            'last_login': self.last_login.isoformat() if self.last_login else None,
            'total_scripts': self.scripts.count(),
            'total_sessions': self.sessions.count()
        }


class Script(db.Model):
    """Model for storing presentation scripts"""
    __tablename__ = 'scripts'
    
    id = db.Column(Integer, primary_key=True)
    user_id = db.Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    title = db.Column(String(200), nullable=False)
    content = db.Column(Text, nullable=False)
    category = db.Column(String(50), default='general', index=True)
    filename = db.Column(String(255), nullable=True)
    uploaded_at = db.Column(DateTime, default=datetime.utcnow, nullable=False)
    updated_at = db.Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Relationships
    sessions = relationship('PracticeSession', backref='script', lazy='dynamic', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<Script {self.title}>'
    
    def to_dict(self, include_content=False):
        """Convert script object to dictionary"""
        data = {
            'id': self.id,
            'user_id': self.user_id,
            'title': self.title,
            'category': self.category,
            'filename': self.filename,
            'uploaded_at': self.uploaded_at.isoformat() if self.uploaded_at else None,
            'updated_at': self.updated_at.isoformat() if self.updated_at else None,
            'word_count': len(self.content.split()) if self.content else 0,
            'character_count': len(self.content) if self.content else 0,
            'session_count': self.sessions.count()
        }
        
        if include_content:
            data['content'] = self.content
        else:
            # Provide a preview
            data['content_preview'] = self.content[:200] + '...' if len(self.content) > 200 else self.content
        
        return data


class PracticeSession(db.Model):
    """Model for storing practice session data"""
    __tablename__ = 'practice_sessions'
    
    id = db.Column(Integer, primary_key=True)
    user_id = db.Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    script_id = db.Column(Integer, ForeignKey('scripts.id'), nullable=False, index=True)
    session_date = db.Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    duration = db.Column(Integer, default=0)  # Duration in seconds
    
    # Performance scores (0-100)
    overall_score = db.Column(Float, default=0.0)
    confidence_score = db.Column(Float, default=0.0)
    clarity_score = db.Column(Float, default=0.0)
    pace_score = db.Column(Float, default=0.0)
    gesture_score = db.Column(Float, default=0.0)
    eye_contact_score = db.Column(Float, default=0.0)
    
    # Additional metrics
    filler_word_count = db.Column(Integer, default=0)
    feedback_data = db.Column(Text, nullable=True)  # JSON string with detailed feedback
    
    # Relationships
    feedback_reports = relationship('FeedbackReport', backref='session', lazy='dynamic', cascade='all, delete-orphan')
    
    def __repr__(self):
        return f'<PracticeSession {self.id} - User {self.user_id}>'
    
    def to_dict(self, include_feedback_data=False):
        """Convert session object to dictionary"""
        data = {
            'id': self.id,
            'user_id': self.user_id,
            'script_id': self.script_id,
            'session_date': self.session_date.isoformat() if self.session_date else None,
            'duration': self.duration,
            'duration_formatted': self._format_duration(self.duration),
            'scores': {
                'overall': round(self.overall_score, 2),
                'confidence': round(self.confidence_score, 2),
                'clarity': round(self.clarity_score, 2),
                'pace': round(self.pace_score, 2),
                'gesture': round(self.gesture_score, 2),
                'eye_contact': round(self.eye_contact_score, 2)
            },
            'filler_word_count': self.filler_word_count,
            'feedback_report_count': self.feedback_reports.count()
        }
        
        if include_feedback_data and self.feedback_data:
            data['feedback_data'] = self.feedback_data
        
        return data
    
    @staticmethod
    def _format_duration(seconds):
        """Format duration in seconds to HH:MM:SS or MM:SS"""
        if seconds < 0:
            return "00:00"
        
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        secs = seconds % 60
        
        if hours > 0:
            return f"{hours:02d}:{minutes:02d}:{secs:02d}"
        else:
            return f"{minutes:02d}:{secs:02d}"
    
    def get_performance_level(self):
        """Get overall performance level category"""
        score = self.overall_score
        if score >= 90:
            return 'excellent'
        elif score >= 80:
            return 'very_good'
        elif score >= 70:
            return 'good'
        elif score >= 60:
            return 'fair'
        else:
            return 'needs_improvement'


class FeedbackReport(db.Model):
    """Model for storing AI-generated feedback reports"""
    __tablename__ = 'feedback_reports'
    
    id = db.Column(Integer, primary_key=True)
    session_id = db.Column(Integer, ForeignKey('practice_sessions.id'), nullable=False, index=True)
    feedback_type = db.Column(String(50), default='comprehensive')  # comprehensive, quick, detailed
    feedback_text = db.Column(Text, nullable=False)  # Main feedback content (JSON or text)
    improvement_suggestions = db.Column(Text, nullable=True)  # Specific improvement tips (JSON or text)
    strengths = db.Column(Text, nullable=True)  # Areas of strength (JSON or text)
    weaknesses = db.Column(Text, nullable=True)  # Areas needing improvement (JSON or text)
    created_at = db.Column(DateTime, default=datetime.utcnow, nullable=False)
    
    def __repr__(self):
        return f'<FeedbackReport {self.id} - Session {self.session_id}>'
    
    def to_dict(self):
        """Convert feedback report object to dictionary"""
        return {
            'id': self.id,
            'session_id': self.session_id,
            'feedback_type': self.feedback_type,
            'feedback_text': self.feedback_text,
            'improvement_suggestions': self.improvement_suggestions,
            'strengths': self.strengths,
            'weaknesses': self.weaknesses,
            'created_at': self.created_at.isoformat() if self.created_at else None
        }


class UserProgress(db.Model):
    """Model for tracking user progress metrics over time"""
    __tablename__ = 'user_progress'
    
    id = db.Column(Integer, primary_key=True)
    user_id = db.Column(Integer, ForeignKey('users.id'), nullable=False, index=True)
    recorded_at = db.Column(DateTime, default=datetime.utcnow, nullable=False, index=True)
    
    # Aggregate metrics
    total_practice_time = db.Column(Integer, default=0)  # Total seconds practiced
    total_sessions = db.Column(Integer, default=0)
    average_overall_score = db.Column(Float, default=0.0)
    average_confidence_score = db.Column(Float, default=0.0)
    average_clarity_score = db.Column(Float, default=0.0)
    average_pace_score = db.Column(Float, default=0.0)
    
    # Progress indicators
    improvement_rate = db.Column(Float, default=0.0)  # Rate of improvement
    consistency_score = db.Column(Float, default=0.0)  # How consistently user practices
    
    def __repr__(self):
        return f'<UserProgress {self.id} - User {self.user_id}>'
    
    def to_dict(self):
        """Convert progress object to dictionary"""
        return {
            'id': self.id,
            'user_id': self.user_id,
            'recorded_at': self.recorded_at.isoformat() if self.recorded_at else None,
            'total_practice_time': self.total_practice_time,
            'total_practice_time_formatted': PracticeSession._format_duration(self.total_practice_time),
            'total_sessions': self.total_sessions,
            'average_scores': {
                'overall': round(self.average_overall_score, 2),
                'confidence': round(self.average_confidence_score, 2),
                'clarity': round(self.average_clarity_score, 2),
                'pace': round(self.average_pace_score, 2)
            },
            'improvement_rate': round(self.improvement_rate, 2),
            'consistency_score': round(self.consistency_score, 2)
        }


# Utility functions for database operations
def create_default_user(username='demo_user', email='demo@example.com'):
    """Create a default user if none exists"""
    existing_user = User.query.filter_by(username=username).first()
    if not existing_user:
        user = User(username=username, email=email)
        db.session.add(user)
        db.session.commit()
        return user
    return existing_user


def get_user_statistics(user_id):
    """Get comprehensive statistics for a user"""
    user = User.query.get(user_id)
    if not user:
        return None
    
    sessions = PracticeSession.query.filter_by(user_id=user_id).all()
    
    if not sessions:
        return {
            'user_id': user_id,
            'total_sessions': 0,
            'total_practice_time': 0,
            'average_scores': {},
            'recent_trend': 'no_data'
        }
    
    total_time = sum(s.duration for s in sessions)
    avg_overall = sum(s.overall_score for s in sessions) / len(sessions)
    avg_confidence = sum(s.confidence_score for s in sessions) / len(sessions)
    avg_clarity = sum(s.clarity_score for s in sessions) / len(sessions)
    avg_pace = sum(s.pace_score for s in sessions) / len(sessions)
    
    # Calculate trend (comparing recent vs older sessions)
    recent_trend = 'stable'
    if len(sessions) >= 6:
        recent_avg = sum(s.overall_score for s in sessions[-3:]) / 3
        older_avg = sum(s.overall_score for s in sessions[-6:-3]) / 3
        if recent_avg > older_avg + 5:
            recent_trend = 'improving'
        elif recent_avg < older_avg - 5:
            recent_trend = 'declining'
    
    return {
        'user_id': user_id,
        'username': user.username,
        'total_sessions': len(sessions),
        'total_practice_time': total_time,
        'total_practice_time_formatted': PracticeSession._format_duration(total_time),
        'average_scores': {
            'overall': round(avg_overall, 2),
            'confidence': round(avg_confidence, 2),
            'clarity': round(avg_clarity, 2),
            'pace': round(avg_pace, 2)
        },
        'recent_trend': recent_trend,
        'total_filler_words': sum(s.filler_word_count for s in sessions),
        'most_practiced_script': get_most_practiced_script(user_id)
    }


def get_most_practiced_script(user_id):
    """Get the script that has been practiced most by a user"""
    from sqlalchemy import func
    
    result = db.session.query(
        Script.id, 
        Script.title, 
        func.count(PracticeSession.id).label('session_count')
    ).join(
        PracticeSession, Script.id == PracticeSession.script_id
    ).filter(
        Script.user_id == user_id
    ).group_by(
        Script.id, Script.title
    ).order_by(
        func.count(PracticeSession.id).desc()
    ).first()
    
    if result:
        return {
            'script_id': result.id,
            'title': result.title,
            'practice_count': result.session_count
        }
    return None


def cleanup_old_sessions(days=90):
    """Delete sessions older than specified days"""
    from datetime import timedelta
    
    cutoff_date = datetime.utcnow() - timedelta(days=days)
    old_sessions = PracticeSession.query.filter(
        PracticeSession.session_date < cutoff_date
    ).all()
    
    count = len(old_sessions)
    for session in old_sessions:
        db.session.delete(session)
    
    db.session.commit()
    return count