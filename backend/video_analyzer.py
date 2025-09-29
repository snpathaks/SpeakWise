import cv2
import numpy as np
import mediapipe as mp
import time
from typing import Dict, List, Tuple, Optional
import logging

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class VideoAnalyzer:
    """
    Video analysis class using OpenCV and MediaPipe for:
    - Facial expression analysis
    - Eye contact detection
    - Gesture recognition
    - Posture analysis
    """
    
    def __init__(self):
        """Initialize MediaPipe solutions and analysis parameters"""
        try:
            # Initialize MediaPipe solutions
            self.mp_face_mesh = mp.solutions.face_mesh
            self.mp_hands = mp.solutions.hands
            self.mp_pose = mp.solutions.pose
            self.mp_drawing = mp.solutions.drawing_utils
            
            # Initialize face mesh for facial analysis
            self.face_mesh = self.mp_face_mesh.FaceMesh(
                max_num_faces=1,
                refine_landmarks=True,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Initialize hands for gesture analysis
            self.hands = self.mp_hands.Hands(
                static_image_mode=False,
                max_num_hands=2,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Initialize pose for posture analysis
            self.pose = self.mp_pose.Pose(
                static_image_mode=False,
                model_complexity=1,
                min_detection_confidence=0.5,
                min_tracking_confidence=0.5
            )
            
            # Analysis history for smoothing
            self.analysis_history = []
            self.max_history = 10
            
            logger.info("VideoAnalyzer initialized successfully")
            
        except Exception as e:
            logger.error(f"Error initializing VideoAnalyzer: {e}")
            raise
    
    def analyze_frame(self, frame: np.ndarray) -> Dict:
        """
        Analyze a single video frame for all presentation metrics
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Convert BGR to RGB for MediaPipe
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, _ = frame.shape
            
            # Initialize results dictionary
            results = {
                'timestamp': time.time(),
                'frame_quality': self._assess_frame_quality(frame),
                'face_detected': False,
                'eye_contact_score': 0.0,
                'facial_expression': 'neutral',
                'expression_confidence': 0.0,
                'gesture_activity': 0.0,
                'gesture_type': 'none',
                'posture_score': 0.0,
                'head_position': 'center',
                'body_alignment': 0.0,
                'engagement_level': 0.0
            }
            
            # Face analysis
            face_results = self.face_mesh.process(rgb_frame)
            if face_results.multi_face_landmarks:
                results['face_detected'] = True
                landmarks = face_results.multi_face_landmarks[0]
                
                # Analyze facial features
                eye_contact = self._calculate_eye_contact(landmarks, w, h)
                expression = self._analyze_facial_expression(landmarks)
                head_pos = self._analyze_head_position(landmarks, w, h)
                
                results.update({
                    'eye_contact_score': eye_contact['score'],
                    'gaze_direction': eye_contact['direction'],
                    'facial_expression': expression['type'],
                    'expression_confidence': expression['confidence'],
                    'head_position': head_pos['position'],
                    'head_stability': head_pos['stability']
                })
            
            # Hand gesture analysis
            hand_results = self.hands.process(rgb_frame)
            if hand_results.multi_hand_landmarks:
                gesture_data = self._analyze_gestures(hand_results.multi_hand_landmarks, w, h)
                results.update({
                    'gesture_activity': gesture_data['activity_level'],
                    'gesture_type': gesture_data['type'],
                    'gesture_appropriateness': gesture_data['appropriateness'],
                    'hand_count': len(hand_results.multi_hand_landmarks)
                })
            
            # Pose and posture analysis
            pose_results = self.pose.process(rgb_frame)
            if pose_results.pose_landmarks:
                posture_data = self._analyze_posture(pose_results.pose_landmarks, w, h)
                results.update({
                    'posture_score': posture_data['score'],
                    'body_alignment': posture_data['alignment'],
                    'shoulder_level': posture_data['shoulder_level'],
                    'stance_stability': posture_data['stability']
                })
            
            # Calculate overall engagement level
            results['engagement_level'] = self._calculate_engagement_level(results)
            
            # Store in history for smoothing
            self._update_history(results)
            
            # Apply smoothing to reduce jitter
            smoothed_results = self._apply_smoothing(results)
            
            return smoothed_results
            
        except Exception as e:
            logger.error(f"Error analyzing frame: {e}")
            return {
                'error': str(e),
                'timestamp': time.time(),
                'frame_quality': 0.0,
                'face_detected': False,
                'eye_contact_score': 0.0,
                'engagement_level': 0.0
            }
    
    def _assess_frame_quality(self, frame: np.ndarray) -> float:
        """Assess the quality of the input frame"""
        try:
            # Check brightness
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            brightness = np.mean(gray)
            
            # Check blur using Laplacian variance
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Normalize scores (0-1)
            brightness_score = min(1.0, brightness / 128.0)
            blur_score = min(1.0, blur_score / 1000.0)
            
            # Combined quality score
            quality = (brightness_score + blur_score) / 2.0
            return max(0.0, min(1.0, quality))
            
        except Exception as e:
            logger.error(f"Error assessing frame quality: {e}")
            return 0.5
    
    def _calculate_eye_contact(self, landmarks, width: int, height: int) -> Dict:
        """Calculate eye contact score based on gaze direction"""
        try:
            # Get eye landmarks (simplified approach)
            left_eye_indices = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
            right_eye_indices = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385, 384, 398]
            
            # Calculate eye centers
            left_eye_points = [(landmarks.landmark[i].x * width, landmarks.landmark[i].y * height) 
                              for i in left_eye_indices]
            right_eye_points = [(landmarks.landmark[i].x * width, landmarks.landmark[i].y * height) 
                               for i in right_eye_indices]
            
            left_eye_center = np.mean(left_eye_points, axis=0)
            right_eye_center = np.mean(right_eye_points, axis=0)
            
            # Calculate gaze direction (simplified)
            eye_center = (left_eye_center + right_eye_center) / 2
            frame_center = np.array([width/2, height/2])
            
            # Distance from center (normalized)
            distance = np.linalg.norm(eye_center - frame_center)
            max_distance = np.linalg.norm([width/2, height/2])
            normalized_distance = distance / max_distance
            
            # Eye contact score (closer to center = higher score)
            eye_contact_score = max(0.0, 1.0 - normalized_distance)
            
            # Determine gaze direction
            dx = eye_center[0] - frame_center[0]
            dy = eye_center[1] - frame_center[1]
            
            if abs(dx) < width * 0.1 and abs(dy) < height * 0.1:
                direction = 'center'
            elif dx > width * 0.1:
                direction = 'right'
            elif dx < -width * 0.1:
                direction = 'left'
            elif dy > height * 0.1:
                direction = 'down'
            else:
                direction = 'up'
            
            return {
                'score': eye_contact_score,
                'direction': direction,
                'eye_center': eye_center.tolist()
            }
            
        except Exception as e:
            logger.error(f"Error calculating eye contact: {e}")
            return {'score': 0.5, 'direction': 'unknown', 'eye_center': [0, 0]}
    
    def _analyze_facial_expression(self, landmarks) -> Dict:
        """Analyze facial expression from landmarks"""
        try:
            # Key facial landmarks for expression analysis
            # Mouth landmarks
            mouth_landmarks = [61, 84, 17, 314, 405, 320, 307, 375, 321, 308, 324, 318]
            mouth_points = [(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in mouth_landmarks]
            
            # Eyebrow landmarks
            left_eyebrow = [70, 63, 105, 66, 107]
            right_eyebrow = [296, 334, 293, 300, 276]
            
            left_brow_points = [(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in left_eyebrow]
            right_brow_points = [(landmarks.landmark[i].x, landmarks.landmark[i].y) for i in right_eyebrow]
            
            # Calculate mouth curvature (smile detection)
            mouth_left = np.array([landmarks.landmark[61].x, landmarks.landmark[61].y])
            mouth_right = np.array([landmarks.landmark[291].x, landmarks.landmark[291].y])
            mouth_top = np.array([landmarks.landmark[13].x, landmarks.landmark[13].y])
            mouth_bottom = np.array([landmarks.landmark[14].x, landmarks.landmark[14].y])
            
            # Mouth aspect ratio
            mouth_width = np.linalg.norm(mouth_right - mouth_left)
            mouth_height = np.linalg.norm(mouth_bottom - mouth_top)
            mouth_ratio = mouth_height / (mouth_width + 1e-6)
            
            # Eyebrow position analysis
            left_brow_center = np.mean(left_brow_points, axis=0)
            right_brow_center = np.mean(right_brow_points, axis=0)
            
            # Simple expression classification
            if mouth_ratio < 0.3:
                if left_brow_center[1] < 0.4:  # High eyebrows
                    expression_type = 'surprised'
                    confidence = 0.7
                else:
                    expression_type = 'happy'
                    confidence = 0.8
            elif mouth_ratio > 0.5:
                expression_type = 'speaking'
                confidence = 0.6
            else:
                if left_brow_center[1] < 0.35:  # Very high eyebrows
                    expression_type = 'concerned'
                    confidence = 0.6
                else:
                    expression_type = 'neutral'
                    confidence = 0.9
            
            return {
                'type': expression_type,
                'confidence': confidence,
                'mouth_ratio': mouth_ratio
            }
            
        except Exception as e:
            logger.error(f"Error analyzing facial expression: {e}")
            return {'type': 'neutral', 'confidence': 0.5, 'mouth_ratio': 0.4}
    
    def _analyze_head_position(self, landmarks, width: int, height: int) -> Dict:
        """Analyze head position and stability"""
        try:
            # Key landmarks for head position
            nose_tip = landmarks.landmark[1]
            chin = landmarks.landmark[175]
            forehead = landmarks.landmark[10]
            
            # Calculate head center
            head_center_x = nose_tip.x * width
            head_center_y = nose_tip.y * height
            
            # Determine position relative to frame center
            frame_center_x, frame_center_y = width/2, height/2
            
            dx = head_center_x - frame_center_x
            dy = head_center_y - frame_center_y
            
            # Position classification
            if abs(dx) < width * 0.15 and abs(dy) < height * 0.15:
                position = 'center'
                stability = 0.9
            elif abs(dx) < width * 0.3 and abs(dy) < height * 0.3:
                position = 'slightly_off_center'
                stability = 0.7
            else:
                position = 'off_center'
                stability = 0.4
            
            return {
                'position': position,
                'stability': stability,
                'center': [head_center_x, head_center_y]
            }
            
        except Exception as e:
            logger.error(f"Error analyzing head position: {e}")
            return {'position': 'unknown', 'stability': 0.5, 'center': [0, 0]}
    
    def _analyze_gestures(self, hand_landmarks: List, width: int, height: int) -> Dict:
        """Analyze hand gestures and movement"""
        try:
            total_activity = 0.0
            gesture_types = []
            appropriateness_scores = []
            
            for hand_landmark in hand_landmarks:
                # Calculate hand bounding box
                hand_points = [(lm.x * width, lm.y * height) for lm in hand_landmark.landmark]
                hand_array = np.array(hand_points)
                
                # Calculate hand size (for gesture emphasis)
                hand_bbox = cv2.boundingRect(hand_array.astype(np.float32))
                hand_area = hand_bbox[2] * hand_bbox[3]
                
                # Normalize hand area relative to frame
                normalized_area = hand_area / (width * height)
                
                # Calculate finger positions for gesture recognition
                finger_tips = [4, 8, 12, 16, 20]  # Thumb, Index, Middle, Ring, Pinky
                finger_positions = [hand_landmark.landmark[i] for i in finger_tips]
                
                # Simple gesture classification
                gesture_type = self._classify_gesture(finger_positions)
                gesture_types.append(gesture_type)
                
                # Activity level based on hand movement and size
                activity_level = min(1.0, normalized_area * 100)
                total_activity += activity_level
                
                # Appropriateness score (gestures within reasonable bounds)
                if 0.2 < hand_landmark.landmark[0].y < 0.8:  # Reasonable vertical position
                    appropriateness = 0.8
                else:
                    appropriateness = 0.4
                appropriateness_scores.append(appropriateness)
            
            # Average scores
            avg_activity = total_activity / len(hand_landmarks) if hand_landmarks else 0.0
            avg_appropriateness = np.mean(appropriateness_scores) if appropriateness_scores else 0.5
            dominant_gesture = max(set(gesture_types), key=gesture_types.count) if gesture_types else 'none'
            
            return {
                'activity_level': avg_activity,
                'type': dominant_gesture,
                'appropriateness': avg_appropriateness,
                'hand_count': len(hand_landmarks)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing gestures: {e}")
            return {'activity_level': 0.0, 'type': 'none', 'appropriateness': 0.5, 'hand_count': 0}
    
    def _classify_gesture(self, finger_positions) -> str:
        """Simple gesture classification based on finger positions"""
        try:
            # This is a simplified gesture classification
            # In a real implementation, you'd use more sophisticated methods
            
            # Count extended fingers (simplified)
            extended_fingers = 0
            for i in range(1, 5):  # Skip thumb for simplicity
                if finger_positions[i].y < finger_positions[i-1].y:
                    extended_fingers += 1
            
            # Basic gesture classification
            if extended_fingers == 0:
                return 'closed_fist'
            elif extended_fingers == 1:
                return 'pointing'
            elif extended_fingers == 2:
                return 'peace_or_two'
            elif extended_fingers >= 3:
                return 'open_hand'
            else:
                return 'natural'
                
        except Exception as e:
            logger.error(f"Error classifying gesture: {e}")
            return 'unknown'
    
    def _analyze_posture(self, pose_landmarks, width: int, height: int) -> Dict:
        """Analyze body posture and alignment"""
        try:
            # Key pose landmarks
            left_shoulder = pose_landmarks.landmark[11]
            right_shoulder = pose_landmarks.landmark[12]
            left_hip = pose_landmarks.landmark[23]
            right_hip = pose_landmarks.landmark[24]
            
            # Calculate shoulder and hip centers
            shoulder_center = np.array([
                (left_shoulder.x + right_shoulder.x) / 2 * width,
                (left_shoulder.y + right_shoulder.y) / 2 * height
            ])
            
            hip_center = np.array([
                (left_hip.x + right_hip.x) / 2 * width,
                (left_hip.y + right_hip.y) / 2 * height
            ])
            
            # Body alignment (vertical alignment of shoulders and hips)
            alignment_offset = abs(shoulder_center[0] - hip_center[0])
            max_offset = width * 0.1  # 10% of frame width
            alignment_score = max(0.0, 1.0 - (alignment_offset / max_offset))
            
            # Shoulder level (horizontal alignment)
            shoulder_diff = abs(left_shoulder.y - right_shoulder.y) * height
            max_shoulder_diff = height * 0.05  # 5% of frame height
            shoulder_level_score = max(0.0, 1.0 - (shoulder_diff / max_shoulder_diff))
            
            # Overall posture score
            posture_score = (alignment_score + shoulder_level_score) / 2.0
            
            # Stability assessment
            stability = 0.8 if posture_score > 0.7 else 0.5
            
            return {
                'score': posture_score,
                'alignment': alignment_score,
                'shoulder_level': shoulder_level_score,
                'stability': stability
            }
            
        except Exception as e:
            logger.error(f"Error analyzing posture: {e}")
            return {'score': 0.5, 'alignment': 0.5, 'shoulder_level': 0.5, 'stability': 0.5}
    
    def _calculate_engagement_level(self, results: Dict) -> float:
        """Calculate overall engagement level based on all metrics"""
        try:
            factors = []
            
            # Face presence and quality
            if results['face_detected']:
                factors.append(0.8)
            else:
                factors.append(0.2)
            
            # Eye contact contribution
            factors.append(results['eye_contact_score'])
            
            # Gesture activity (moderate activity is best)
            gesture_score = results['gesture_activity']
            if 0.3 <= gesture_score <= 0.7:
                factors.append(0.8)
            elif gesture_score > 0:
                factors.append(0.6)
            else:
                factors.append(0.3)
            
            # Posture contribution
            factors.append(results['posture_score'])
            
            # Expression positivity
            expression = results['facial_expression']
            if expression in ['happy', 'engaged']:
                factors.append(0.9)
            elif expression in ['neutral', 'speaking']:
                factors.append(0.7)
            else:
                factors.append(0.5)
            
            # Calculate weighted average
            engagement_level = np.mean(factors)
            return max(0.0, min(1.0, engagement_level))
            
        except Exception as e:
            logger.error(f"Error calculating engagement level: {e}")
            return 0.5
    
    def _update_history(self, results: Dict):
        """Update analysis history for smoothing"""
        self.analysis_history.append(results)
        if len(self.analysis_history) > self.max_history:
            self.analysis_history.pop(0)
    
    def _apply_smoothing(self, current_results: Dict) -> Dict:
        """Apply temporal smoothing to reduce jitter in results"""
        if len(self.analysis_history) < 3:
            return current_results
        
        try:
            # Keys to smooth
            smooth_keys = ['eye_contact_score', 'gesture_activity', 'posture_score', 
                          'engagement_level', 'expression_confidence']
            
            smoothed = current_results.copy()
            
            for key in smooth_keys:
                if key in current_results:
                    # Get recent values
                    recent_values = [h[key] for h in self.analysis_history[-5:] if key in h]
                    if recent_values:
                        # Apply weighted average (more recent values have higher weight)
                        weights = np.exp(np.linspace(0, 1, len(recent_values)))
                        smoothed_value = np.average(recent_values, weights=weights)
                        smoothed[key] = float(smoothed_value)
            
            return smoothed
            
        except Exception as e:
            logger.error(f"Error applying smoothing: {e}")
            return current_results
    
    def get_analysis_summary(self) -> Dict:
        """Get summary of recent analysis results"""
        if not self.analysis_history:
            return {'error': 'No analysis history available'}
        
        try:
            recent_results = self.analysis_history[-10:]  # Last 10 frames
            
            summary = {
                'avg_eye_contact': np.mean([r.get('eye_contact_score', 0) for r in recent_results]),
                'avg_engagement': np.mean([r.get('engagement_level', 0) for r in recent_results]),
                'avg_posture': np.mean([r.get('posture_score', 0) for r in recent_results]),
                'avg_gesture_activity': np.mean([r.get('gesture_activity', 0) for r in recent_results]),
                'face_detection_rate': sum(1 for r in recent_results if r.get('face_detected', False)) / len(recent_results),
                'dominant_expression': self._get_dominant_expression(recent_results),
                'analysis_count': len(recent_results)
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating analysis summary: {e}")
            return {'error': str(e)}
    
    def _get_dominant_expression(self, results: List[Dict]) -> str:
        """Get the most common expression from recent results"""
        expressions = [r.get('facial_expression', 'neutral') for r in results]
        if expressions:
            return max(set(expressions), key=expressions.count)
        return 'neutral'
    
    def reset_analysis(self):
        """Reset analysis history"""
        self.analysis_history = []
        logger.info("Analysis history reset")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            if hasattr(self, 'face_mesh'):
                self.face_mesh.close()
            if hasattr(self, 'hands'):
                self.hands.close()
            if hasattr(self, 'pose'):
                self.pose.close()
            logger.info("VideoAnalyzer cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during cleanup: {e}")

# Utility function for testing
def test_video_analyzer():
    """Test the VideoAnalyzer with webcam"""
    analyzer = VideoAnalyzer()
    cap = cv2.VideoCapture(0)
    
    print("Press 'q' to quit, 's' to show summary")
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Analyze frame
            results = analyzer.analyze_frame(frame)
            
            # Display results on frame
            cv2.putText(frame, f"Eye Contact: {results['eye_contact_score']:.2f}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Engagement: {results['engagement_level']:.2f}", 
                       (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Expression: {results['facial_expression']}", 
                       (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            cv2.imshow('Video Analysis Test', frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                summary = analyzer.get_analysis_summary()
                print("Analysis Summary:", summary)
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        analyzer.cleanup()

if __name__ == "__main__":
    test_video_analyzer()