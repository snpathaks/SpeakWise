import librosa
import numpy as np
import whisper
import soundfile as sf
import os
import time
from typing import Dict, List, Optional, Tuple
import logging
import re
from collections import defaultdict
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AudioAnalyzer:
    """
    Audio analysis class using Whisper and librosa for:
    - Speech-to-text transcription
    - Filler word detection
    - Voice quality analysis (pitch, pace, clarity)
    - Speaking pattern analysis
    """
    
    def __init__(self, model_size: str = "base"):
        """
        Initialize AudioAnalyzer with Whisper model
        
        Args:
            model_size: Whisper model size ("tiny", "base", "small", "medium", "large")
        """
        self.model_size = model_size
        self.whisper_model = None
        self.filler_words = [
            'um', 'uh', 'uhm', 'er', 'eh', 'ah', 'oh',
            'like', 'you know', 'so', 'actually', 'basically',
            'literally', 'totally', 'really', 'very',
            'kind of', 'sort of', 'i mean', 'well',
            'okay', 'alright', 'right', 'yeah', 'yes'
        ]
        
        # Analysis history for pattern detection
        self.analysis_history = []
        self.max_history = 50
        
        # Voice characteristics baseline (updated during analysis)
        self.voice_baseline = {
            'average_pitch': None,
            'pitch_range': None,
            'average_pace': None,
            'volume_baseline': None
        }
        
        logger.info(f"AudioAnalyzer initialized with model size: {model_size}")
    
    def _get_whisper_model(self):
        """Lazy loading of Whisper model to improve startup time"""
        if self.whisper_model is None:
            logger.info(f"Loading Whisper model: {self.model_size}")
            try:
                self.whisper_model = whisper.load_model(self.model_size)
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading Whisper model: {e}")
                raise
        return self.whisper_model
    
    def analyze_audio_file(self, audio_file_path: str) -> Dict:
        """
        Analyze audio file for all speech metrics
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            if not os.path.exists(audio_file_path):
                raise FileNotFoundError(f"Audio file not found: {audio_file_path}")
            
            # Load audio file
            audio_data, sample_rate = librosa.load(audio_file_path, sr=22050)
            
            if len(audio_data) == 0:
                raise ValueError("Audio file is empty or corrupted")
            
            # Perform analysis
            results = self.analyze_audio_data(audio_data, sample_rate, audio_file_path)
            
            return results
            
        except Exception as e:
            logger.error(f"Error analyzing audio file: {e}")
            return {
                'error': str(e),
                'timestamp': time.time(),
                'transcription': '',
                'analysis_successful': False
            }
    
    def analyze_audio_data(self, audio_data: np.ndarray, sample_rate: int, 
                          file_path: Optional[str] = None) -> Dict:
        """
        Analyze audio data for speech metrics
        
        Args:
            audio_data: Audio data array
            sample_rate: Sample rate of audio
            file_path: Optional path to temporary file for Whisper
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            start_time = time.time()
            
            # Initialize results
            results = {
                'timestamp': start_time,
                'duration': len(audio_data) / sample_rate,
                'sample_rate': sample_rate,
                'analysis_successful': True,
                'transcription': '',
                'confidence_score': 0.0,
                'filler_word_count': 0,
                'filler_word_rate': 0.0,
                'speaking_rate': 0.0,  # words per minute
                'pause_analysis': {},
                'pitch_analysis': {},
                'volume_analysis': {},
                'clarity_score': 0.0,
                'voice_stability': 0.0,
                'energy_levels': {},
                'speech_segments': []
            }
            
            # Basic audio quality checks
            if np.max(np.abs(audio_data)) < 0.01:
                logger.warning("Audio level very low, analysis may be inaccurate")
                results['volume_warning'] = 'Audio level too low'
            
            # Voice Activity Detection (VAD)
            speech_segments = self._detect_speech_segments(audio_data, sample_rate)
            results['speech_segments'] = speech_segments
            
            # Transcription using Whisper
            transcription_data = self._transcribe_audio(audio_data, sample_rate, file_path)
            results.update(transcription_data)
            
            # Filler word analysis
            if results['transcription']:
                filler_analysis = self._analyze_filler_words(results['transcription'])
                results.update(filler_analysis)
            
            # Voice quality analysis
            pitch_analysis = self._analyze_pitch(audio_data, sample_rate)
            results['pitch_analysis'] = pitch_analysis
            
            # Volume and energy analysis
            volume_analysis = self._analyze_volume(audio_data, sample_rate)
            results['volume_analysis'] = volume_analysis
            
            # Speaking rate analysis
            speaking_rate_data = self._analyze_speaking_rate(
                results['transcription'], results['duration']
            )
            results.update(speaking_rate_data)
            
            # Pause analysis
            pause_analysis = self._analyze_pauses(audio_data, sample_rate)
            results['pause_analysis'] = pause_analysis
            
            # Overall clarity score
            clarity_score = self._calculate_clarity_score(results)
            results['clarity_score'] = clarity_score
            
            # Voice stability
            stability_score = self._calculate_voice_stability(results)
            results['voice_stability'] = stability_score
            
            # Update analysis history
            self._update_analysis_history(results)
            
            # Update voice baseline
            self._update_voice_baseline(results)
            
            processing_time = time.time() - start_time
            results['processing_time'] = processing_time
            
            logger.info(f"Audio analysis completed in {processing_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in audio analysis: {e}")
            return {
                'error': str(e),
                'timestamp': time.time(),
                'analysis_successful': False,
                'transcription': '',
                'clarity_score': 0.0
            }
    
    def _detect_speech_segments(self, audio_data: np.ndarray, sample_rate: int) -> List[Dict]:
        """Detect speech segments in audio using energy-based VAD"""
        try:
            # Frame settings
            frame_length = int(0.025 * sample_rate)  # 25ms frames
            hop_length = int(0.01 * sample_rate)     # 10ms hop
            
            # Compute RMS energy
            rms_energy = librosa.feature.rms(
                y=audio_data, 
                frame_length=frame_length, 
                hop_length=hop_length
            )[0]
            
            # Threshold for speech detection
            energy_threshold = np.percentile(rms_energy, 30)  # Adaptive threshold
            
            # Find speech frames
            speech_frames = rms_energy > energy_threshold
            
            # Convert to time segments
            frame_times = librosa.frames_to_time(
                np.arange(len(speech_frames)), 
                sr=sample_rate, 
                hop_length=hop_length
            )
            
            # Group consecutive speech frames into segments
            segments = []
            start_time = None
            
            for i, is_speech in enumerate(speech_frames):
                if is_speech and start_time is None:
                    start_time = frame_times[i]
                elif not is_speech and start_time is not None:
                    segments.append({
                        'start': start_time,
                        'end': frame_times[i],
                        'duration': frame_times[i] - start_time
                    })
                    start_time = None
            
            # Handle case where speech continues to end
            if start_time is not None:
                segments.append({
                    'start': start_time,
                    'end': frame_times[-1],
                    'duration': frame_times[-1] - start_time
                })
            
            return segments
            
        except Exception as e:
            logger.error(f"Error in speech segment detection: {e}")
            return []
    
    def _transcribe_audio(self, audio_data: np.ndarray, sample_rate: int, 
                         file_path: Optional[str] = None) -> Dict:
        """Transcribe audio using Whisper"""
        try:
            model = self._get_whisper_model()
            
            # Create temporary file if not provided
            temp_file_created = False
            if file_path is None:
                file_path = f"temp_whisper_{int(time.time())}.wav"
                sf.write(file_path, audio_data, sample_rate)
                temp_file_created = True
            
            # Transcribe with Whisper
            result = model.transcribe(
                file_path,
                language="en",  # Can be made configurable
                task="transcribe",
                verbose=False
            )
            
            # Clean up temporary file
            if temp_file_created:
                try:
                    os.remove(file_path)
                except:
                    pass
            
            transcription = result["text"].strip()
            segments = result.get("segments", [])
            
            # Calculate confidence score from segments
            if segments:
                # Whisper doesn't provide confidence in older versions
                # Use segment count and length as proxy
                avg_segment_length = np.mean([len(seg.get("text", "")) for seg in segments])
                confidence_score = min(1.0, avg_segment_length / 20.0)  # Rough estimate
            else:
                confidence_score = 0.5 if transcription else 0.0
            
            return {
                'transcription': transcription,
                'confidence_score': confidence_score,
                'segments': segments,
                'word_count': len(transcription.split()) if transcription else 0
            }
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}")
            return {
                'transcription': '',
                'confidence_score': 0.0,
                'segments': [],
                'word_count': 0,
                'transcription_error': str(e)
            }
    
    def _analyze_filler_words(self, transcription: str) -> Dict:
        """Analyze filler words in transcription"""
        try:
            if not transcription:
                return {
                    'filler_word_count': 0,
                    'filler_word_rate': 0.0,
                    'filler_words_found': [],
                    'clean_transcription': ''
                }
            
            # Clean and normalize text
            text = transcription.lower()
            text = re.sub(r'[^\w\s]', ' ', text)  # Remove punctuation
            words = text.split()
            
            # Find filler words
            filler_words_found = []
            clean_words = []
            
            i = 0
            while i < len(words):
                found_filler = False
                
                # Check for multi-word fillers first
                for filler in sorted(self.filler_words, key=len, reverse=True):
                    filler_words_list = filler.split()
                    if i + len(filler_words_list) <= len(words):
                        if words[i:i+len(filler_words_list)] == filler_words_list:
                            filler_words_found.append({
                                'filler': filler,
                                'position': i,
                                'type': 'multi_word' if len(filler_words_list) > 1 else 'single_word'
                            })
                            i += len(filler_words_list)
                            found_filler = True
                            break
                
                if not found_filler:
                    clean_words.append(words[i])
                    i += 1
            
            total_words = len(words)
            filler_count = len(filler_words_found)
            filler_rate = (filler_count / total_words) * 100 if total_words > 0 else 0
            
            return {
                'filler_word_count': filler_count,
                'filler_word_rate': filler_rate,
                'filler_words_found': filler_words_found,
                'clean_transcription': ' '.join(clean_words),
                'total_words': total_words,
                'clean_words': len(clean_words)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing filler words: {e}")
            return {
                'filler_word_count': 0,
                'filler_word_rate': 0.0,
                'filler_words_found': [],
                'clean_transcription': transcription
            }
    
    def _analyze_pitch(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Analyze pitch characteristics"""
        try:
            # Extract pitch using librosa
            pitches, magnitudes = librosa.piptrack(
                y=audio_data, 
                sr=sample_rate, 
                threshold=0.1,
                fmin=50,  # Minimum frequency (Hz)
                fmax=400  # Maximum frequency (Hz)
            )
            
            # Extract fundamental frequency
            pitch_values = []
            for t in range(pitches.shape[1]):
                index = magnitudes[:, t].argmax()
                pitch = pitches[index, t]
                if pitch > 0:
                    pitch_values.append(pitch)
            
            if not pitch_values:
                return {
                    'average_pitch': 0,
                    'pitch_range': 0,
                    'pitch_variation': 0,
                    'pitch_stability': 0,
                    'error': 'No pitch detected'
                }
            
            pitch_array = np.array(pitch_values)
            
            # Calculate statistics
            avg_pitch = np.mean(pitch_array)
            pitch_std = np.std(pitch_array)
            pitch_range = np.max(pitch_array) - np.min(pitch_array)
            
            # Pitch variation (coefficient of variation)
            pitch_variation = pitch_std / avg_pitch if avg_pitch > 0 else 0
            
            # Pitch stability (inverse of variation, normalized)
            pitch_stability = max(0, 1 - (pitch_variation / 0.5))  # Normalize to 0-1
            
            # Classify pitch level
            if avg_pitch < 120:
                pitch_level = 'low'
            elif avg_pitch < 200:
                pitch_level = 'medium'
            else:
                pitch_level = 'high'
            
            return {
                'average_pitch': float(avg_pitch),
                'pitch_standard_deviation': float(pitch_std),
                'pitch_range': float(pitch_range),
                'pitch_variation': float(pitch_variation),
                'pitch_stability': float(pitch_stability),
                'pitch_level': pitch_level,
                'pitch_samples': len(pitch_values)
            }
            
        except Exception as e:
            logger.error(f"Error analyzing pitch: {e}")
            return {
                'average_pitch': 0,
                'pitch_range': 0,
                'pitch_variation': 0,
                'pitch_stability': 0,
                'error': str(e)
            }
    
    def _analyze_volume(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Analyze volume and energy characteristics"""
        try:
            # RMS energy analysis
            frame_length = 2048
            rms_energy = librosa.feature.rms(
                y=audio_data, 
                frame_length=frame_length
            )[0]
            
            # Convert to dB
            rms_db = librosa.amplitude_to_db(rms_energy)
            
            # Statistics
            avg_volume = np.mean(rms_db)
            volume_std = np.std(rms_db)
            volume_range = np.max(rms_db) - np.min(rms_db)
            
            # Volume consistency (inverse of standard deviation)
            volume_consistency = max(0, 1 - (volume_std / 20))  # Normalize
            
            # Dynamic range analysis
            dynamic_range = volume_range
            
            # Energy distribution analysis
            energy_percentiles = np.percentile(rms_db, [25, 50, 75, 90])
            
            return {
                'average_volume_db': float(avg_volume),
                'volume_standard_deviation': float(volume_std),
                'volume_range_db': float(volume_range),
                'volume_consistency': float(volume_consistency),
                'dynamic_range': float(dynamic_range),
                'energy_percentiles': {
                    '25th': float(energy_percentiles[0]),
                    '50th': float(energy_percentiles[1]),
                    '75th': float(energy_percentiles[2]),
                    '90th': float(energy_percentiles[3])
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing volume: {e}")
            return {
                'average_volume_db': 0,
                'volume_consistency': 0,
                'dynamic_range': 0,
                'error': str(e)
            }
    
    def _analyze_speaking_rate(self, transcription: str, duration: float) -> Dict:
        """Analyze speaking rate (words per minute)"""
        try:
            if not transcription or duration <= 0:
                return {
                    'speaking_rate_wpm': 0,
                    'speaking_rate_category': 'unknown',
                    'pace_score': 0
                }
            
            word_count = len(transcription.split())
            speaking_rate = (word_count / duration) * 60  # words per minute
            
            # Categorize speaking rate
            if speaking_rate < 120:
                category = 'slow'
                pace_score = 0.6  # Slow is okay but not ideal
            elif speaking_rate < 160:
                category = 'normal'
                pace_score = 1.0  # Ideal range
            elif speaking_rate < 200:
                category = 'fast'
                pace_score = 0.8  # Fast but acceptable
            else:
                category = 'very_fast'
                pace_score = 0.4  # Too fast
            
            return {
                'speaking_rate_wpm': float(speaking_rate),
                'speaking_rate_category': category,
                'pace_score': pace_score,
                'word_count': word_count,
                'duration_seconds': duration
            }
            
        except Exception as e:
            logger.error(f"Error analyzing speaking rate: {e}")
            return {
                'speaking_rate_wpm': 0,
                'speaking_rate_category': 'unknown',
                'pace_score': 0
            }
    
    def _analyze_pauses(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Analyze pause patterns in speech"""
        try:
            # Voice Activity Detection for pause analysis
            frame_length = 2048
            hop_length = 512
            
            # Calculate RMS energy
            rms = librosa.feature.rms(
                y=audio_data,
                frame_length=frame_length,
                hop_length=hop_length
            )[0]
            
            # Define silence threshold
            silence_threshold = np.percentile(rms, 20)  # Bottom 20% as silence
            
            # Find silence frames
            silence_frames = rms < silence_threshold
            
            # Convert frames to time
            frame_times = librosa.frames_to_time(
                np.arange(len(silence_frames)),
                sr=sample_rate,
                hop_length=hop_length
            )
            
            # Find pause segments
            pauses = []
            in_pause = False
            pause_start = 0
            
            for i, is_silent in enumerate(silence_frames):
                if is_silent and not in_pause:
                    pause_start = frame_times[i]
                    in_pause = True
                elif not is_silent and in_pause:
                    pause_duration = frame_times[i] - pause_start
                    if pause_duration > 0.1:  # Only count pauses > 100ms
                        pauses.append({
                            'start': pause_start,
                            'end': frame_times[i],
                            'duration': pause_duration
                        })
                    in_pause = False
            
            # Calculate pause statistics
            if pauses:
                pause_durations = [p['duration'] for p in pauses]
                avg_pause_duration = np.mean(pause_durations)
                total_pause_time = sum(pause_durations)
                pause_frequency = len(pauses) / (len(audio_data) / sample_rate)  # pauses per second
            else:
                avg_pause_duration = 0
                total_pause_time = 0
                pause_frequency = 0
            
            # Pause appropriateness score
            # Ideal: 1-3 pauses per minute, 0.5-2 seconds each
            ideal_pause_freq = 0.03  # ~2 pauses per minute
            if 0.01 <= pause_frequency <= 0.05:  # 0.6-3 pauses per minute
                freq_score = 1.0
            else:
                freq_score = 0.5
            
            if 0.5 <= avg_pause_duration <= 2.0:
                duration_score = 1.0
            else:
                duration_score = 0.6
            
            pause_score = (freq_score + duration_score) / 2
            
            return {
                'pause_count': len(pauses),
                'total_pause_time': float(total_pause_time),
                'average_pause_duration': float(avg_pause_duration),
                'pause_frequency': float(pause_frequency),
                'pause_score': float(pause_score),
                'pause_details': pauses[:10]  # Limit to first 10 pauses
            }
            
        except Exception as e:
            logger.error(f"Error analyzing pauses: {e}")
            return {
                'pause_count': 0,
                'total_pause_time': 0,
                'average_pause_duration': 0,
                'pause_frequency': 0,
                'pause_score': 0.5
            }
    
    def _calculate_clarity_score(self, results: Dict) -> float:
        """Calculate overall clarity score based on multiple factors"""
        try:
            factors = []
            
            # Transcription confidence
            factors.append(results.get('confidence_score', 0.5))
            
            # Filler word penalty
            filler_rate = results.get('filler_word_rate', 0)
            filler_score = max(0, 1 - (filler_rate / 10))  # Penalize >10% filler rate
            factors.append(filler_score)
            
            # Speaking rate appropriateness
            pace_score = results.get('pace_score', 0.5)
            factors.append(pace_score)
            
            # Pitch stability
            pitch_stability = results.get('pitch_analysis', {}).get('pitch_stability', 0.5)
            factors.append(pitch_stability)
            
            # Volume consistency
            volume_consistency = results.get('volume_analysis', {}).get('volume_consistency', 0.5)
            factors.append(volume_consistency)
            
            # Pause appropriateness
            pause_score = results.get('pause_analysis', {}).get('pause_score', 0.5)
            factors.append(pause_score)
            
            # Calculate weighted average
            clarity_score = np.mean(factors)
            return max(0.0, min(1.0, clarity_score))
            
        except Exception as e:
            logger.error(f"Error calculating clarity score: {e}")
            return 0.5
    
    def _calculate_voice_stability(self, results: Dict) -> float:
        """Calculate voice stability score"""
        try:
            stability_factors = []
            
            # Pitch stability
            pitch_stability = results.get('pitch_analysis', {}).get('pitch_stability', 0.5)
            stability_factors.append(pitch_stability)
            
            # Volume consistency
            volume_consistency = results.get('volume_analysis', {}).get('volume_consistency', 0.5)
            stability_factors.append(volume_consistency)
            
            # Speaking rate consistency (from history if available)
            if len(self.analysis_history) > 3:
                recent_rates = [h.get('speaking_rate_wpm', 0) for h in self.analysis_history[-5:]]
                if all(r > 0 for r in recent_rates):
                    rate_std = np.std(recent_rates)
                    rate_mean = np.mean(recent_rates)
                    rate_consistency = max(0, 1 - (rate_std / rate_mean)) if rate_mean > 0 else 0.5
                    stability_factors.append(rate_consistency)
            
            voice_stability = np.mean(stability_factors)
            return max(0.0, min(1.0, voice_stability))
            
        except Exception as e:
            logger.error(f"Error calculating voice stability: {e}")
            return 0.5
    
    def _update_analysis_history(self, results: Dict):
        """Update analysis history for pattern detection"""
        self.analysis_history.append({
            'timestamp': results['timestamp'],
            'speaking_rate_wpm': results.get('speaking_rate_wpm', 0),
            'clarity_score': results.get('clarity_score', 0),
            'filler_word_rate': results.get('filler_word_rate', 0),
            'average_pitch': results.get('pitch_analysis', {}).get('average_pitch', 0),
            'volume_consistency': results.get('volume_analysis', {}).get('volume_consistency', 0)
        })
        
        # Maintain history size
        if len(self.analysis_history) > self.max_history:
            self.analysis_history.pop(0)
    
    def _update_voice_baseline(self, results: Dict):
        """Update voice baseline characteristics"""
        try:
            # Update pitch baseline
            avg_pitch = results.get('pitch_analysis', {}).get('average_pitch', 0)
            if avg_pitch > 0:
                if self.voice_baseline['average_pitch'] is None:
                    self.voice_baseline['average_pitch'] = avg_pitch
                else:
                    # Running average
                    self.voice_baseline['average_pitch'] = (
                        self.voice_baseline['average_pitch'] * 0.8 + avg_pitch * 0.2
                    )
            
            # Update pace baseline
            speaking_rate = results.get('speaking_rate_wpm', 0)
            if speaking_rate > 0:
                if self.voice_baseline['average_pace'] is None:
                    self.voice_baseline['average_pace'] = speaking_rate
                else:
                    self.voice_baseline['average_pace'] = (
                        self.voice_baseline['average_pace'] * 0.8 + speaking_rate * 0.2
                    )
            
        except Exception as e:
            logger.error(f"Error updating voice baseline: {e}")
    
    def get_analysis_summary(self) -> Dict:
        """Get summary of recent analysis results"""
        if not self.analysis_history:
            return {'error': 'No analysis history available'}
        
        try:
            recent = self.analysis_history[-10:]  # Last 10 analyses
            
            summary = {
                'session_count': len(recent),
                'average_clarity': np.mean([r['clarity_score'] for r in recent]),
                'average_speaking_rate': np.mean([r['speaking_rate_wpm'] for r in recent if r['speaking_rate_wpm'] > 0]),
                'average_filler_rate': np.mean([r['filler_word_rate'] for r in recent]),
                'voice_consistency': np.std([r['average_pitch'] for r in recent if r['average_pitch'] > 0]),
                'improvement_trend': self._calculate_improvement_trend(recent),
                'voice_baseline': self.voice_baseline.copy()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating analysis summary: {e}")
            return {'error': str(e)}
    
    def _calculate_improvement_trend(self, recent_results: List[Dict]) -> str:
        """Calculate if performance is improving, stable, or declining"""
        if len(recent_results) < 3:
            return 'insufficient_data'
        
        try:
            # Use clarity scores for trend analysis
            clarity_scores = [r['clarity_score'] for r in recent_results]
            
            # Simple linear trend
            x = np.arange(len(clarity_scores))
            slope = np.polyfit(x, clarity_scores, 1)[0]
            
            if slope > 0.02:
                return 'improving'
            elif slope < -0.02:
                return 'declining'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f"Error calculating improvement trend: {e}")
            return 'unknown'
    
    def reset_analysis(self):
        """Reset analysis history and baseline"""
        self.analysis_history = []
        self.voice_baseline = {
            'average_pitch': None,
            'pitch_range': None,
            'average_pace': None,
            'volume_baseline': None
        }
        logger.info("Audio analysis history and baseline reset")
    
    def cleanup(self):
        """Clean up resources"""
        try:
            # Whisper model cleanup is handled by garbage collection
            self.whisper_model = None
            logger.info("AudioAnalyzer cleaned up successfully")
        except Exception as e:
            logger.error(f"Error during audio analyzer cleanup: {e}")

# Utility function for testing
def test_audio_analyzer():
    """Test the AudioAnalyzer with a sample audio file"""
    analyzer = AudioAnalyzer(model_size="base")
    
    print("AudioAnalyzer Test")
    print("Place a test audio file named 'test_audio.wav' in the current directory")
    
    test_file = "test_audio.wav"
    if os.path.exists(test_file):
        print(f"Analyzing {test_file}...")
        results = analyzer.analyze_audio_file(test_file)
        
        print("\nAnalysis Results:")
        print(f"Transcription: {results.get('transcription', 'N/A')[:100]}...")
        print(f"Clarity Score: {results.get('clarity_score', 0):.2f}")
        print(f"Speaking Rate: {results.get('speaking_rate_wpm', 0):.1f} WPM")
        print(f"Filler Words: {results.get('filler_word_count', 0)} ({results.get('filler_word_rate', 0):.1f}%)")
        print(f"Voice Stability: {results.get('voice_stability', 0):.2f}")
        
        pitch_data = results.get('pitch_analysis', {})
        print(f"Average Pitch: {pitch_data.get('average_pitch', 0):.1f} Hz")
        
        volume_data = results.get('volume_analysis', {})
        print(f"Volume Consistency: {volume_data.get('volume_consistency', 0):.2f}")
        
    else:
        print(f"Test file {test_file} not found. Please provide a test audio file.")

if __name__ == "__main__":
    test_audio_analyzer()