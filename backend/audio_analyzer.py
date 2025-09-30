import librosa
import numpy as np
import whisper
import soundfile as sf
import os
import time
import uuid
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass, asdict
import logging
import re
from collections import defaultdict
import json

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class AudioConfig:
    """Configuration for audio analysis"""
    # Whisper settings
    model_size: str = "base"
    language: str = "en"
    
    # Audio processing thresholds
    energy_percentile_threshold: int = 30
    silence_percentile_threshold: int = 20
    min_pause_duration: float = 0.1  # seconds
    low_volume_threshold: float = 0.01
    
    # Pitch analysis settings
    pitch_min_freq: int = 50  # Hz
    pitch_max_freq: int = 400  # Hz
    pitch_threshold: float = 0.1
    
    # Frame settings
    frame_length: int = 2048
    hop_length: int = 512
    vad_frame_length_ms: int = 25  # milliseconds
    vad_hop_length_ms: int = 10  # milliseconds
    
    # History settings
    max_history: int = 50
    
    # File size limit (100 MB)
    max_file_size_mb: int = 100
    
    # Speaking rate categories (words per minute)
    slow_rate_max: int = 120
    normal_rate_max: int = 160
    fast_rate_max: int = 200


@dataclass
class AnalysisResult:
    """Structured result from audio analysis"""
    timestamp: float
    duration: float
    sample_rate: int
    analysis_successful: bool
    transcription: str
    confidence_score: float
    filler_word_count: int
    filler_word_rate: float
    speaking_rate_wpm: float
    pause_analysis: Dict
    pitch_analysis: Dict
    volume_analysis: Dict
    clarity_score: float
    voice_stability: float
    word_count: int
    processing_time: float
    speech_segments: List[Dict]
    error: Optional[str] = None
    warnings: List[str] = None
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)


class AudioAnalyzer:
    """
    Audio analysis class using Whisper and librosa for:
    - Speech-to-text transcription
    - Filler word detection
    - Voice quality analysis (pitch, pace, clarity)
    - Speaking pattern analysis
    """
    
    def __init__(self, config: Optional[AudioConfig] = None):
        """
        Initialize AudioAnalyzer with Whisper model
        
        Args:
            config: AudioConfig instance with analysis parameters
        """
        self.config = config or AudioConfig()
        self.whisper_model = None
        
        # Expanded filler words list
        self.filler_words = [
            'um', 'uh', 'uhm', 'er', 'eh', 'ah', 'oh', 'hmm',
            'like', 'you know', 'so', 'actually', 'basically',
            'literally', 'totally', 'really', 'very',
            'kind of', 'sort of', 'i mean', 'well',
            'okay', 'alright', 'right', 'yeah', 'yes', 'yep',
            'sure', 'fine', 'good', 'great'
        ]
        
        # Analysis history for pattern detection
        self.analysis_history = []
        
        # Voice characteristics baseline (updated during analysis)
        self.voice_baseline = {
            'average_pitch': None,
            'pitch_range': None,
            'average_pace': None,
            'volume_baseline': None
        }
        
        logger.info(f"AudioAnalyzer initialized with model size: {self.config.model_size}")
    
    def _get_whisper_model(self):
        """Lazy loading of Whisper model to improve startup time"""
        if self.whisper_model is None:
            logger.info(f"Loading Whisper model: {self.config.model_size}")
            try:
                self.whisper_model = whisper.load_model(self.config.model_size)
                logger.info("Whisper model loaded successfully")
            except Exception as e:
                logger.error(f"Error loading Whisper model: {e}")
                raise RuntimeError(f"Failed to load Whisper model: {e}")
        return self.whisper_model
    
    def _validate_audio_file(self, file_path: str) -> None:
        """
        Validate audio file before processing
        
        Args:
            file_path: Path to audio file
            
        Raises:
            FileNotFoundError: If file doesn't exist
            ValueError: If file is too large or invalid
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Audio file not found: {file_path}")
        
        file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
        if file_size_mb > self.config.max_file_size_mb:
            raise ValueError(
                f"File too large: {file_size_mb:.1f}MB "
                f"(max: {self.config.max_file_size_mb}MB)"
            )
    
    def analyze_audio_file(self, audio_file_path: str) -> Dict:
        """
        Analyze audio file for all speech metrics
        
        Args:
            audio_file_path: Path to audio file
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Validate file
            self._validate_audio_file(audio_file_path)
            
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
            warnings = []
            
            # Initialize results structure
            results = {
                'timestamp': start_time,
                'duration': len(audio_data) / sample_rate,
                'sample_rate': sample_rate,
                'analysis_successful': True,
                'transcription': '',
                'confidence_score': 0.0,
                'filler_word_count': 0,
                'filler_word_rate': 0.0,
                'speaking_rate_wpm': 0.0,
                'pause_analysis': {},
                'pitch_analysis': {},
                'volume_analysis': {},
                'clarity_score': 0.0,
                'voice_stability': 0.0,
                'speech_segments': [],
                'word_count': 0,
                'warnings': []
            }
            
            # Basic audio quality checks
            max_amplitude = np.max(np.abs(audio_data))
            if max_amplitude < self.config.low_volume_threshold:
                warning = f"Audio level very low ({max_amplitude:.4f}), analysis may be inaccurate"
                logger.warning(warning)
                warnings.append(warning)
            
            # Voice Activity Detection (VAD)
            speech_segments = self._detect_speech_segments(audio_data, sample_rate)
            results['speech_segments'] = speech_segments
            
            if not speech_segments:
                warnings.append("No speech segments detected")
            
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
            results['warnings'] = warnings
            
            logger.info(f"Audio analysis completed in {processing_time:.2f} seconds")
            
            return results
            
        except Exception as e:
            logger.error(f"Error in audio analysis: {e}", exc_info=True)
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
            frame_length = int(self.config.vad_frame_length_ms / 1000 * sample_rate)
            hop_length = int(self.config.vad_hop_length_ms / 1000 * sample_rate)
            
            # Compute RMS energy
            rms_energy = librosa.feature.rms(
                y=audio_data, 
                frame_length=frame_length, 
                hop_length=hop_length
            )[0]
            
            # Adaptive threshold for speech detection
            energy_threshold = np.percentile(
                rms_energy, 
                self.config.energy_percentile_threshold
            )
            
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
                        'start': float(start_time),
                        'end': float(frame_times[i]),
                        'duration': float(frame_times[i] - start_time)
                    })
                    start_time = None
            
            # Handle case where speech continues to end
            if start_time is not None:
                segments.append({
                    'start': float(start_time),
                    'end': float(frame_times[-1]),
                    'duration': float(frame_times[-1] - start_time)
                })
            
            return segments
            
        except Exception as e:
            logger.error(f"Error in speech segment detection: {e}", exc_info=True)
            return []
    
    def _transcribe_audio(self, audio_data: np.ndarray, sample_rate: int, 
                         file_path: Optional[str] = None) -> Dict:
        """Transcribe audio using Whisper"""
        temp_file_path = None
        try:
            model = self._get_whisper_model()
            
            # Create temporary file if not provided
            if file_path is None:
                # Use UUID for unique temp file names
                temp_file_path = f"temp_whisper_{uuid.uuid4().hex}.wav"
                sf.write(temp_file_path, audio_data, sample_rate)
                file_path = temp_file_path
            
            # Transcribe with Whisper
            result = model.transcribe(
                file_path,
                language=self.config.language,
                task="transcribe",
                verbose=False
            )
            
            transcription = result["text"].strip()
            segments = result.get("segments", [])
            
            # Calculate improved confidence score
            confidence_score = self._calculate_transcription_confidence(
                transcription, segments, audio_data
            )
            
            return {
                'transcription': transcription,
                'confidence_score': confidence_score,
                'segments': segments,
                'word_count': len(transcription.split()) if transcription else 0
            }
            
        except Exception as e:
            logger.error(f"Error in transcription: {e}", exc_info=True)
            return {
                'transcription': '',
                'confidence_score': 0.0,
                'segments': [],
                'word_count': 0,
                'transcription_error': str(e)
            }
        finally:
            # Clean up temporary file with proper error handling
            if temp_file_path and os.path.exists(temp_file_path):
                try:
                    os.remove(temp_file_path)
                    logger.debug(f"Cleaned up temporary file: {temp_file_path}")
                except Exception as e:
                    logger.error(f"Failed to remove temporary file {temp_file_path}: {e}")
    
    def _calculate_transcription_confidence(self, transcription: str, 
                                           segments: List[Dict], 
                                           audio_data: np.ndarray) -> float:
        """
        Calculate confidence score for transcription
        
        Uses multiple factors:
        - Segment count and length
        - Text coherence (word length distribution)
        - Audio quality indicators
        """
        if not transcription:
            return 0.0
        
        factors = []
        
        # Factor 1: Segment length consistency
        if segments:
            segment_lengths = [len(seg.get("text", "")) for seg in segments]
            if segment_lengths:
                avg_length = np.mean(segment_lengths)
                # Normalized score: longer segments = higher confidence
                length_score = min(1.0, avg_length / 30.0)
                factors.append(length_score)
        
        # Factor 2: Word count relative to duration
        word_count = len(transcription.split())
        if word_count > 5:
            factors.append(0.8)
        elif word_count > 0:
            factors.append(0.5)
        else:
            factors.append(0.0)
        
        # Factor 3: Audio quality (signal-to-noise estimate)
        rms = np.sqrt(np.mean(audio_data**2))
        if rms > 0.05:
            factors.append(0.9)
        elif rms > 0.01:
            factors.append(0.7)
        else:
            factors.append(0.4)
        
        return float(np.mean(factors)) if factors else 0.5
    
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
            text = re.sub(r'\s+', ' ', text)  # Normalize whitespace
            words = text.split()
            
            # Build regex pattern for efficient matching
            # Sort by length (descending) to match longer phrases first
            sorted_fillers = sorted(self.filler_words, key=len, reverse=True)
            pattern = r'\b(' + '|'.join(re.escape(f) for f in sorted_fillers) + r')\b'
            
            # Find all filler word matches
            filler_matches = list(re.finditer(pattern, text))
            filler_words_found = [{
                'filler': match.group(),
                'position': len(text[:match.start()].split()),
                'type': 'multi_word' if ' ' in match.group() else 'single_word'
            } for match in filler_matches]
            
            # Create clean transcription
            clean_text = re.sub(pattern, '', text)
            clean_text = re.sub(r'\s+', ' ', clean_text).strip()
            
            total_words = len(words)
            filler_count = len(filler_words_found)
            filler_rate = (filler_count / total_words) * 100 if total_words > 0 else 0
            
            return {
                'filler_word_count': filler_count,
                'filler_word_rate': float(filler_rate),
                'filler_words_found': filler_words_found,
                'clean_transcription': clean_text,
                'total_words': total_words,
                'clean_words': len(clean_text.split())
            }
            
        except Exception as e:
            logger.error(f"Error analyzing filler words: {e}", exc_info=True)
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
                threshold=self.config.pitch_threshold,
                fmin=self.config.pitch_min_freq,
                fmax=self.config.pitch_max_freq
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
                    'pitch_level': 'unknown',
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
            pitch_stability = max(0, 1 - (pitch_variation / 0.5))
            
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
            logger.error(f"Error analyzing pitch: {e}", exc_info=True)
            return {
                'average_pitch': 0,
                'pitch_range': 0,
                'pitch_variation': 0,
                'pitch_stability': 0,
                'pitch_level': 'unknown',
                'error': str(e)
            }
    
    def _analyze_volume(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Analyze volume and energy characteristics"""
        try:
            # RMS energy analysis
            rms_energy = librosa.feature.rms(
                y=audio_data, 
                frame_length=self.config.frame_length
            )[0]
            
            # Convert to dB (avoid log of zero)
            rms_energy = np.maximum(rms_energy, 1e-10)
            rms_db = librosa.amplitude_to_db(rms_energy)
            
            # Statistics
            avg_volume = np.mean(rms_db)
            volume_std = np.std(rms_db)
            volume_range = np.max(rms_db) - np.min(rms_db)
            
            # Volume consistency (inverse of standard deviation, normalized)
            volume_consistency = max(0, 1 - (volume_std / 20))
            
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
            logger.error(f"Error analyzing volume: {e}", exc_info=True)
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
            
            # Categorize speaking rate using config
            if speaking_rate < self.config.slow_rate_max:
                category = 'slow'
                pace_score = 0.6
            elif speaking_rate < self.config.normal_rate_max:
                category = 'normal'
                pace_score = 1.0
            elif speaking_rate < self.config.fast_rate_max:
                category = 'fast'
                pace_score = 0.8
            else:
                category = 'very_fast'
                pace_score = 0.4
            
            return {
                'speaking_rate_wpm': float(speaking_rate),
                'speaking_rate_category': category,
                'pace_score': pace_score,
                'word_count': word_count,
                'duration_seconds': duration
            }
            
        except Exception as e:
            logger.error(f"Error analyzing speaking rate: {e}", exc_info=True)
            return {
                'speaking_rate_wpm': 0,
                'speaking_rate_category': 'unknown',
                'pace_score': 0
            }
    
    def _analyze_pauses(self, audio_data: np.ndarray, sample_rate: int) -> Dict:
        """Analyze pause patterns in speech"""
        try:
            # Calculate RMS energy
            rms = librosa.feature.rms(
                y=audio_data,
                frame_length=self.config.frame_length,
                hop_length=self.config.hop_length
            )[0]
            
            # Define silence threshold
            silence_threshold = np.percentile(
                rms, 
                self.config.silence_percentile_threshold
            )
            
            # Find silence frames
            silence_frames = rms < silence_threshold
            
            # Convert frames to time
            frame_times = librosa.frames_to_time(
                np.arange(len(silence_frames)),
                sr=sample_rate,
                hop_length=self.config.hop_length
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
                    if pause_duration > self.config.min_pause_duration:
                        pauses.append({
                            'start': float(pause_start),
                            'end': float(frame_times[i]),
                            'duration': float(pause_duration)
                        })
                    in_pause = False
            
            # Calculate pause statistics
            if pauses:
                pause_durations = [p['duration'] for p in pauses]
                avg_pause_duration = np.mean(pause_durations)
                total_pause_time = sum(pause_durations)
                pause_frequency = len(pauses) / (len(audio_data) / sample_rate)
            else:
                avg_pause_duration = 0
                total_pause_time = 0
                pause_frequency = 0
            
            # Pause appropriateness score
            # Ideal: 1-3 pauses per minute (0.016-0.05 per second), 0.5-2 seconds each
            if 0.016 <= pause_frequency <= 0.05:
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
            logger.error(f"Error analyzing pauses: {e}", exc_info=True)
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
            weights = []
            
            # Transcription confidence (weight: 2x)
            confidence = results.get('confidence_score', 0.5)
            factors.append(confidence)
            weights.append(2.0)
            
            # Filler word penalty (weight: 1.5x)
            filler_rate = results.get('filler_word_rate', 0)
            filler_score = max(0, 1 - (filler_rate / 10))
            factors.append(filler_score)
            weights.append(1.5)
            
            # Speaking rate appropriateness (weight: 1x)
            pace_score = results.get('pace_score', 0.5)
            factors.append(pace_score)
            weights.append(1.0)
            
            # Pitch stability (weight: 1x)
            pitch_stability = results.get('pitch_analysis', {}).get('pitch_stability', 0.5)
            factors.append(pitch_stability)
            weights.append(1.0)
            
            # Volume consistency (weight: 1x)
            volume_consistency = results.get('volume_analysis', {}).get('volume_consistency', 0.5)
            factors.append(volume_consistency)
            weights.append(1.0)
            
            # Pause appropriateness (weight: 1x)
            pause_score = results.get('pause_analysis', {}).get('pause_score', 0.5)
            factors.append(pause_score)
            weights.append(1.0)
            
            # Calculate weighted average
            clarity_score = np.average(factors, weights=weights)
            return float(max(0.0, min(1.0, clarity_score)))
            
        except Exception as e:
            logger.error(f"Error calculating clarity score: {e}", exc_info=True)
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
                recent_rates = [
                    h.get('speaking_rate_wpm', 0) 
                    for h in self.analysis_history[-5:]
                ]
                if all(r > 0 for r in recent_rates):
                    rate_std = np.std(recent_rates)
                    rate_mean = np.mean(recent_rates)
                    rate_consistency = max(0, 1 - (rate_std / rate_mean)) if rate_mean > 0 else 0.5
                    stability_factors.append(rate_consistency)
            
            voice_stability = np.mean(stability_factors)
            return float(max(0.0, min(1.0, voice_stability)))
            
        except Exception as e:
            logger.error(f"Error calculating voice stability: {e}", exc_info=True)
            return 0.5
    
    def _update_analysis_history(self, results: Dict):
        """Update analysis history for pattern detection"""
        try:
            self.analysis_history.append({
                'timestamp': results['timestamp'],
                'speaking_rate_wpm': results.get('speaking_rate_wpm', 0),
                'clarity_score': results.get('clarity_score', 0),
                'filler_word_rate': results.get('filler_word_rate', 0),
                'average_pitch': results.get('pitch_analysis', {}).get('average_pitch', 0),
                'volume_consistency': results.get('volume_analysis', {}).get('volume_consistency', 0)
            })
            
            # Maintain history size
            if len(self.analysis_history) > self.config.max_history:
                self.analysis_history.pop(0)
        except Exception as e:
            logger.error(f"Error updating analysis history: {e}", exc_info=True)
    
    def _update_voice_baseline(self, results: Dict):
        """Update voice baseline characteristics using exponential moving average"""
        try:
            alpha = 0.2  # Smoothing factor for exponential moving average
            
            # Update pitch baseline
            avg_pitch = results.get('pitch_analysis', {}).get('average_pitch', 0)
            if avg_pitch > 0:
                if self.voice_baseline['average_pitch'] is None:
                    self.voice_baseline['average_pitch'] = avg_pitch
                else:
                    self.voice_baseline['average_pitch'] = (
                        self.voice_baseline['average_pitch'] * (1 - alpha) + avg_pitch * alpha
                    )
            
            # Update pitch range baseline
            pitch_range = results.get('pitch_analysis', {}).get('pitch_range', 0)
            if pitch_range > 0:
                if self.voice_baseline['pitch_range'] is None:
                    self.voice_baseline['pitch_range'] = pitch_range
                else:
                    self.voice_baseline['pitch_range'] = (
                        self.voice_baseline['pitch_range'] * (1 - alpha) + pitch_range * alpha
                    )
            
            # Update pace baseline
            speaking_rate = results.get('speaking_rate_wpm', 0)
            if speaking_rate > 0:
                if self.voice_baseline['average_pace'] is None:
                    self.voice_baseline['average_pace'] = speaking_rate
                else:
                    self.voice_baseline['average_pace'] = (
                        self.voice_baseline['average_pace'] * (1 - alpha) + speaking_rate * alpha
                    )
            
            # Update volume baseline
            avg_volume = results.get('volume_analysis', {}).get('average_volume_db', 0)
            if avg_volume != 0:
                if self.voice_baseline['volume_baseline'] is None:
                    self.voice_baseline['volume_baseline'] = avg_volume
                else:
                    self.voice_baseline['volume_baseline'] = (
                        self.voice_baseline['volume_baseline'] * (1 - alpha) + avg_volume * alpha
                    )
            
        except Exception as e:
            logger.error(f"Error updating voice baseline: {e}", exc_info=True)
    
    def get_analysis_summary(self) -> Dict:
        """Get summary of recent analysis results"""
        if not self.analysis_history:
            return {'error': 'No analysis history available'}
        
        try:
            recent = self.analysis_history[-10:]  # Last 10 analyses
            
            # Calculate averages for valid values only
            clarity_scores = [r['clarity_score'] for r in recent if r['clarity_score'] > 0]
            speaking_rates = [r['speaking_rate_wpm'] for r in recent if r['speaking_rate_wpm'] > 0]
            filler_rates = [r['filler_word_rate'] for r in recent]
            pitches = [r['average_pitch'] for r in recent if r['average_pitch'] > 0]
            
            summary = {
                'session_count': len(recent),
                'average_clarity': float(np.mean(clarity_scores)) if clarity_scores else 0.0,
                'average_speaking_rate': float(np.mean(speaking_rates)) if speaking_rates else 0.0,
                'average_filler_rate': float(np.mean(filler_rates)) if filler_rates else 0.0,
                'voice_consistency': float(np.std(pitches)) if len(pitches) > 1 else 0.0,
                'improvement_trend': self._calculate_improvement_trend(recent),
                'voice_baseline': self.voice_baseline.copy()
            }
            
            return summary
            
        except Exception as e:
            logger.error(f"Error generating analysis summary: {e}", exc_info=True)
            return {'error': str(e)}
    
    def _calculate_improvement_trend(self, recent_results: List[Dict]) -> str:
        """Calculate if performance is improving, stable, or declining"""
        if len(recent_results) < 3:
            return 'insufficient_data'
        
        try:
            # Use clarity scores for trend analysis
            clarity_scores = [r['clarity_score'] for r in recent_results if r['clarity_score'] > 0]
            
            if len(clarity_scores) < 3:
                return 'insufficient_data'
            
            # Simple linear trend using least squares
            x = np.arange(len(clarity_scores))
            slope = np.polyfit(x, clarity_scores, 1)[0]
            
            # Thresholds for trend classification
            improving_threshold = 0.02
            declining_threshold = -0.02
            
            if slope > improving_threshold:
                return 'improving'
            elif slope < declining_threshold:
                return 'declining'
            else:
                return 'stable'
                
        except Exception as e:
            logger.error(f"Error calculating improvement trend: {e}", exc_info=True)
            return 'unknown'
    
    def export_analysis_history(self, filepath: str) -> bool:
        """
        Export analysis history to JSON file
        
        Args:
            filepath: Path to save JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(filepath, 'w') as f:
                json.dump({
                    'analysis_history': self.analysis_history,
                    'voice_baseline': self.voice_baseline,
                    'config': asdict(self.config)
                }, f, indent=2)
            logger.info(f"Analysis history exported to {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error exporting analysis history: {e}", exc_info=True)
            return False
    
    def import_analysis_history(self, filepath: str) -> bool:
        """
        Import analysis history from JSON file
        
        Args:
            filepath: Path to JSON file
            
        Returns:
            bool: True if successful, False otherwise
        """
        try:
            with open(filepath, 'r') as f:
                data = json.load(f)
            
            self.analysis_history = data.get('analysis_history', [])
            self.voice_baseline = data.get('voice_baseline', self.voice_baseline)
            
            logger.info(f"Analysis history imported from {filepath}")
            return True
        except Exception as e:
            logger.error(f"Error importing analysis history: {e}", exc_info=True)
            return False
    
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
            logger.error(f"Error during audio analyzer cleanup: {e}", exc_info=True)
    
    def __enter__(self):
        """Context manager entry"""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.cleanup()
        return False


def create_analyzer(model_size: str = "base", language: str = "en", **kwargs) -> AudioAnalyzer:
    """
    Factory function to create AudioAnalyzer with custom configuration
    
    Args:
        model_size: Whisper model size
        language: Language code for transcription
        **kwargs: Additional configuration parameters
        
    Returns:
        AudioAnalyzer instance
    """
    config = AudioConfig(model_size=model_size, language=language, **kwargs)
    return AudioAnalyzer(config)