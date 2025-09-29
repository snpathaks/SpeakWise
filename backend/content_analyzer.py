import re
import os
from collections import Counter
from datetime import datetime
import anthropic

class ContentAnalyzer:
    """Analyzes presentation content and provides AI-powered feedback"""
    
    def __init__(self):
        """Initialize the content analyzer"""
        self.api_key = os.getenv('ANTHROPIC_API_KEY')
        self.client = None
        
        if self.api_key:
            try:
                self.client = anthropic.Anthropic(api_key=self.api_key)
            except Exception as e:
                print(f"Warning: Could not initialize Anthropic client: {e}")
        
        # Common filler words to detect
        self.filler_words = {
            'um', 'uh', 'like', 'you know', 'so', 'actually', 'basically',
            'literally', 'right', 'okay', 'well', 'i mean', 'kind of',
            'sort of', 'just', 'really', 'very', 'quite'
        }
        
        # Transition words for structure analysis
        self.transition_words = {
            'first', 'second', 'third', 'finally', 'next', 'then',
            'however', 'therefore', 'moreover', 'furthermore',
            'in conclusion', 'to summarize', 'in addition'
        }
    
    def analyze_content(self, script_content, transcription=''):
        """
        Analyze presentation content and delivery
        
        Args:
            script_content: The original presentation script
            transcription: The actual spoken transcription
            
        Returns:
            Dictionary containing comprehensive feedback
        """
        feedback = {
            'timestamp': datetime.utcnow().isoformat(),
            'script_analysis': self._analyze_script_structure(script_content),
            'delivery_analysis': {},
            'improvements': [],
            'strengths': [],
            'ai_insights': {}
        }
        
        # Analyze delivery if transcription is provided
        if transcription:
            feedback['delivery_analysis'] = self._analyze_delivery(
                script_content, transcription
            )
            feedback['filler_word_analysis'] = self._detect_filler_words(transcription)
        
        # Get AI-powered insights
        if self.client:
            try:
                feedback['ai_insights'] = self._get_ai_feedback(
                    script_content, transcription
                )
            except Exception as e:
                print(f"Error getting AI feedback: {e}")
                feedback['ai_insights'] = self._get_fallback_feedback(script_content)
        else:
            feedback['ai_insights'] = self._get_fallback_feedback(script_content)
        
        # Compile improvement suggestions
        feedback['improvements'] = self._compile_improvements(feedback)
        feedback['strengths'] = self._identify_strengths(feedback)
        
        return feedback
    
    def _analyze_script_structure(self, script):
        """Analyze the structure and organization of the script"""
        words = script.split()
        sentences = re.split(r'[.!?]+', script)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        # Calculate readability metrics
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        # Detect structure elements
        has_introduction = any(word in script.lower() for word in 
                              ['hello', 'welcome', 'today', 'introduce'])
        has_conclusion = any(word in script.lower() for word in 
                            ['conclusion', 'summary', 'thank you', 'questions'])
        
        # Count transition words
        transition_count = sum(1 for word in self.transition_words 
                              if word in script.lower())
        
        return {
            'word_count': len(words),
            'sentence_count': len(sentences),
            'avg_sentence_length': round(avg_sentence_length, 2),
            'has_introduction': has_introduction,
            'has_conclusion': has_conclusion,
            'transition_word_count': transition_count,
            'estimated_duration': round(len(words) / 150, 2)  # ~150 words per minute
        }
    
    def _analyze_delivery(self, script, transcription):
        """Compare script to actual delivery"""
        script_words = set(script.lower().split())
        transcription_words = set(transcription.lower().split())
        
        # Calculate adherence to script
        common_words = script_words.intersection(transcription_words)
        adherence_score = len(common_words) / len(script_words) * 100 if script_words else 0
        
        # Detect improvisation
        improvised_words = transcription_words - script_words
        improvisation_rate = len(improvised_words) / len(transcription_words) * 100 if transcription_words else 0
        
        return {
            'adherence_score': round(adherence_score, 2),
            'improvisation_rate': round(improvisation_rate, 2),
            'script_word_count': len(script.split()),
            'delivered_word_count': len(transcription.split()),
            'coverage': round((len(transcription.split()) / len(script.split()) * 100), 2) if script.split() else 0
        }
    
    def _detect_filler_words(self, transcription):
        """Detect and count filler words in transcription"""
        text_lower = transcription.lower()
        
        filler_counts = {}
        for filler in self.filler_words:
            # Use word boundaries for accurate matching
            pattern = r'\b' + re.escape(filler) + r'\b'
            count = len(re.findall(pattern, text_lower))
            if count > 0:
                filler_counts[filler] = count
        
        total_fillers = sum(filler_counts.values())
        words = transcription.split()
        filler_percentage = (total_fillers / len(words) * 100) if words else 0
        
        return {
            'total_count': total_fillers,
            'filler_percentage': round(filler_percentage, 2),
            'breakdown': filler_counts,
            'severity': self._categorize_filler_severity(filler_percentage)
        }
    
    def _categorize_filler_severity(self, percentage):
        """Categorize filler word usage severity"""
        if percentage < 2:
            return 'excellent'
        elif percentage < 5:
            return 'good'
        elif percentage < 8:
            return 'needs_improvement'
        else:
            return 'poor'
    
    def _get_ai_feedback(self, script, transcription):
        """Get AI-powered feedback using Claude"""
        prompt = f"""Analyze this presentation and provide constructive feedback.

Original Script:
{script[:2000]}  # Limit for API

{"Delivered Transcription:" if transcription else ""}
{transcription[:2000] if transcription else ""}

Provide feedback in the following JSON format:
{{
    "overall_assessment": "brief overall assessment",
    "content_quality": {{
        "score": 0-100,
        "feedback": "specific feedback on content"
    }},
    "structure": {{
        "score": 0-100,
        "feedback": "feedback on organization and flow"
    }},
    "clarity": {{
        "score": 0-100,
        "feedback": "feedback on clarity and understandability"
    }},
    "engagement": {{
        "score": 0-100,
        "feedback": "feedback on audience engagement"
    }},
    "key_strengths": ["strength 1", "strength 2", "strength 3"],
    "improvement_areas": ["area 1", "area 2", "area 3"],
    "specific_suggestions": ["suggestion 1", "suggestion 2", "suggestion 3"]
}}

Be constructive, specific, and actionable in your feedback."""

        try:
            message = self.client.messages.create(
                model="claude-sonnet-4-20250514",
                max_tokens=1500,
                temperature=0.7,
                messages=[
                    {"role": "user", "content": prompt}
                ]
            )
            
            # Parse the response
            response_text = message.content[0].text
            
            # Extract JSON from response
            import json
            try:
                # Try to find JSON in the response
                json_match = re.search(r'\{.*\}', response_text, re.DOTALL)
                if json_match:
                    feedback_data = json.loads(json_match.group())
                    return feedback_data
                else:
                    # If no JSON found, return formatted text
                    return {
                        'overall_assessment': response_text,
                        'ai_available': True
                    }
            except json.JSONDecodeError:
                return {
                    'overall_assessment': response_text,
                    'ai_available': True
                }
                
        except Exception as e:
            print(f"AI feedback error: {e}")
            return self._get_fallback_feedback(script)
    
    def _get_fallback_feedback(self, script):
        """Provide rule-based feedback when AI is unavailable"""
        words = script.split()
        sentences = re.split(r'[.!?]+', script)
        sentences = [s.strip() for s in sentences if s.strip()]
        
        avg_sentence_length = len(words) / len(sentences) if sentences else 0
        
        feedback = {
            'ai_available': False,
            'overall_assessment': 'Analysis based on presentation best practices',
            'content_quality': {
                'score': 75,
                'feedback': 'Your presentation has good content foundation.'
            },
            'key_strengths': [],
            'improvement_areas': [],
            'specific_suggestions': []
        }
        
        # Analyze and provide feedback
        if len(words) > 500:
            feedback['key_strengths'].append('Comprehensive content with good depth')
        elif len(words) < 200:
            feedback['improvement_areas'].append('Consider expanding your content for better coverage')
        
        if avg_sentence_length > 25:
            feedback['improvement_areas'].append('Sentences are quite long - consider breaking them down')
            feedback['specific_suggestions'].append('Aim for 15-20 words per sentence for better clarity')
        elif avg_sentence_length < 10:
            feedback['specific_suggestions'].append('Good use of concise sentences - keep it up!')
        
        # Check for structure
        has_intro = any(word in script.lower() for word in ['hello', 'welcome', 'today'])
        has_conclusion = any(word in script.lower() for word in ['conclusion', 'summary', 'thank you'])
        
        if has_intro and has_conclusion:
            feedback['key_strengths'].append('Well-structured with clear introduction and conclusion')
        else:
            if not has_intro:
                feedback['improvement_areas'].append('Add a strong introduction')
            if not has_conclusion:
                feedback['improvement_areas'].append('Include a clear conclusion')
        
        return feedback
    
    def _compile_improvements(self, feedback):
        """Compile all improvement suggestions"""
        improvements = []
        
        # From script structure
        structure = feedback['script_analysis']
        if structure['avg_sentence_length'] > 25:
            improvements.append({
                'category': 'clarity',
                'suggestion': 'Break down long sentences for better comprehension',
                'priority': 'high'
            })
        
        if structure['transition_word_count'] < 3:
            improvements.append({
                'category': 'structure',
                'suggestion': 'Use more transition words to improve flow',
                'priority': 'medium'
            })
        
        # From filler words
        if 'filler_word_analysis' in feedback:
            filler_analysis = feedback['filler_word_analysis']
            if filler_analysis['severity'] in ['needs_improvement', 'poor']:
                improvements.append({
                    'category': 'delivery',
                    'suggestion': f"Reduce filler words - currently at {filler_analysis['filler_percentage']}%",
                    'priority': 'high'
                })
        
        # From delivery analysis
        if 'delivery_analysis' in feedback and feedback['delivery_analysis']:
            delivery = feedback['delivery_analysis']
            if delivery.get('adherence_score', 100) < 60:
                improvements.append({
                    'category': 'preparation',
                    'suggestion': 'Practice more to stay closer to your script',
                    'priority': 'medium'
                })
        
        # From AI insights
        if 'ai_insights' in feedback and 'improvement_areas' in feedback['ai_insights']:
            for area in feedback['ai_insights']['improvement_areas']:
                improvements.append({
                    'category': 'ai_suggestion',
                    'suggestion': area,
                    'priority': 'medium'
                })
        
        return improvements
    
    def _identify_strengths(self, feedback):
        """Identify strengths from the analysis"""
        strengths = []
        
        structure = feedback['script_analysis']
        
        if structure['has_introduction'] and structure['has_conclusion']:
            strengths.append('Well-organized presentation structure')
        
        if structure['transition_word_count'] >= 5:
            strengths.append('Good use of transitions for flow')
        
        if 10 <= structure['avg_sentence_length'] <= 20:
            strengths.append('Optimal sentence length for clarity')
        
        if 'filler_word_analysis' in feedback:
            if feedback['filler_word_analysis']['severity'] in ['excellent', 'good']:
                strengths.append('Minimal use of filler words')
        
        if 'delivery_analysis' in feedback and feedback['delivery_analysis']:
            if feedback['delivery_analysis'].get('adherence_score', 0) > 80:
                strengths.append('Strong adherence to prepared script')
        
        # From AI insights
        if 'ai_insights' in feedback and 'key_strengths' in feedback['ai_insights']:
            strengths.extend(feedback['ai_insights']['key_strengths'])
        
        return strengths
    
    def generate_report_summary(self, feedback):
        """Generate a human-readable summary of the feedback"""
        summary = {
            'overview': '',
            'scores': {},
            'top_strengths': [],
            'top_improvements': [],
            'next_steps': []
        }
        
        # Create overview
        word_count = feedback['script_analysis']['word_count']
        estimated_time = feedback['script_analysis']['estimated_duration']
        
        summary['overview'] = (
            f"Your presentation contains {word_count} words, "
            f"estimated at {estimated_time} minutes of delivery time."
        )
        
        # Compile scores from AI insights if available
        if 'ai_insights' in feedback and 'content_quality' in feedback['ai_insights']:
            ai_insights = feedback['ai_insights']
            if isinstance(ai_insights.get('content_quality'), dict):
                summary['scores']['content'] = ai_insights['content_quality'].get('score', 75)
            if isinstance(ai_insights.get('structure'), dict):
                summary['scores']['structure'] = ai_insights['structure'].get('score', 75)
            if isinstance(ai_insights.get('clarity'), dict):
                summary['scores']['clarity'] = ai_insights['clarity'].get('score', 75)
        
        # Top strengths (limit to 3)
        summary['top_strengths'] = feedback['strengths'][:3]
        
        # Top improvements (limit to 3 highest priority)
        priority_order = {'high': 0, 'medium': 1, 'low': 2}
        sorted_improvements = sorted(
            feedback['improvements'],
            key=lambda x: priority_order.get(x['priority'], 3)
        )
        summary['top_improvements'] = [
            imp['suggestion'] for imp in sorted_improvements[:3]
        ]
        
        # Next steps
        summary['next_steps'] = [
            'Practice delivery multiple times',
            'Record yourself and review',
            'Focus on reducing filler words',
            'Work on identified improvement areas'
        ]
        
        return summary