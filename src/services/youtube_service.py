"""
YouTube transcript service for fetching and processing video transcripts.
"""

import time
from typing import List, Dict, Optional
from youtube_transcript_api import YouTubeTranscriptApi, TranscriptsDisabled, NoTranscriptFound

from config.settings import settings
from src.core.exceptions import YouTubeAPIError, TranscriptProcessingError
from src.utils.logging_config import LoggerMixin, performance_logger
from src.utils.cache import cache_manager


class YouTubeService(LoggerMixin):
    """Service for handling YouTube transcript operations"""
    
    def __init__(self):
        self.default_languages = settings.rag.youtube_languages
        self.cache_ttl = 86400  # 24 hours for transcripts
    
    @cache_manager.cache_result("youtube_transcript", ttl=86400)
    def get_transcript(self, video_id: str, languages: Optional[List[str]] = None) -> str:
        """
        Get transcript for a YouTube video
        
        Args:
            video_id: YouTube video ID
            languages: List of preferred language codes (default: ['en'])
            
        Returns:
            Transcript text as a single string
            
        Raises:
            YouTubeAPIError: When transcript cannot be fetched
            TranscriptProcessingError: When transcript processing fails
        """
        start_time = time.time()
        languages = languages or self.default_languages
        
        try:
            self.logger.info(f"Fetching transcript for video {video_id}")
            
            # Get available transcripts
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            # Try to find transcript in preferred languages
            transcript_data = None
            selected_language = None
            
            for lang in languages:
                try:
                    transcript = transcript_list.find_transcript([lang])
                    transcript_data = transcript.fetch()
                    selected_language = lang
                    self.logger.info(f"Found transcript in language: {lang}")
                    break
                except NoTranscriptFound:
                    continue
            
            # If no preferred language found, try any available transcript
            if not transcript_data:
                try:
                    # Get first available transcript
                    available_transcripts = list(transcript_list)
                    if available_transcripts:
                        transcript = available_transcripts[0]
                        transcript_data = transcript.fetch()
                        selected_language = transcript.language_code
                        self.logger.info(f"Using available transcript in language: {selected_language}")
                    else:
                        raise YouTubeAPIError(
                            f"No transcripts available for video {video_id}",
                            "YOUTUBE_TRANSCRIPT_UNAVAILABLE"
                        )
                except Exception as e:
                    raise YouTubeAPIError(
                        f"Failed to fetch any transcript for video {video_id}: {str(e)}",
                        "YOUTUBE_TRANSCRIPT_UNAVAILABLE"
                    )
            
            # Process transcript data
            transcript_text = self._process_transcript_data(transcript_data)
            
            execution_time = time.time() - start_time
            performance_logger.log_execution_time(
                "youtube_transcript_fetch",
                execution_time,
                video_id=video_id,
                language=selected_language,
                transcript_length=len(transcript_text)
            )
            
            self.logger.info(
                f"Successfully fetched transcript for {video_id} "
                f"({len(transcript_text)} characters)"
            )
            
            return transcript_text
            
        except TranscriptsDisabled:
            raise YouTubeAPIError(
                f"Transcripts are disabled for video {video_id}",
                "YOUTUBE_TRANSCRIPT_UNAVAILABLE"
            )
        except Exception as e:
            if isinstance(e, (YouTubeAPIError, TranscriptProcessingError)):
                raise
            
            execution_time = time.time() - start_time
            performance_logger.log_execution_time(
                "youtube_transcript_fetch_failed",
                execution_time,
                video_id=video_id,
                error=str(e)
            )
            
            raise YouTubeAPIError(
                f"Unexpected error fetching transcript for {video_id}: {str(e)}",
                "YOUTUBE_API_ERROR"
            )
    
    def _process_transcript_data(self, transcript_data: List[Dict]) -> str:
        """
        Process raw transcript data into clean text
        
        Args:
            transcript_data: List of transcript segments
            
        Returns:
            Clean transcript text
            
        Raises:
            TranscriptProcessingError: When processing fails
        """
        try:
            if not transcript_data:
                raise TranscriptProcessingError(
                    "Empty transcript data received",
                    "EMPTY_TRANSCRIPT_DATA"
                )
            
            # Extract text from transcript segments
            text_segments = []
            for segment in transcript_data:
                if 'text' in segment:
                    text = segment['text'].strip()
                    if text:
                        text_segments.append(text)
            
            if not text_segments:
                raise TranscriptProcessingError(
                    "No text content found in transcript",
                    "NO_TEXT_CONTENT"
                )
            
            # Join segments with spaces
            transcript_text = " ".join(text_segments)
            
            # Basic cleaning
            transcript_text = self._clean_transcript_text(transcript_text)
            
            return transcript_text
            
        except Exception as e:
            if isinstance(e, TranscriptProcessingError):
                raise
            
            raise TranscriptProcessingError(
                f"Failed to process transcript data: {str(e)}",
                "TRANSCRIPT_PROCESSING_FAILED"
            )
    
    def _clean_transcript_text(self, text: str) -> str:
        """
        Clean transcript text by removing artifacts and normalizing
        
        Args:
            text: Raw transcript text
            
        Returns:
            Cleaned transcript text
        """
        # Remove common transcript artifacts
        text = text.replace("[Music]", "")
        text = text.replace("[Applause]", "")
        text = text.replace("[Laughter]", "")
        text = text.replace("♪", "")
        
        # Normalize whitespace
        text = " ".join(text.split())
        
        return text.strip()
    
    def get_transcript_metadata(self, video_id: str) -> Dict:
        """
        Get metadata about available transcripts for a video
        
        Args:
            video_id: YouTube video ID
            
        Returns:
            Dictionary with transcript metadata
        """
        try:
            transcript_list = YouTubeTranscriptApi.list_transcripts(video_id)
            
            available_transcripts = []
            for transcript in transcript_list:
                available_transcripts.append({
                    "language": transcript.language,
                    "language_code": transcript.language_code,
                    "is_generated": transcript.is_generated,
                    "is_translatable": transcript.is_translatable
                })
            
            return {
                "video_id": video_id,
                "available_transcripts": available_transcripts,
                "total_count": len(available_transcripts)
            }
            
        except Exception as e:
            self.logger.error(f"Failed to get transcript metadata for {video_id}: {e}")
            return {
                "video_id": video_id,
                "available_transcripts": [],
                "total_count": 0,
                "error": str(e)
            }
    
    def clear_cache(self, video_id: Optional[str] = None):
        """
        Clear transcript cache
        
        Args:
            video_id: Specific video ID to clear, or None to clear all
        """
        if video_id:
            # Clear specific video transcript
            cache_key = cache_manager._generate_key("youtube_transcript", video_id)
            cache_manager.delete(cache_key)
            self.logger.info(f"Cleared cache for video {video_id}")
        else:
            # This would require implementing a pattern-based delete in cache
            self.logger.info("Cache clearing for all transcripts not implemented")
