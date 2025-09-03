"""
Tests for YouTube service functionality.
"""

import pytest
from unittest.mock import Mock, patch, MagicMock
from youtube_transcript_api import TranscriptsDisabled, NoTranscriptFound

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent / "src"))

from src.services.youtube_service import YouTubeService
from src.core.exceptions import YouTubeAPIError, TranscriptProcessingError


class TestYouTubeService:
    """Test suite for YouTube service"""
    
    def setup_method(self):
        """Setup test method"""
        self.service = YouTubeService()
    
    @patch('src.services.youtube_service.YouTubeTranscriptApi.list_transcripts')
    def test_get_transcript_success(self, mock_list_transcripts, sample_youtube_transcript):
        """Test successful transcript retrieval"""
        # Mock transcript data
        mock_transcript = Mock()
        mock_transcript.fetch.return_value = sample_youtube_transcript
        mock_transcript.language_code = "en"
        
        mock_transcript_list = Mock()
        mock_transcript_list.find_transcript.return_value = mock_transcript
        mock_list_transcripts.return_value = mock_transcript_list
        
        # Test
        result = self.service.get_transcript("test_video_id")
        
        # Assertions
        assert isinstance(result, str)
        assert len(result) > 0
        assert "transformers" in result.lower()
        assert "attention mechanism" in result.lower()
    
    @patch('src.services.youtube_service.YouTubeTranscriptApi.list_transcripts')
    def test_get_transcript_no_preferred_language(self, mock_list_transcripts, sample_youtube_transcript):
        """Test transcript retrieval when preferred language not available"""
        # Mock transcript that throws NoTranscriptFound for English
        mock_transcript_list = Mock()
        mock_transcript_list.find_transcript.side_effect = NoTranscriptFound()
        
        # Mock available transcript in different language
        mock_transcript = Mock()
        mock_transcript.fetch.return_value = sample_youtube_transcript
        mock_transcript.language_code = "es"
        mock_transcript_list.__iter__ = Mock(return_value=iter([mock_transcript]))
        
        mock_list_transcripts.return_value = mock_transcript_list
        
        # Test
        result = self.service.get_transcript("test_video_id")
        
        # Assertions
        assert isinstance(result, str)
        assert len(result) > 0
    
    @patch('src.services.youtube_service.YouTubeTranscriptApi.list_transcripts')
    def test_get_transcript_disabled(self, mock_list_transcripts):
        """Test transcript retrieval when transcripts are disabled"""
        mock_list_transcripts.side_effect = TranscriptsDisabled("video_id")
        
        with pytest.raises(YouTubeAPIError) as exc_info:
            self.service.get_transcript("test_video_id")
        
        assert exc_info.value.error_code == "YOUTUBE_TRANSCRIPT_UNAVAILABLE"
    
    @patch('src.services.youtube_service.YouTubeTranscriptApi.list_transcripts')
    def test_get_transcript_no_transcripts_available(self, mock_list_transcripts):
        """Test transcript retrieval when no transcripts are available"""
        mock_transcript_list = Mock()
        mock_transcript_list.find_transcript.side_effect = NoTranscriptFound()
        mock_transcript_list.__iter__ = Mock(return_value=iter([]))
        
        mock_list_transcripts.return_value = mock_transcript_list
        
        with pytest.raises(YouTubeAPIError) as exc_info:
            self.service.get_transcript("test_video_id")
        
        assert exc_info.value.error_code == "YOUTUBE_TRANSCRIPT_UNAVAILABLE"
    
    def test_process_transcript_data_success(self, sample_youtube_transcript):
        """Test successful transcript data processing"""
        result = self.service._process_transcript_data(sample_youtube_transcript)
        
        assert isinstance(result, str)
        assert "Welcome to this video about transformers" in result
        assert "attention mechanism" in result
        assert "revolutionary" in result
    
    def test_process_transcript_data_empty(self):
        """Test transcript processing with empty data"""
        with pytest.raises(TranscriptProcessingError) as exc_info:
            self.service._process_transcript_data([])
        
        assert exc_info.value.error_code == "EMPTY_TRANSCRIPT_DATA"
    
    def test_process_transcript_data_no_text(self):
        """Test transcript processing with no text content"""
        transcript_data = [{"start": 0.0, "duration": 3.0}]  # No 'text' field
        
        with pytest.raises(TranscriptProcessingError) as exc_info:
            self.service._process_transcript_data(transcript_data)
        
        assert exc_info.value.error_code == "NO_TEXT_CONTENT"
    
    def test_clean_transcript_text(self):
        """Test transcript text cleaning"""
        dirty_text = "This is [Music] a test [Applause] with ♪ artifacts"
        cleaned = self.service._clean_transcript_text(dirty_text)
        
        assert "[Music]" not in cleaned
        assert "[Applause]" not in cleaned
        assert "♪" not in cleaned
        assert "This is a test with artifacts" == cleaned
    
    @patch('src.services.youtube_service.YouTubeTranscriptApi.list_transcripts')
    def test_get_transcript_metadata(self, mock_list_transcripts):
        """Test transcript metadata retrieval"""
        # Mock transcript objects
        mock_transcript1 = Mock()
        mock_transcript1.language = "English"
        mock_transcript1.language_code = "en"
        mock_transcript1.is_generated = False
        mock_transcript1.is_translatable = True
        
        mock_transcript2 = Mock()
        mock_transcript2.language = "Spanish"
        mock_transcript2.language_code = "es"
        mock_transcript2.is_generated = True
        mock_transcript2.is_translatable = True
        
        mock_transcript_list = [mock_transcript1, mock_transcript2]
        mock_list_transcripts.return_value = mock_transcript_list
        
        # Test
        result = self.service.get_transcript_metadata("test_video_id")
        
        # Assertions
        assert result["video_id"] == "test_video_id"
        assert result["total_count"] == 2
        assert len(result["available_transcripts"]) == 2
        assert result["available_transcripts"][0]["language"] == "English"
        assert result["available_transcripts"][1]["language_code"] == "es"
    
    @patch('src.services.youtube_service.YouTubeTranscriptApi.list_transcripts')
    def test_get_transcript_metadata_error(self, mock_list_transcripts):
        """Test transcript metadata retrieval with error"""
        mock_list_transcripts.side_effect = Exception("API Error")
        
        result = self.service.get_transcript_metadata("test_video_id")
        
        assert result["video_id"] == "test_video_id"
        assert result["total_count"] == 0
        assert "error" in result
    
    @patch('src.utils.cache.cache_manager.delete')
    def test_clear_cache(self, mock_cache_delete):
        """Test cache clearing"""
        mock_cache_delete.return_value = True
        
        # Test clearing specific video
        self.service.clear_cache("test_video_id")
        mock_cache_delete.assert_called_once()
        
        # Test clearing all (should just log a message)
        self.service.clear_cache()  # No exception should be raised
