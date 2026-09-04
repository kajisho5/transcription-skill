"""transcription-skill: audio/video -> speech recognition -> structured Transcript.

The package turns speech into timestamped text and nothing more. It does not decide how that
text is used in a production (that is video-production-agent's job), does not style or burn
subtitles (subtitle-skill), and does not measure media properties (media-analysis-skill).
"""
__version__ = "0.1.0"
SKILL_ID = "transcription-skill"
