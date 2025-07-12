# UnQTube2 Hybrid Language Mode Implementation Plan

## Phase 1: Core Architecture Updates (Week 1)

### 1.1 Update Data Models
- [ ] Create `LanguageConfig` and `ScriptContent` models
- [ ] Update `VideoParams` schema to support multiple scripts
- [ ] Add `target_duration` field to video parameters
- [ ] Create migration scripts for existing data

### 1.2 Language Service Implementation
- [ ] Create `LanguageService` class
- [ ] Implement `generate_multilingual_content()` method
- [ ] Add language validation and compatibility checking
- [ ] Create language-to-voice mapping configuration

### 1.3 Database Schema Updates
- [ ] Add language support tables
- [ ] Update video parameters table
- [ ] Add voice configuration table
- [ ] Create indexes for language-based queries

## Phase 2: Enhanced Script Generation (Week 2)

### 2.1 LLM Service Enhancement
- [ ] Modify script generation to support target languages
- [ ] Implement native script generation (avoid translation)
- [ ] Add duration-based script adaptation
- [ ] Create script quality validation

### 2.2 Smart Voice Auto-Correction
- [ ] Implement `get_compatible_voice()` function
- [ ] Add voice-language compatibility matrix
- [ ] Create fallback voice selection logic
- [ ] Add warning logging for voice corrections

### 2.3 Script Processing Pipeline
```python
# Implementation order:
1. Generate script in target language (native)
2. Validate script-voice compatibility
3. Auto-correct voice if needed
4. Adapt script for target duration
5. Generate subtitle version if different language
```

## Phase 3: Gemini TTS Integration (Week 3)

### 3.1 Core TTS Implementation
- [ ] Implement `GeminiTTSService` class (from artifact)
- [ ] Add async support for better performance
- [ ] Implement error handling and retries
- [ ] Add audio format validation

### 3.2 Voice Preview Feature
- [ ] Create `generate_voice_preview()` function
- [ ] Add preview endpoint to FastAPI
- [ ] Implement preview caching mechanism
- [ ] Add preview cleanup job

### 3.3 TTS Service Integration
- [ ] Update existing voice service to use Gemini TTS
- [ ] Add TTS provider selection logic
- [ ] Implement fallback between TTS providers
- [ ] Add TTS performance monitoring

## Phase 4: UI/UX Enhancements (Week 4)

### 4.1 Streamlit UI Updates
- [ ] Add language selection dropdowns
- [ ] Implement voice preview button
- [ ] Add target duration slider
- [ ] Create hybrid mode toggle

### 4.2 Voice Selection Enhancement
```python
# UI Components to add:
1. Language selector for voiceover
2. Language selector for subtitles
3. Voice preview button with loading state
4. Compatible voice auto-selection indicator
5. Duration target input with validation
```

### 4.3 User Experience Improvements
- [ ] Add real-time voice compatibility checking
- [ ] Implement language-specific voice filtering
- [ ] Add voice preview audio player
- [ ] Create language selection presets

## Phase 5: Backend API Updates (Week 5)

### 5.1 FastAPI Endpoint Updates
- [ ] Update video generation endpoint
- [ ] Add voice preview endpoint
- [ ] Add language compatibility endpoint
- [ ] Implement duration estimation endpoint

### 5.2 Task Queue Enhancement
- [ ] Update video generation task
- [ ] Add subtitle generation task
- [ ] Implement parallel processing for audio/video
- [ ] Add progress tracking for multi-step generation

### 5.3 Error Handling & Logging
- [ ] Enhanced error messages for language mismatches
- [ ] Add structured logging for debugging
- [ ] Implement retry mechanisms
- [ ] Add performance metrics collection

## Phase 6: Testing & Validation (Week 6)

### 6.1 Unit Tests
- [ ] Test language service functions
- [ ] Test Gemini TTS service
- [ ] Test voice compatibility checking
- [ ] Test script generation with duration constraints

### 6.2 Integration Tests
- [ ] Test end-to-end video generation
- [ ] Test hybrid language mode
- [ ] Test voice preview functionality
- [ ] Test error recovery scenarios

### 6.3 Performance Testing
- [ ] Load test TTS services
- [ ] Test concurrent voice preview requests
- [ ] Validate memory usage with large scripts
- [ ] Test subtitle synchronization accuracy

## Configuration Files to Update

### 6.4 Environment Variables
```bash
# Add to .env
GEMINI_API_KEY=your_gemini_api_key
GEMINI_BASE_URL=https://generativelanguage.googleapis.com/v1beta/models
DEFAULT_VOICE_PREVIEW_TEXT="Hello, this is a voice preview."
MAX_SCRIPT_LENGTH=2000
DEFAULT_TARGET_DURATION=60
```

### 6.5 Language Configuration
```json
{
  "supported_languages": {
    "te": {"name": "Telugu", "code": "te", "rtl": false},
    "hi": {"name": "Hindi", "code": "hi", "rtl": false},
    "en": {"name": "English", "code": "en", "rtl": false}
  },
  "voice_mappings": {
    "te": ["te-IN-ShrutiNeural", "te-IN-MohanNeural"],
    "hi": ["hi-IN-SwaraNeural", "hi-IN-MadhurNeural"],
    "en": ["en-US-AriaNeural", "en-GB-SoniaNeural"]
  }
}
```

## Key Implementation Notes

### Priority Order
1. **Critical**: Fix the immediate language mismatch bug
2. **High**: Implement Gemini TTS service
3. **Medium**: Add voice preview functionality
4. **Low**: Add duration control features

### Risk Mitigation
- Implement comprehensive fallback mechanisms
- Add extensive logging for debugging
- Create rollback procedures for each phase
- Test with multiple language combinations

### Performance Considerations
- Use async operations for TTS calls
- Implement caching for voice previews
- Add rate limiting for API calls
- Optimize subtitle generation pipeline

### Security Considerations
- Validate all input text for script injection
- Implement API key rotation
- Add rate limiting to prevent abuse
- Sanitize file paths for audio output

This plan provides a structured approach to implementing your hybrid language mode while addressing the immediate bug and adding the requested features. Each phase builds upon the previous one, ensuring a stable progression toward the final goal.