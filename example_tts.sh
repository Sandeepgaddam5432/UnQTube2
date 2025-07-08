#!/bin/bash

# Gemini 2.5 TTS API Configuration
API_KEY="AIzaSyDaiPi28dBK7Yu6fzXFCkTciTrbs47FgSU"
BASE_URL="https://generativelanguage.googleapis.com/v1beta/models"

# Available TTS Models
GEMINI_2_5_PRO_TTS="gemini-2.5-pro:generateContent"
GEMINI_2_5_FLASH_TTS="gemini-2.5-flash:generateContent"

# Create output directories
mkdir -p /sdcard/tts/
mkdir -p ~/temp/  # Use home directory instead of /tmp/

# Function to convert text to speech using Gemini 2.5
gemini_tts() {
    local text="$1"
    local filename="$2"
    local model="${3:-gemini-2.5-flash}"  # Default to Flash model
    local voice_config="${4:-natural}"     # Voice configuration
    
    # Determine model endpoint
    local model_endpoint=""
    if [[ "$model" == *"pro"* ]]; then
        model_endpoint="$GEMINI_2_5_PRO_TTS"
    else
        model_endpoint="$GEMINI_2_5_FLASH_TTS"
    fi
    
    # Create JSON payload for Gemini TTS
    local json_payload=$(cat <<EOF
{
  "contents": [{
    "parts": [{
      "text": "Generate speech for the following text: $text"
    }]
  }],
  "generationConfig": {
    "response_modalities": ["AUDIO"],
    "speech_config": {
      "voice_config": {
        "prebuilt_voice_config": {
          "voice_name": "$voice_config"
        }
      }
    }
  }
}
EOF
)
    
    echo "Converting text to speech using Gemini 2.5..."
    echo "Model: $model"
    echo "Text: $text"
    echo "Voice: $voice_config"
    
    # Make API call with retry logic
    local api_url="$BASE_URL/$model_endpoint?key=$API_KEY"
    
    echo "Making API request to: $model_endpoint"
    echo "Waiting 5 seconds to avoid rate limit..."
    sleep 5
    
    curl -X POST \
        -H "Content-Type: application/json" \
        -d "$json_payload" \
        "$api_url" \
        -o "~/temp/gemini_tts_response.json" \
        -w "HTTP Status: %{http_code}\n" \
        --connect-timeout 30 \
        --max-time 60
    
    # Process response
    if [ -f "~/temp/gemini_tts_response.json" ]; then
        # Check for errors
        if grep -q "error" "~/temp/gemini_tts_response.json"; then
            echo "Error occurred:"
            cat "~/temp/gemini_tts_response.json" | jq '.' 2>/dev/null || cat "~/temp/gemini_tts_response.json"
            return 1
        fi
        
        # Try to extract audio data (format may vary)
        # Gemini might return audio in different formats
        audio_data=$(cat "~/temp/gemini_tts_response.json" | jq -r '.candidates[0].content.parts[0].audio_data // .audioContent // empty' 2>/dev/null)
        
        if [ -n "$audio_data" ] && [ "$audio_data" != "null" ]; then
            # Decode base64 audio and save
            echo "$audio_data" | base64 -d > "/sdcard/tts/$filename.mp3"
            echo "✅ Audio saved to: /sdcard/tts/$filename.mp3"
        else
            echo "❌ No audio data found in response. Response:"
            cat "~/temp/gemini_tts_response.json" | jq '.' 2>/dev/null || cat "~/temp/gemini_tts_response.json"
        fi
        
        # Cleanup
        rm "~/temp/gemini_tts_response.json"
    else
        echo "❌ Failed to get response from Gemini API"
    fi
}

# Alternative TTS function using generateContent with specific TTS prompt
gemini_tts_alternative() {
    local text="$1"
    local filename="$2"
    local model="${3:-gemini-2.5-flash}"
    
    local model_name=""
    if [[ "$model" == *"pro"* ]]; then
        model_name="gemini-2.5-pro"
    else
        model_name="gemini-2.5-flash"
    fi
    
    # Simple generateContent approach
    local json_payload=$(cat <<EOF
{
  "contents": [{
    "parts": [{
      "text": "Convert this text to natural speech audio: $text"
    }]
  }],
  "generationConfig": {
    "maxOutputTokens": 1000,
    "temperature": 0.7
  }
}
EOF
)
    
    echo "Trying alternative Gemini TTS approach..."
    
    local api_url="$BASE_URL/$model_name:generateContent?key=$API_KEY"
    
    curl -X POST \
        -H "Content-Type: application/json" \
        -d "$json_payload" \
        "$api_url" \
        -o "~/temp/gemini_alt_response.json"
    
    if [ -f "~/temp/gemini_alt_response.json" ]; then
        echo "Response received:"
        cat "~/temp/gemini_alt_response.json" | jq '.' 2>/dev/null || cat "~/temp/gemini_alt_response.json"
        rm "~/temp/gemini_alt_response.json"
    fi
}

# Function to test API connectivity
test_gemini_api() {
    echo "Testing Gemini API connectivity..."
    
    local test_payload='{"contents":[{"parts":[{"text":"Hello, this is a test"}]}]}'
    local api_url="$BASE_URL/gemini-2.5-flash:generateContent?key=$API_KEY"
    
    curl -X POST \
        -H "Content-Type: application/json" \
        -d "$test_payload" \
        "$api_url" \
        -w "HTTP Status: %{http_code}\n" \
        -o "~/temp/api_test.json"
    
    if [ -f "~/temp/api_test.json" ]; then
        echo "API Test Response:"
        cat "~/temp/api_test.json" | jq '.' 2>/dev/null || cat "~/temp/api_test.json"
        rm "~/temp/api_test.json"
    fi
}

# Display usage information
show_usage() {
    echo "=== Gemini 2.5 TTS Script ==="
    echo ""
    echo "Available functions:"
    echo "  gemini_tts \"text\" \"filename\" [model] [voice]"
    echo "  gemini_tts_alternative \"text\" \"filename\" [model]"
    echo "  test_gemini_api"
    echo ""
    echo "Models: gemini-2.5-pro, gemini-2.5-flash"
    echo "Voice configs: natural, casual, expressive"
    echo ""
    echo "Examples:"
    echo "  gemini_tts \"Hello world\" \"test\" \"gemini-2.5-flash\" \"natural\""
    echo "  gemini_tts \"Namaste, ela unnaru?\" \"telugu_greeting\" \"gemini-2.5-pro\""
    echo ""
}

# Main execution
echo "🎵 Gemini 2.5 TTS Script Ready!"
show_usage

# Test API first
echo "Running API connectivity test..."
test_gemini_api

echo ""
echo "Enter text for TTS conversion:"
read -p "Text: " user_text
read -p "Filename (without extension): " user_filename
read -p "Model (pro/flash) [default: flash]: " user_model

# Set default model
if [ -z "$user_model" ]; then
    user_model="gemini-2.5-flash"
elif [ "$user_model" = "pro" ]; then
    user_model="gemini-2.5-pro"
elif [ "$user_model" = "flash" ]; then
    user_model="gemini-2.5-flash"
fi

if [ -n "$user_text" ] && [ -n "$user_filename" ]; then
    echo ""
    echo "Attempting TTS conversion..."
    gemini_tts "$user_text" "$user_filename" "$user_model"
    
    echo ""
    echo "If above failed, trying alternative approach..."
    gemini_tts_alternative "$user_text" "$user_filename" "$user_model"
else
    echo "❌ Please provide both text and filename"
fi

echo ""
echo "Script completed! Check /sdcard/tts/ for output files."
