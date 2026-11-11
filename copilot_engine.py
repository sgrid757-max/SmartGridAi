import os
import google.generativeai as genai

def test_ai_connection() -> bool:
    """Test if valid API keys are configured."""
    return bool(os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY") or os.getenv("GROQ_API_KEY"))

def ask_copilot(prompt: str, chat_history: list = None) -> dict:
    """
    Sends prompt to Google Gemini API using the official SDK and returns the response.
    """
    gemini_key = os.getenv("GEMINI_API_KEY")
    
    if not gemini_key:
        return {
            "text": "⚠️ **API Key Missing:** Please enter your Gemini API key in the sidebar or `.streamlit/secrets.toml`.",
            "sources": []
        }

    try:
        # Configure official Gemini client
        genai.configure(api_key=gemini_key)
        
        # Use gemini-1.5-flash model
        model_name = os.getenv("GEMINI_MODEL", "gemini-1.5-flash")
        model = genai.GenerativeModel(model_name)
        
        # Format history if available
        formatted_history = []
        if chat_history:
            for msg in chat_history[:-1]: # Exclude the latest user prompt
                role = "user" if msg["role"] == "user" else "model"
                content = msg.get("content", "")
                if content:
                    formatted_history.append({"role": role, "parts": [content]})
        
        # Start chat session or generate content directly
        if formatted_history:
            chat = model.start_chat(history=formatted_history)
            response = chat.send_message(prompt)
        else:
            response = model.generate_content(prompt)
            
        return {
            "text": response.text,
            "sources": []
        }
    except Exception as e:
        return {
            "text": f"❌ **Error connecting to Gemini API:** {str(e)}",
            "sources": []
        }