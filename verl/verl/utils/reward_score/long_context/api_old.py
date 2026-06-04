from openai import OpenAI
import os
import inspect
from .logger import cost_logger

_sync_client = None

def _get_caller_function_name():
    """Get the name of the function that called call_api"""
    # Walk up the stack to find the actual caller (skip call_api and wrapper)
    for frame_info in inspect.stack()[2:]:
        func_name = frame_info.function
        # Skip common wrapper/internal names
        if func_name not in ('wrapper', '<module>', '_call_api_internal'):
            return func_name
    return "unknown"

def load_api_key():
    return os.getenv("YEY_API_KEY")

def get_sync_client():
    global _sync_client
    if _sync_client is None:
        _sync_client = OpenAI(api_key=load_api_key(), base_url="https://yeysai.com/v1")
    return _sync_client

def call_api(model, contents, caller_name=None, temperature=1.0):
    """Single request version of the API call."""
    client = get_sync_client()
    
    # Handle both string content and list of messages
    if isinstance(contents, str):
        messages = [{"role": "user", "content": contents}]
    else:
        messages = contents

    # Retry loop for connection errors
    max_retries = 5
    last_exception = None
    
    for attempt in range(max_retries):
        try:
            completion = client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature
            )
            
            # Extract token usage
            input_tokens = 0
            output_tokens = 0
            if hasattr(completion, 'usage') and completion.usage:
                input_tokens = getattr(completion.usage, 'prompt_tokens', 0) or 0
                output_tokens = getattr(completion.usage, 'completion_tokens', 0) or 0
            else:
                print(f"⚠️ [API] No usage info found in response from {model}")
        
            function_name = caller_name or _get_caller_function_name()
            # Log cost
            cost_logger.log(model, function_name, input_tokens, output_tokens)
            
            if completion.choices[0].message.content is None:
                print(f"No response from {model}")
                return "No response"
            return completion.choices[0].message.content
            
        except Exception as e:
            last_exception = e
            import time
            print(f"⚠️ [API] Connection error on attempt {attempt+1}/{max_retries}: {e}")
            if attempt < max_retries - 1:
                sleep_time = 3 * (2 ** attempt)  # Exponential backoff: 3, 6, 12, 24...
                print(f"Waiting {sleep_time}s before retry...")
                time.sleep(sleep_time)
                
    # If we get here, all retries failed
    print(f"❌ [API] All retries failed. Last error: {last_exception}")
    raise last_exception