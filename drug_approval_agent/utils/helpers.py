import datetime
import hashlib
import json
import os
import pickle
import requests
from typing import Any, Dict, List, Optional, Union
import logging
import time

# Setup logging
logging.basicConfig(
    level=os.environ.get("LOG_LEVEL", "INFO"),
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler(os.environ.get("LOG_FILE", "drug_approval_agent.log"))
    ]
)
logger = logging.getLogger(__name__)

def format_date(date_obj: Union[str, datetime.datetime, datetime.date]) -> str:
    """
    Format date to YYYY-MM-DD string
    """
    if isinstance(date_obj, str):
        try:
            date_obj = datetime.datetime.fromisoformat(date_obj.replace('Z', '+00:00'))
        except ValueError:
            try:
                date_obj = datetime.datetime.strptime(date_obj, "%Y-%m-%d")
            except ValueError:
                logger.warning(f"Could not parse date: {date_obj}")
                return date_obj
    
    if isinstance(date_obj, (datetime.datetime, datetime.date)):
        return date_obj.strftime("%Y-%m-%d")
    
    return str(date_obj)

def get_cache_key(data: Any) -> str:
    """
    Generate a unique cache key based on the input data
    """
    if isinstance(data, dict):
        # Sort dictionary to ensure consistent hashing
        data = json.dumps(data, sort_keys=True)
    
    if not isinstance(data, str):
        data = str(data)
    
    return hashlib.md5(data.encode()).hexdigest()

def save_to_cache(data: Any, cache_key: str, cache_dir: Optional[str] = None) -> bool:
    """
    Save data to the cache
    """
    if cache_dir is None:
        cache_dir = os.environ.get("CACHE_DIR", "./data/cache")
    
    os.makedirs(cache_dir, exist_ok=True)
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    try:
        with open(cache_path, 'wb') as f:
            pickle.dump({
                'data': data,
                'timestamp': datetime.datetime.now()
            }, f)
        return True
    except Exception as e:
        logger.error(f"Error saving to cache: {e}")
        return False

def load_from_cache(cache_key: str, ttl: int = 86400, cache_dir: Optional[str] = None) -> Optional[Any]:
    """
    Load data from the cache, with TTL in seconds (default 24 hours)
    """
    if cache_dir is None:
        cache_dir = os.environ.get("CACHE_DIR", "./data/cache")
    
    cache_path = os.path.join(cache_dir, f"{cache_key}.pkl")
    
    if not os.path.exists(cache_path):
        return None
    
    try:
        with open(cache_path, 'rb') as f:
            cache_data = pickle.load(f)
        
        # Check if the cache is expired
        if (datetime.datetime.now() - cache_data['timestamp']).total_seconds() > ttl:
            logger.debug(f"Cache expired for key: {cache_key}")
            return None
        
        return cache_data['data']
    except Exception as e:
        logger.error(f"Error loading from cache: {e}")
        return None

def make_api_request(url: str, params: Dict = None, headers: Dict = None, 
                  method: str = "GET", data: Dict = None, json_data: Dict = None,
                  use_cache: bool = True, cache_ttl: int = 3600,
                  timeout: int = 10, max_retries: int = 3) -> Union[Dict, List]:
    """
    Make an API request with caching and error handling
    """
    # Generate cache key
    if use_cache:
        # Create safe copies of parameters for logging and caching
        safe_params = None
        if params:
            safe_params = params.copy()
            # Mask API keys in params
            for key in safe_params:
                if any(api_key_name in key.lower() for api_key_name in ['api_key', 'apikey', 'key', 'token', 'secret']):
                    safe_params[key] = "********"
        
        cache_key = get_cache_key({
            'url': url,
            'params': params,  # Use original params for accurate cache key
            'headers': headers,
            'method': method,
            'data': data,
            'json_data': json_data
        })
        
        # Check if we have a cached response
        cached_response = load_from_cache(cache_key, ttl=cache_ttl)
        if cached_response is not None:
            logger.info(f"Cache hit for {url}")
            return cached_response
    
    # Set up headers
    if headers is None:
        headers = {
            'User-Agent': 'DrugApprovalAgent/1.0'
        }
    
    # Create safe copies for logging
    safe_url = url
    safe_params = params.copy() if params else None
    safe_headers = headers.copy() if headers else None
    
    # Mask API keys in URLs, params, and headers for logging
    if safe_params:
        for key in safe_params:
            if any(api_key_name in key.lower() for api_key_name in ['api_key', 'apikey', 'key', 'token', 'secret']):
                safe_params[key] = "********"
    
    if safe_headers:
        for key in safe_headers:
            if any(api_key_name in key.lower() for api_key_name in ['api_key', 'apikey', 'key', 'token', 'secret', 'authorization']):
                safe_headers[key] = "********"
    
    # Retry mechanism
    retries = 0
    while retries < max_retries:
        try:
            if method.upper() == "GET":
                response = requests.get(url, params=params, headers=headers, timeout=timeout)
            elif method.upper() == "POST":
                response = requests.post(url, params=params, headers=headers, 
                                        data=data, json=json_data, timeout=timeout)
            else:
                raise ValueError(f"Unsupported HTTP method: {method}")
            
            # Check if we got a successful response
            response.raise_for_status()
            
            try:
                result = response.json()
            except Exception as e:
                logger.error(f"Failed to parse JSON response: {e}")
                # Use sanitized URL for logging
                logger.debug(f"Response from {safe_url} with params {safe_params}: {response.text[:1000]}")
                result = {"error": f"Failed to parse response: {str(e)}"}
            
            # Save to cache if successful
            if use_cache and 'error' not in result:
                save_to_cache(result, cache_key)
            
            return result
            
        except requests.exceptions.RequestException as e:
            retries += 1
            logger.warning(f"API request failed (attempt {retries}/{max_retries}): {e}")
            
            # Only retry on certain error types
            if isinstance(e, (requests.exceptions.ConnectionError, 
                            requests.exceptions.Timeout, 
                            requests.exceptions.TooManyRedirects)) and retries < max_retries:
                # Exponential backoff
                wait_time = 2 ** retries
                logger.info(f"Retrying in {wait_time} seconds...")
                time.sleep(wait_time)
                continue
            
            if isinstance(e, requests.exceptions.HTTPError):
                status_code = e.response.status_code
                logger.error(f"HTTP Error: {status_code}")
                
                # Provide more context based on status code
                if status_code == 404:
                    error_msg = "Resource not found (404)"
                elif status_code == 401:
                    error_msg = "Authentication required (401)"
                elif status_code == 403:
                    error_msg = "Access forbidden (403)"
                elif status_code == 429:
                    error_msg = "Rate limit exceeded (429)"
                elif status_code >= 500:
                    error_msg = f"Server error ({status_code})"
                else:
                    error_msg = f"HTTP error ({status_code})"
                
                # Try to get more error details from response
                try:
                    response_json = e.response.json()
                    if 'error' in response_json:
                        error_msg += f": {response_json['error']}"
                except:
                    # If we can't parse JSON, use the text
                    if e.response.text:
                        error_msg += f": {e.response.text[:100]}..."
                
                return {"error": error_msg}
            
            # General error fallback
            return {"error": f"API request failed: {str(e)}"}
    
    # If we exhausted all retries
    return {"error": "Maximum retries exceeded"}

def extract_date_range(text: str) -> Dict[str, str]:
    """
    Extract date range from text using simple patterns
    Returns a dictionary with 'start_date' and 'end_date' keys
    """
    from datetime import datetime, timedelta
    import re
    
    today = datetime.now().date()
    
    # Default to last 2 years
    result = {
        'start_date': format_date(today - timedelta(days=365*2)),
        'end_date': format_date(today)
    }
    
    # Look for common date patterns
    year_pattern = r'\b(20\d{2})\b'
    years = re.findall(year_pattern, text)
    
    if years:
        years = [int(y) for y in years]
        years.sort()
        
        if len(years) == 1:
            # If only one year mentioned, use that as the start
            result['start_date'] = format_date(datetime(years[0], 1, 1))
        elif len(years) >= 2:
            # Use the first and last year mentioned
            result['start_date'] = format_date(datetime(years[0], 1, 1))
            result['end_date'] = format_date(datetime(years[-1], 12, 31))
    
    # Look for "last X years/months"
    last_pattern = r'last\s+(\d+)\s+(year|month|day)s?'
    match = re.search(last_pattern, text, re.IGNORECASE)
    
    if match:
        num = int(match.group(1))
        unit = match.group(2).lower()
        
        if unit == 'year':
            result['start_date'] = format_date(today - timedelta(days=365*num))
        elif unit == 'month':
            result['start_date'] = format_date(today - timedelta(days=30*num))
        elif unit == 'day':
            result['start_date'] = format_date(today - timedelta(days=num))
    
    return result

def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace and normalizing
    """
    import re
    
    # Replace newlines and tabs with spaces
    text = re.sub(r'[\n\t\r]+', ' ', text)
    
    # Replace multiple spaces with a single space
    text = re.sub(r'\s+', ' ', text)
    
    # Strip leading/trailing whitespace
    text = text.strip()
    
    return text 