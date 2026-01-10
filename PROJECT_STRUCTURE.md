# Project Structure

This document explains the organization of the SEO Content Rewriter project.

## Directory Structure

```
seo-content-rewriter/
├── streamlit_app.py          # Main Streamlit application (entry point)
├── requirements.txt          # Python dependencies
├── README.md                 # Main documentation
├── .gitignore                # Git ignore rules
├── run_streamlit.bat         # Windows startup script
├── run_streamlit.sh          # Linux/Mac startup script
│
├── src/                      # Source code package
│   ├── __init__.py          # Package initialization
│   ├── ai_providers.py     # AI provider abstraction layer
│   │                        # - Supports Gemini and OpenAI
│   │                        # - Handles language and tone
│   └── writer.py            # Core HTML processing logic
│                            # - Text rewriting
│                            # - Image extraction and replacement
│                            # - Section identification
│                            # - Background image handling
│
└── examples/                # Example HTML files (optional)
    └── index.html           # Sample HTML file for testing
```

## File Descriptions

### Core Application Files

- **streamlit_app.py**: Main Streamlit application with UI and processing logic
- **requirements.txt**: All Python package dependencies

### Source Code (src/)

- **ai_providers.py**: 
  - Abstract base class for AI providers
  - GeminiProvider implementation
  - OpenAIProvider implementation
  - Factory function for creating providers

- **writer.py**:
  - HTML parsing and processing
  - Text node extraction
  - Image extraction (both `<img>` tags and CSS background-images)
  - Section identification
  - Backup creation
  - Image placeholder detection

### Configuration Files

- **.gitignore**: Excludes unnecessary files from version control
- **README.md**: Complete documentation and usage guide

### Scripts

- **run_streamlit.bat**: Windows batch script to start the app
- **run_streamlit.sh**: Linux/Mac shell script to start the app

## What's Excluded

The following are excluded from the repository (via .gitignore):

- `__pycache__/` - Python cache files
- `venv/` or `env/` - Virtual environments
- `uploads/` - User-uploaded files
- `*.backup_*` - Backup files
- `.env` - Environment variables (API keys)
- `*.log` - Log files

## Adding New Features

To add new features:

1. **New AI Provider**: Add to `src/ai_providers.py`
2. **New Image Source**: Add function to `streamlit_app.py` and update `get_image_from_source()`
3. **New Processing Logic**: Add to `src/writer.py`
4. **UI Changes**: Modify `streamlit_app.py`

## Dependencies

All dependencies are listed in `requirements.txt`. Main dependencies:

- `streamlit` - Web framework
- `beautifulsoup4` - HTML parsing
- `google-generativeai` - Gemini API
- `openai` - OpenAI API
- `requests` - HTTP requests for image sources
