# ✨ Website Content Rewriter

A powerful web application for automatically rewriting HTML content using AI and replacing placeholder images with real images from multiple sources.

![Streamlit](https://img.shields.io/badge/Streamlit-FF4B4B?style=for-the-badge&logo=Streamlit&logoColor=white)
![Python](https://img.shields.io/badge/Python-3.8+-3776AB?style=for-the-badge&logo=python&logoColor=white)

## 🚀 Features

- **🤖 Multiple AI Providers**: Support for Google Gemini and OpenAI GPT
- **🌍 Multi-language**: Support for 7 languages (French, English, Arabic, Spanish, German, Italian, Portuguese)
- **🎨 Tone Control**: 6 different writing tones (Professional, Friendly, Casual, Formal, Persuasive, Informative)
- **🖼️ Multiple Image Sources**: Unsplash, Pexels, Pixabay, Openverse, Flickr, AI Generate, Picsum
- **📝 Smart Content Rewriting**: Maintains original text length while improving quality
- **🖼️ Image Replacement**: Replaces placeholder images with real images from selected sources
- **📥 Easy Download**: Download processed files individually or as ZIP
- **🌐 Online Ready**: Deploy to Streamlit Cloud, Heroku, AWS, or any platform

## 📋 Requirements

- Python 3.8 or higher
- API key from either:
  - [Google Gemini](https://aistudio.google.com/app/apikey) (free tier available)
  - [OpenAI](https://platform.openai.com/api-keys) (paid)

### Supported AI Models

**Gemini Models:**
- `models/gemini-2.5-flash` (Default - Fast & Free)
- `models/gemini-2.5-pro` (More powerful, paid)
- `models/gemini-pro-latest` (Latest Pro version)
- `models/gemini-flash-latest` (Latest Flash version)
- `models/gemini-2.0-flash` (Stable version)
- `models/gemini-2.0-flash-lite` (Lightweight version)
- `models/gemini-3-pro-preview` (Preview)
- `models/gemini-3-flash-preview` (Preview)

**OpenAI Models:**
- `gpt-4o` (Latest GPT-4)
- `gpt-4o-mini` (Faster, cheaper)
- `gpt-3.5-turbo` (Legacy)

## 🛠️ Installation

1. **Navigate to the project folder:**
```bash
cd website-content-rewriter
```

2. **Create a virtual environment (recommended):**
```bash
python -m venv venv
# Windows
venv\Scripts\activate
# Linux/Mac
source venv/bin/activate
```

3. **Install dependencies:**
```bash
pip install -r requirements.txt
```

4. **Run the application:**
```bash
streamlit run app.py
```

Or use the provided scripts:
- **Windows**: `run_streamlit.bat`
- **Linux/Mac**: `chmod +x run_streamlit.sh && ./run_streamlit.sh`

The app will open in your browser at `http://localhost:8501`

## 📖 Usage

1. **Upload HTML Files**: Use the file uploader in the sidebar to select one or more HTML files
2. **Configure AI Provider**: 
   - Select Gemini or OpenAI
   - Enter your API key
   - Adjust model settings (optional)
3. **Set Content Settings**:
   - Choose language and tone
   - Enter brand information, city, keywords, etc.
4. **Select Image Sources**: Choose which image sources to use
5. **Start Processing**: Click "Start Processing" button
6. **View Results**: 
   - Check the logs for progress
   - View the image gallery
   - Download processed files

## 🌐 Deploying Online

### Streamlit Cloud (Recommended - Free)

1. Push your code to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your GitHub repository
4. Deploy!

### Other Platforms

The app can also be deployed on:
- **Heroku**: Use the Procfile and requirements.txt
- **AWS**: Deploy using AWS App Runner or EC2
- **Google Cloud Platform**: Use Cloud Run
- **Any platform** that supports Python and Streamlit

## 🎯 Features in Detail

### AI Content Rewriting
- ✅ Maintains original text length (±10%)
- ✅ Completes sentences properly
- ✅ Handles headers and titles correctly
- ✅ Natural, professional writing style
- ✅ SEO-optimized content

### Image Processing
- ✅ Replaces placeholder images
- ✅ Handles CSS background-images
- ✅ Inserts images into empty containers (optional)
- ✅ Optimizes alt text for SEO
- ✅ Multiple image source support

### Real-time Logging
- ✅ **Live log updates** - See logs appear in real-time during processing
- ✅ View all logs with timestamps and emoji indicators
- ✅ Error and warning messages highlighted
- ✅ Automatic log display updates without page refresh
- ✅ Last 200 logs displayed for optimal performance

### Rate Limiting & Error Handling
- ✅ Automatic handling of 429 (Rate Limit) errors
- ✅ Respects API retryDelay values from error responses
- ✅ Exponential backoff for retries
- ✅ Graceful error handling with clear messages

## 📁 Project Structure

```
website-content-rewriter/
├── app.py                   # Main Streamlit application
├── requirements.txt         # Python dependencies
├── README.md               # This file
├── .gitignore              # Git ignore rules
├── run_streamlit.bat       # Windows startup script
├── run_streamlit.sh        # Linux/Mac startup script
├── src/                    # Source code
│   ├── ai_providers.py     # AI provider abstraction (Gemini, OpenAI)
│   └── writer.py           # Core HTML processing logic
├── examples/               # Example HTML files
│   └── sample.html         # Sample HTML file for testing
└── uploads/                # Uploaded files (auto-created)
```

## 🔧 Configuration

### Environment Variables (Optional)

You can create a `.env` file in the project root:

```env
GEMINI_API_KEY=your_gemini_api_key_here
OPENAI_API_KEY=your_openai_api_key_here
```

### Default Settings

- **Language**: French (fr)
- **Tone**: Professional
- **Image Sources**: Unsplash, Pexels, Picsum
- **Backup**: Enabled by default

## 🐛 Troubleshooting

### Common Issues

**API Key Error:**
- Make sure your API key is correct and has sufficient credits
- For Gemini: Get your key from [Google AI Studio](https://aistudio.google.com/app/apikey)
- For OpenAI: Get your key from [OpenAI Platform](https://platform.openai.com/api-keys)

**Model Not Found (404 Error):**
- The app automatically uses supported models
- If you see 404 errors, the selected model may not be available for your API key
- Try selecting a different model from the dropdown

**Rate Limiting (429 Error):**
- The app automatically handles rate limits by respecting API retryDelay values
- If you hit rate limits frequently, increase the "Request Delay" setting
- Consider using Gemini Flash (free tier) for testing

**Import Errors:**
- Ensure all dependencies are installed: `pip install -r requirements.txt`
- Make sure you're using Python 3.8 or higher
- If using `google-genai`, ensure version >= 0.2.0

**Image Not Loading:**
- Some image sources may be rate-limited, try different sources
- Check your internet connection
- Some sources may require API keys (not currently implemented)

**Processing Slow:**
- Increase request delay in settings
- Reduce number of files processed at once
- Use faster models (e.g., Gemini Flash instead of Pro)
- Check your API quota limits

**Logs Not Updating:**
- Logs update in real-time during processing
- If logs don't appear, check browser console for errors
- Ensure Streamlit version >= 1.28.0

## 📝 Notes

- **File Storage**: Processed files are saved in the `uploads/` folder (auto-created)
- **Backups**: Backups are created automatically (if enabled) with timestamp
- **Processing**: Files are processed sequentially for better control and error handling
- **Images**: All image URLs are external (no local storage needed)
- **Logs**: Logs are stored in session state and update in real-time
- **API Usage**: The app respects rate limits and uses exponential backoff for retries
- **Model Selection**: Use the dropdown to select from supported models (prevents 404 errors)

## 🔄 Recent Updates

- ✅ **New Gemini SDK**: Updated to use `google-genai` (official new SDK)
- ✅ **Latest Models**: Support for Gemini 2.5 and 3.0 models
- ✅ **Live Logging**: Real-time log updates during processing
- ✅ **Rate Limit Handling**: Automatic 429 error handling with retryDelay support
- ✅ **Model Validation**: Prevents invalid model selection
- ✅ **Improved Error Messages**: Clear, actionable error messages

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available for personal and commercial use.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing framework
- [Google Gemini](https://ai.google.dev/) for AI capabilities (using official `google-genai` SDK)
- [OpenAI](https://openai.com/) for AI capabilities
- Image sources: Unsplash, Pexels, Pixabay, Openverse, Flickr, Picsum

## 🔗 Links

- **Gemini API**: [Google AI Studio](https://aistudio.google.com/)
- **OpenAI API**: [OpenAI Platform](https://platform.openai.com/)
- **Streamlit**: [Streamlit Documentation](https://docs.streamlit.io/)

## 📧 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Made with ❤️ for content creators and developers**
