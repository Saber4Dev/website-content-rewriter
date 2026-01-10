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
  - [Google Gemini](https://makersuite.google.com/app/apikey) (free)
  - [OpenAI](https://platform.openai.com/api-keys) (paid)

## 🛠️ Installation

1. **Navigate to the project folder:**
```bash
cd website-content-rewriter
```

2. **Install dependencies:**
```bash
pip install -r requirements.txt
```

3. **Run the application:**
```bash
streamlit run streamlit_app.py
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
- ✅ See processing progress in real-time
- ✅ View all logs with timestamps
- ✅ Error and warning messages

## 📁 Project Structure

```
website-content-rewriter/
├── streamlit_app.py          # Main Streamlit application
├── requirements.txt          # Python dependencies
├── README.md                # This file
├── .gitignore               # Git ignore rules
├── run_streamlit.bat        # Windows startup script
├── run_streamlit.sh         # Linux/Mac startup script
├── PROJECT_STRUCTURE.md      # Detailed structure documentation
├── GITHUB_SETUP.md          # GitHub setup guide
├── src/                     # Source code
│   ├── __init__.py
│   ├── ai_providers.py      # AI provider abstraction (Gemini, OpenAI)
│   └── writer.py            # Core HTML processing logic
└── examples/                # Example HTML files
    ├── README.md
    └── sample.html          # Sample HTML file for testing
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

- **API Key Error**: Make sure your API key is correct and has sufficient credits
- **Import Errors**: Ensure all dependencies are installed: `pip install -r requirements.txt`
- **Image Not Loading**: Some image sources may be rate-limited, try different sources
- **Processing Slow**: Increase request delay or reduce number of files processed at once

## 📝 Notes

- Processed files are saved in the `uploads/` folder (auto-created)
- Backups are created automatically (if enabled)
- The app processes files sequentially for better control
- All image URLs are external (no local storage needed)

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License

This project is open source and available for personal and commercial use.

## 🙏 Acknowledgments

- [Streamlit](https://streamlit.io/) for the amazing framework
- [Google Gemini](https://ai.google.dev/) and [OpenAI](https://openai.com/) for AI capabilities
- Image sources: Unsplash, Pexels, Pixabay, Openverse, Flickr

## 📧 Support

For issues, questions, or contributions, please open an issue on GitHub.

---

**Made with ❤️ for content creators and developers**
