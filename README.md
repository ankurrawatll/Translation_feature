# 🌐 Multilingual Chatbot (Indic + Gemini 2.0 Flash)

A Streamlit-based multilingual chatbot that supports 10 Indian languages and uses Google's Gemini 2.0 Flash for intelligent conversations.

## ✨ Features

- **Multilingual Support**: Hindi, Punjabi, Gujarati, Tamil, Telugu, Malayalam, Bengali, Marathi, Kannada, and English
- **AI-Powered**: Powered by Google Gemini 2.0 Flash for intelligent responses
- **Voice Input/Output**: Speech-to-Text (STT) and Text-to-Speech (TTS) capabilities
- **Real-time Translation**: Automatic translation between languages using Gemini
- **Localized UI**: Interface adapts to the selected language
- **Responsive Design**: Modern, user-friendly interface with chat bubbles

## 🚀 How It Works

1. **Language Selection**: Choose your preferred language from the dropdown
2. **Input**: Type your message or use voice input (🎤 button)
3. **Processing**: Your message is translated to English, sent to Gemini, and translated back
4. **Response**: Receive AI-generated responses in your selected language
5. **Voice Output**: Listen to responses using the 🔊 button

## 🛠️ Installation

### Local Development

1. Clone the repository:
```bash
git clone https://github.com/yourusername/Translation_feature.git
cd Translation_feature
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Set up your Gemini API key:
   - Get an API key from [Google AI Studio](https://makersuite.google.com/app/apikey)
   - Update the `GEMINI_API_KEY` in `app.py`

4. Run the app:
```bash
streamlit run app.py
```

### Streamlit Cloud Deployment

1. Push your code to GitHub
2. Connect your repository to [Streamlit Community Cloud](https://streamlit.io/cloud)
3. Deploy automatically

## 🔧 Configuration

- **Gemini API Key**: Required for AI responses
- **Supported Languages**: 10 Indian languages with proper BCP-47 tags for TTS/STT
- **Model**: Uses Gemini 2.0 Flash for optimal performance

## 📱 Usage Examples

- Ask questions in Hindi: "मौसम कैसा है?"
- Get explanations in Tamil: "கணினி என்றால் என்ன?"
- Request summaries in Bengali: "এই বই সম্পর্কে সংক্ষেপে বলুন"

## 🌍 Supported Languages

| Language Code | Language Name | TTS/STT Tag |
|---------------|----------------|-------------|
| `eng_Latn` | English | `en-US` |
| `hin_Deva` | Hindi | `hi-IN` |
| `ben_Beng` | Bengali | `bn-IN` |
| `guj_Gujr` | Gujarati | `gu-IN` |
| `mar_Deva` | Marathi | `mr-IN` |
| `pan_Guru` | Punjabi | `pa-IN` |
| `tam_Taml` | Tamil | `ta-IN` |
| `tel_Telu` | Telugu | `te-IN` |
| `mal_Mlym` | Malayalam | `ml-IN` |
| `kan_Knda` | Kannada | `kn-IN` |

## 🔒 Privacy & Security

- Inputs are processed through Google's Gemini API
- No data is stored locally beyond the current session
- Avoid sharing sensitive personal information
- API keys should be kept secure

## 🎯 Use Cases

- **Language Learning**: Practice conversations in different Indian languages
- **Customer Support**: Multilingual customer service chatbots
- **Education**: Educational content in regional languages
- **Accessibility**: Making AI accessible to non-English speakers

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## 📄 License

This project is open source and available under the [MIT License](LICENSE).

## 🙏 Acknowledgments

- Google Gemini 2.0 Flash for AI capabilities
- Streamlit for the web framework
- AI4Bharat for Indic language support concepts

## 📞 Support

For issues and questions:
- Create an issue on GitHub
- Check the Streamlit documentation
- Review Gemini API documentation

---

**Note**: Make sure to replace `yourusername` in the clone URL with your actual GitHub username.
