# AI Chatbot with PDF Processing

A Streamlit-based AI chatbot that can search the web, query academic papers, and process PDF documents using LangChain and Groq.

## Features

- 🤖 **AI Chat**: Powered by Groq's Llama 3.1 model
- 📄 **PDF Processing**: Upload and query PDF documents
- 🔍 **Web Search**: Real-time information via DuckDuckGo
- 📚 **Academic Search**: Query Arxiv papers and Wikipedia
- 🎤 **Voice Input**: Speech-to-text functionality
- 💬 **Chat Memory**: Maintains conversation context

## Setup

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Environment Configuration

Create a `.env` file in the project root:

```env
GROQ_API_KEY=your_groq_api_key_here
```

Get your Groq API key from: https://console.groq.com/

### 3. Run the Application

```bash
streamlit run app.py
```

## Usage

### Basic Chat
- Type messages in the chat input
- Use voice input by clicking the microphone button
- The AI will respond using web search, academic sources, or general knowledge

### PDF Processing
1. Upload a PDF file using the sidebar
2. Wait for processing confirmation
3. Ask questions about the document content
4. Use keywords like "pdf", "document", "file" to specifically query the PDF

### Smart Routing
The app automatically routes queries based on content:
- **PDF queries**: Contains "pdf", "document", "file", "uploaded"
- **Academic queries**: Starts with "arxiv:"
- **News/Current events**: Contains "latest", "current", "today", "news", "breaking"
- **General queries**: Uses the full agent with all tools

## Troubleshooting

### Common Issues

1. **Import Errors**
   ```bash
   pip install --upgrade langchain langchain-community langchain-groq
   ```

2. **GROQ_API_KEY Missing**
   - Ensure your `.env` file exists and contains the API key
   - Restart the application after adding the key

3. **PDF Processing Fails**
   - Ensure the PDF contains extractable text (not just images)
   - Try with a different PDF file

4. **Tools Unavailable**
   - Check internet connection
   - Some tools may have rate limits

### Testing

Run the test script to verify everything is working:

```bash
python test_app.py
```

## Code Corrections Made

The following issues were fixed in the original code:

1. **Import Corrections**:
   - Fixed deprecated imports from `langchain.document_loaders` → `langchain_community.document_loaders`
   - Fixed deprecated imports from `langchain.vectorstores` → `langchain_community.vectorstores`

2. **Error Handling**:
   - Added try-catch blocks for LLM initialization
   - Added error handling for tool initialization
   - Added robust PDF processing with cleanup
   - Added error handling for agent execution

3. **Robustness Improvements**:
   - Added API key validation
   - Added graceful degradation when tools fail
   - Added proper temporary file cleanup
   - Added user-friendly error messages

## Architecture

```
app.py
├── Configuration & Setup
├── Session State Management
├── Tool Initialization (Search, Arxiv, Wikipedia)
├── PDF Processing Pipeline
├── Agent Creation
├── Chat Interface
└── Smart Query Routing
```

## Dependencies

Key dependencies include:
- `streamlit`: Web interface
- `langchain`: LLM framework
- `langchain-groq`: Groq integration
- `langchain-community`: Community tools
- `faiss-cpu`: Vector database
- `sentence-transformers`: Embeddings
- `pypdf`: PDF processing

## License

This project is open source and available under the MIT License.
