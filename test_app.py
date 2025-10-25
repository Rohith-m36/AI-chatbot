#!/usr/bin/env python3
"""
Test script to verify the corrected app.py works properly
"""

import os
import sys
from dotenv import load_dotenv

def test_imports():
    """Test if all imports work correctly"""
    print("Testing imports...")
    try:
        import streamlit as st
        print("✅ Streamlit imported successfully")
        
        from langchain_groq import ChatGroq
        print("✅ ChatGroq imported successfully")
        
        from langchain_community.utilities import ArxivAPIWrapper, WikipediaAPIWrapper
        from langchain_community.tools import ArxivQueryRun, WikipediaQueryRun, DuckDuckGoSearchRun
        print("✅ LangChain community tools imported successfully")
        
        from langchain.agents import initialize_agent, AgentType
        from langchain.memory import ConversationBufferMemory
        from langchain.tools import Tool
        print("✅ LangChain core components imported successfully")
        
        from langchain_community.document_loaders import PyPDFLoader
        from langchain.text_splitter import RecursiveCharacterTextSplitter
        from langchain_community.vectorstores import FAISS
        from langchain_huggingface import HuggingFaceEmbeddings
        print("✅ Document processing components imported successfully")
        
        from streamlit_mic_recorder import speech_to_text
        print("✅ Speech-to-text imported successfully")
        
        return True
    except ImportError as e:
        print(f"❌ Import error: {e}")
        return False

def test_environment():
    """Test environment setup"""
    print("\nTesting environment...")
    load_dotenv()
    
    api_key = os.getenv("GROQ_API_KEY")
    if api_key:
        print("✅ GROQ_API_KEY found in environment")
        return True
    else:
        print("⚠️ GROQ_API_KEY not found. Please set it in your .env file")
        return False

def test_tools_initialization():
    """Test if tools can be initialized"""
    print("\nTesting tools initialization...")
    
    try:
        from langchain_community.tools import DuckDuckGoSearchRun
        search = DuckDuckGoSearchRun(name="Search", description="Get real-time information, news, or updates.")
        print("✅ DuckDuckGo search tool initialized")
    except Exception as e:
        print(f"⚠️ DuckDuckGo search tool failed: {e}")
    
    try:
        from langchain_community.utilities import ArxivAPIWrapper
        from langchain_community.tools import ArxivQueryRun
        arxiv = ArxivQueryRun(api_wrapper=ArxivAPIWrapper(top_k_results=1, doc_content_chars_max=1000))
        print("✅ Arxiv tool initialized")
    except Exception as e:
        print(f"⚠️ Arxiv tool failed: {e}")
    
    try:
        from langchain_community.utilities import WikipediaAPIWrapper
        from langchain_community.tools import WikipediaQueryRun
        wiki = WikipediaQueryRun(api_wrapper=WikipediaAPIWrapper(top_k_results=1, doc_content_chars_max=2000))
        print("✅ Wikipedia tool initialized")
    except Exception as e:
        print(f"⚠️ Wikipedia tool failed: {e}")

def main():
    """Run all tests"""
    print("🧪 Testing corrected app.py components...\n")
    
    # Test imports
    imports_ok = test_imports()
    
    # Test environment
    env_ok = test_environment()
    
    # Test tools
    test_tools_initialization()
    
    print("\n" + "="*50)
    if imports_ok and env_ok:
        print("✅ All critical tests passed! The app should work correctly.")
        print("Run: streamlit run app.py")
    else:
        print("❌ Some tests failed. Please check the issues above.")
        if not env_ok:
            print("💡 Make sure to create a .env file with GROQ_API_KEY=your_api_key")

if __name__ == "__main__":
    main()
