
<h2>Install the dependencies</h2>
<p>
<pre><code>sudo dnf install git python3 python3-pip -y</code></pre>
<p>
<h2>Install Python packages</h2>
<pre><code>pip install langchain langchain-community faiss-cpu openai pypdf gradio llama-cpp-python sentence-transformers</code></pre>
<p>
<h2>Package	Purpose</h2><p>
langchain	- Framework for building LLM-powered applications using chains of tools, prompts, models, and memory. Think of it like glue between different AI components.<p>
langchain-community	- A companion to langchain with integrations and tools contributed by the community — e.g., wrappers for models, tools, or document loaders.<p>
faiss-cpu	- Facebook's library for vector similarity search. Useful for embedding-based retrieval (like RAG). This is the CPU-only version.<p>
openai	- Official SDK for calling OpenAI’s API (e.g., GPT-4, ChatGPT, embeddings).<p>
pypdf -	Used to read and extract text from PDF files. Great for ingesting documents into an LLM pipeline.<p>
gradio - A Python library to build simple web UIs for ML apps (drag-and-drop, chat interface, etc.) — great for demos and testing.<p>
llama-cpp-python	- Python bindings for running LLaMA models locally using C++ backend. Supports quantized GGUF models for low-resource inference.<p>
sentence-transformers	 - Library for creating semantic embeddings using pre-trained transformer models (e.g., for search, similarity, clustering).<p>
<p><p><p>
<h3>This command installs Ollama, a tool for running LLMs (like LLaMA, Mistral, etc.) locally on your machine.</h3><p>
<pre><code>curl -fsSL https://ollama.com/install.sh | sh</code></pre><p>

<h3>This command runs a Python script called my_doctor.py using Python 3.</h3><p>
<pre><code>python3 my_doctor.py</code></pre>
<p>
BE PATIENT - Wait as the model is trained on the PDF's - it will look like this:

<code>user@blackbox:~/Projects/myPhysician$ python3 my_doctor.py</code>
<code>/Projects/myPhysician/my_doctor.py:4: LangChainDeprecationWarning: Importing HuggingFaceEmbeddings from langchain.embeddings is deprecated. Please replace deprecated imports:</code>
<code></code>
<code>>> from langchain.embeddings import HuggingFaceEmbeddings</code>
<code></code>
<code>with new imports of:</code>
<code></code>
<code>>> from langchain_community.embeddings import HuggingFaceEmbeddings</code>
<code>You can use the langchain cli to **automatically** upgrade many imports. Please see documentation here <https://python.langchain.com/docs/versions/v0_2/></code>
<code>  from langchain.embeddings import HuggingFaceEmbeddings</code>
<code>/home/jmoses/Projects/myPhysician/my_doctor.py:5: LangChainDeprecationWarning: Importing FAISS from langchain.vectorstores is deprecated. Please replace deprecated imports:</code>
<code></code>
<code>>> from langchain.vectorstores import FAISS</code>
<code></code>
<code>with new imports of:</code>
<code></code>
<code>>> from langchain_community.vectorstores import FAISS</code>
<code>You can use the langchain cli to **automatically** upgrade many imports. Please see documentation here <https://python.langchain.com/docs/versions/v0_2/></code>
<code>  from langchain.vectorstores import FAISS</code>
<code>/home/jmoses/.local/lib/python3.13/site-packages/langchain/llms/__init__.py:549: LangChainDeprecationWarning: Importing LLMs from langchain is deprecated. Importing from langchain will no longer be supported as of langchain==0.2.0. Please import from langchain-community instead:</code>
<code></code>
<code>`from langchain_community.llms import Ollama`.</code>
<code></code>
<code>To install langchain-community run `pip install -U langchain-community`.</code>
<code>  warnings.warn(</code>
<code>/home/user/Projects/myPhysician/my_doctor.py:25: LangChainDeprecationWarning: The class `HuggingFaceEmbeddings` was deprecated in LangChain 0.2.2 and will be removed in 1.0. An updated version of the class exists in the :class:`~langchain-huggingface package and should be used instead. To use it run `pip install -U :class:`~langchain-huggingface` and import as `from :class:`~langchain_huggingface import HuggingFaceEmbeddings``.</code>
<code>  embedding = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")</code>
<code>/home/jmoses/Projects/myPhysician/my_doctor.py:31: LangChainDeprecationWarning: The class `Ollama` was deprecated in LangChain 0.3.1 and will be removed in 1.0.0. An updated version of the class exists in the :class:`~langchain-ollama package and should be used instead. To use it run `pip install -U :class:`~langchain-ollama` and import as `from :class:`~langchain_ollama import OllamaLLM``.</code>
<code>  llm = Ollama(model="mistral")</code>
<code>Running on local URL:  http://127.0.0.1:7860</code>
<code></code>
<code>To create a public link, set `share=True` in `launch()`.</code>
