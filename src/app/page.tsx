'use client';

import { useState, useRef, useEffect } from 'react';
import { Upload, Play, Loader2, FileText, Send, Trash2, User, Bot, Download, Save, Plus, MessageSquare, Menu, X } from 'lucide-react';

interface ProgressCallback {
  loaded: number;
  total: number;
}

interface WllamaConfig {
  n_ctx: number;
  n_batch: number;
  n_threads: number;
  n_gpu_layers: number;
  use_mlock: boolean;
  use_mmap: boolean;
  progressCallback: (progress: ProgressCallback) => void;
}

interface Wllama {
  loadModel: (blobs: File[], config: WllamaConfig) => Promise<void>;
  createCompletion: (prompt: string, options: any) => Promise<string>;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
}

interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

export default function WllamaUI() {
  const [modelFile, setModelFile] = useState<FileList | null>(null);
  const [isLoading, setIsLoading] = useState(false);
  const [loadProgress, setLoadProgress] = useState(0);
  const [input, setInput] = useState('');
  const [conversations, setConversations] = useState<Conversation[]>([]);
  const [currentConversationId, setCurrentConversationId] = useState<string | null>(null);
  const [error, setError] = useState('');
  const [status, setStatus] = useState('');
  const [nCtx, setNCtx] = useState(4096);
  const [isGenerating, setIsGenerating] = useState(false);
  const [sidebarOpen, setSidebarOpen] = useState(true);
  const wllamaRef = useRef<Wllama | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  // Load conversations from localStorage on mount
  useEffect(() => {
    const savedConversations = localStorage.getItem('wllama-conversations');
    if (savedConversations) {
      try {
        const parsed = JSON.parse(savedConversations);
        const conversationsWithDates = parsed.map((conv: any) => ({
          ...conv,
          createdAt: new Date(conv.createdAt),
          updatedAt: new Date(conv.updatedAt),
          messages: conv.messages.map((msg: any) => ({
            ...msg,
            timestamp: new Date(msg.timestamp)
          }))
        }));
        setConversations(conversationsWithDates);
        
        // Load last active conversation
        const lastActiveId = localStorage.getItem('wllama-last-conversation-id');
        if (lastActiveId && conversationsWithDates.find((c: Conversation) => c.id === lastActiveId)) {
          setCurrentConversationId(lastActiveId);
        } else if (conversationsWithDates.length > 0) {
          setCurrentConversationId(conversationsWithDates[0].id);
        }
      } catch (err) {
        console.error('Failed to load conversations:', err);
      }
    }
  }, []);

  // Save conversations to localStorage whenever they change
  useEffect(() => {
    if (conversations.length > 0) {
      localStorage.setItem('wllama-conversations', JSON.stringify(conversations));
    }
  }, [conversations]);

  // Save current conversation ID
  useEffect(() => {
    if (currentConversationId) {
      localStorage.setItem('wllama-last-conversation-id', currentConversationId);
    }
  }, [currentConversationId]);

  const currentConversation = conversations.find(c => c.id === currentConversationId);
  const messages = currentConversation?.messages || [];

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const createNewConversation = () => {
    const newConv: Conversation = {
      id: Date.now().toString(),
      title: 'New Chat',
      messages: [],
      createdAt: new Date(),
      updatedAt: new Date()
    };
    setConversations(prev => [newConv, ...prev]);
    setCurrentConversationId(newConv.id);
  };

  const updateConversationTitle = (conversationId: string, firstMessage: string) => {
    setConversations(prev => prev.map(conv => {
      if (conv.id === conversationId && conv.title === 'New Chat') {
        return {
          ...conv,
          title: firstMessage.slice(0, 30) + (firstMessage.length > 30 ? '...' : '')
        };
      }
      return conv;
    }));
  };

  const deleteConversation = (conversationId: string) => {
    setConversations(prev => prev.filter(c => c.id !== conversationId));
    if (currentConversationId === conversationId) {
      const remaining = conversations.filter(c => c.id !== conversationId);
      setCurrentConversationId(remaining.length > 0 ? remaining[0].id : null);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const hasGguf = Array.from(files).some(f => f.name.endsWith('.gguf'));
      if (hasGguf) {
        setModelFile(files);
        setError('');
        setStatus(`Model file(s) selected: ${files.length} file(s)`);
      } else {
        setError('Please select at least one valid .gguf model file');
        setModelFile(null);
      }
    }
  };

  const loadModel = async () => {
    if (!modelFile) {
      setError('Please select a model file first');
      return;
    }

    setIsLoading(true);
    setError('');
    setLoadProgress(0);
    setStatus('Initializing Wllama...');

    try {
      const WllamaModule = await import('@wllama/wllama/esm/index.js');
      const Wllama = WllamaModule.Wllama;
      
      const CONFIG_PATHS = {
        'single-thread/wllama.wasm': './wllama/esm/single-thread/wllama.wasm',
        'multi-thread/wllama.wasm': './wllama/esm/multi-thread/wllama.wasm',
      };

      wllamaRef.current = new Wllama(CONFIG_PATHS);

      const progressCallback = ({ loaded, total }: ProgressCallback) => {
        const progressPercentage = Math.round((loaded / total) * 100);
        setLoadProgress(progressPercentage);
        setStatus(`Loading model... ${progressPercentage}%`);
      };

      setStatus('Loading model from files...');
      
      const start = Date.now();
      const blobs = Array.from(modelFile);
      
      const config: WllamaConfig = {
        n_ctx: nCtx,
        n_batch: 2048,
        n_threads: 8,
        n_gpu_layers: 0,
        use_mlock: false,
        use_mmap: true,
        progressCallback,
      };
      
      await wllamaRef.current.loadModel(blobs, config);
      
      const took = Date.now() - start;
      setStatus(`Model loaded successfully! (${took} ms)`);
      setLoadProgress(100);
      
      // Create first conversation if none exists
      if (conversations.length === 0) {
        createNewConversation();
      }
    } catch (err: any) {
      setError('Failed to load model: ' + (err?.message || String(err)));
      setStatus('');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const buildConversationPrompt = (messages: Message[], newUserMessage: string) => {
    let prompt = '';
    
    messages.forEach(msg => {
      if (msg.role === 'user') {
        prompt += `<start_of_turn>user\n${msg.content}<end_of_turn>\n`;
      } else {
        prompt += `<start_of_turn>model\n${msg.content}<end_of_turn>\n`;
      }
    });
    
    prompt += `<start_of_turn>user\n${newUserMessage}<end_of_turn>\n<start_of_turn>model\n`;
    
    return prompt;
  };

  const sendMessage = async () => {
    if (!wllamaRef.current) {
      setError('Please load a model first');
      return;
    }

    if (!input.trim()) {
      return;
    }

    // Create new conversation if none exists
    if (!currentConversationId) {
      createNewConversation();
      return;
    }

    const userMessage = input.trim();
    setInput('');
    setIsGenerating(true);
    setError('');

    const newUserMessage: Message = {
      role: 'user',
      content: userMessage,
      timestamp: new Date()
    };

    // Update conversation with user message
    setConversations(prev => prev.map(conv => {
      if (conv.id === currentConversationId) {
        const updatedMessages = [...conv.messages, newUserMessage];
        return {
          ...conv,
          messages: updatedMessages,
          updatedAt: new Date()
        };
      }
      return conv;
    }));

    // Update title if this is the first message
    if (messages.length === 0) {
      updateConversationTitle(currentConversationId, userMessage);
    }

    try {
      const formattedPrompt = buildConversationPrompt(messages, userMessage);

      const outputText = await wllamaRef.current.createCompletion(formattedPrompt, {
        nPredict: 512,
        sampling: {
          temp: 0.7,
          top_k: 40,
          top_p: 0.9,
          repeat_penalty: 1.1,
          repeat_last_n: 64,
        },
      });

      const cleanOutput = outputText.replace(/<end_of_turn>/g, '').trim();

      const assistantMessage: Message = {
        role: 'assistant',
        content: cleanOutput,
        timestamp: new Date()
      };

      setConversations(prev => prev.map(conv => {
        if (conv.id === currentConversationId) {
          return {
            ...conv,
            messages: [...conv.messages, assistantMessage],
            updatedAt: new Date()
          };
        }
        return conv;
      }));
    } catch (err: any) {
      setError('Failed to generate response: ' + (err?.message || String(err)));
      console.error(err);
    } finally {
      setIsGenerating(false);
      inputRef.current?.focus();
    }
  };

  const handleKeyPress = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage();
    }
  };

  const exportChat = () => {
    if (!currentConversation) return;
    
    const chatText = currentConversation.messages.map(msg => 
      `[${msg.timestamp.toLocaleString()}] ${msg.role.toUpperCase()}: ${msg.content}`
    ).join('\n\n');
    
    const blob = new Blob([chatText], { type: 'text/plain' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `${currentConversation.title}-${new Date().toISOString().split('T')[0]}.txt`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-64' : 'w-0'} transition-all duration-300 bg-slate-950/50 backdrop-blur-lg border-r border-white/10 flex flex-col overflow-hidden`}>
        <div className="p-4 border-b border-white/10">
          <h2 className="text-white font-bold text-lg mb-3">Wllama</h2>
          <button
            onClick={createNewConversation}
            disabled={!wllamaRef.current}
            className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            <Plus className="w-4 h-4" />
            New conversation
          </button>
        </div>
        
        <div className="flex-1 overflow-y-auto p-2">
          {conversations.map(conv => (
            <div
              key={conv.id}
              className={`group relative mb-1 p-3 rounded-lg cursor-pointer transition-colors ${
                currentConversationId === conv.id
                  ? 'bg-purple-600/30 border border-purple-500/50'
                  : 'bg-white/5 hover:bg-white/10'
              }`}
              onClick={() => setCurrentConversationId(conv.id)}
            >
              <div className="flex items-start gap-2">
                <MessageSquare className="w-4 h-4 text-purple-300 flex-shrink-0 mt-1" />
                <div className="flex-1 min-w-0">
                  <p className="text-white text-sm truncate">{conv.title}</p>
                  <p className="text-purple-300 text-xs">
                    {conv.messages.length} messages
                  </p>
                </div>
                <button
                  onClick={(e) => {
                    e.stopPropagation();
                    deleteConversation(conv.id);
                  }}
                  className="opacity-0 group-hover:opacity-100 text-red-400 hover:text-red-300 transition-opacity"
                >
                  <Trash2 className="w-4 h-4" />
                </button>
              </div>
            </div>
          ))}
        </div>
      </div>

      {/* Main Content */}
      <div className="flex-1 flex flex-col h-screen">
        {/* Header */}
        <div className="bg-white/10 backdrop-blur-lg border-b border-white/20 p-4">
          <div className="flex items-center gap-3">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="text-white hover:bg-white/10 p-2 rounded-lg transition-colors"
            >
              {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
            <div className="flex-1">
              <h1 className="text-2xl font-bold text-white">
                {currentConversation?.title || 'Wllama Chatbot'}
              </h1>
              <p className="text-purple-200 text-sm">
                {wllamaRef.current ? 'Model loaded - Ready to chat' : 'Load a model to start'}
              </p>
            </div>
          </div>
        </div>

        {/* Model Loading Section */}
        {!wllamaRef.current && (
          <div className="bg-white/5 border-b border-white/10 p-6">
            <div className="max-w-3xl mx-auto">
              <div className="mb-4">
                <label className="block text-white font-semibold mb-3">
                  <FileText className="inline w-5 h-5 mr-2" />
                  Select Model File (.gguf)
                </label>
                <div className="flex gap-3">
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".gguf"
                    multiple
                    onChange={handleFileSelect}
                    className="hidden"
                  />
                  <button
                    onClick={() => fileInputRef.current?.click()}
                    className="flex-1 bg-purple-600 hover:bg-purple-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
                  >
                    <Upload className="w-5 h-5" />
                    Choose File
                  </button>
                  <button
                    onClick={loadModel}
                    disabled={!modelFile || isLoading}
                    className="flex-1 bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
                  >
                    {isLoading && loadProgress < 100 ? (
                      <Loader2 className="w-5 h-5 animate-spin" />
                    ) : (
                      <Play className="w-5 h-5" />
                    )}
                    Load Model
                  </button>
                </div>
                {modelFile && (
                  <p className="text-green-300 text-sm mt-2">
                    Selected: {modelFile.length} file(s)
                  </p>
                )}
              </div>

              <div className="mb-4">
                <label className="block text-purple-200 text-sm mb-2">
                  Context Size: {nCtx}
                </label>
                <input
                  type="range"
                  min="128"
                  max="8192"
                  step="128"
                  value={nCtx}
                  onChange={(e) => setNCtx(Number(e.target.value))}
                  className="w-full"
                />
              </div>

              {loadProgress > 0 && loadProgress < 100 && (
                <div className="mb-4">
                  <div className="bg-white/20 rounded-full h-3 overflow-hidden">
                    <div
                      className="bg-gradient-to-r from-purple-500 to-pink-500 h-full transition-all duration-300"
                      style={{ width: `${loadProgress}%` }}
                    />
                  </div>
                  <p className="text-white text-sm mt-2 text-center">
                    {loadProgress}%
                  </p>
                </div>
              )}
            </div>
          </div>
        )}

        {/* Status Messages */}
        {status && (
          <div className="bg-blue-500/20 border-b border-blue-400/50 text-blue-100 px-6 py-3">
            {status}
          </div>
        )}

        {error && (
          <div className="bg-red-500/20 border-b border-red-400/50 text-red-100 px-6 py-3">
            {error}
          </div>
        )}

        {/* Chat Messages Area */}
        <div className="flex-1 overflow-y-auto p-6 space-y-4">
          {messages.length === 0 && wllamaRef.current && (
            <div className="text-center text-purple-300 py-12">
              <Bot className="w-16 h-16 mx-auto mb-4 opacity-50" />
              <p className="text-lg">Start a conversation!</p>
              <p className="text-sm mt-2">Type a message below to begin chatting.</p>
            </div>
          )}

          {messages.map((message, index) => (
            <div
              key={index}
              className={`flex gap-3 ${
                message.role === 'user' ? 'justify-end' : 'justify-start'
              }`}
            >
              {message.role === 'assistant' && (
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center">
                  <Bot className="w-5 h-5 text-white" />
                </div>
              )}
              
              <div
                className={`max-w-[70%] rounded-2xl p-4 ${
                  message.role === 'user'
                    ? 'bg-purple-600 text-white'
                    : 'bg-white/10 text-white border border-white/20'
                }`}
              >
                <p className="whitespace-pre-wrap break-words">{message.content}</p>
                <p className="text-xs mt-2 opacity-60">
                  {message.timestamp.toLocaleTimeString()}
                </p>
              </div>

              {message.role === 'user' && (
                <div className="flex-shrink-0 w-8 h-8 rounded-full bg-pink-600 flex items-center justify-center">
                  <User className="w-5 h-5 text-white" />
                </div>
              )}
            </div>
          ))}

          {isGenerating && (
            <div className="flex gap-3 justify-start">
              <div className="flex-shrink-0 w-8 h-8 rounded-full bg-purple-600 flex items-center justify-center">
                <Bot className="w-5 h-5 text-white" />
              </div>
              <div className="bg-white/10 text-white border border-white/20 rounded-2xl p-4">
                <Loader2 className="w-5 h-5 animate-spin" />
              </div>
            </div>
          )}

          <div ref={messagesEndRef} />
        </div>

        {/* Input Area */}
        <div className="bg-white/10 backdrop-blur-lg border-t border-white/20 p-4">
          <div className="max-w-4xl mx-auto">
            <div className="flex gap-3 items-end">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyDown={handleKeyPress}
                placeholder={wllamaRef.current ? "Type your message... (Shift+Enter for new line)" : "Load a model first..."}
                disabled={!wllamaRef.current || isGenerating}
                className="flex-1 bg-white/10 border border-white/20 text-white placeholder-gray-400 rounded-lg p-3 min-h-[60px] max-h-[200px] focus:outline-none focus:ring-2 focus:ring-purple-500 resize-none disabled:opacity-50"
                rows={2}
              />
              <button
                onClick={sendMessage}
                disabled={!wllamaRef.current || isGenerating || !input.trim()}
                className="bg-gradient-to-r from-purple-600 to-pink-600 hover:from-purple-700 hover:to-pink-700 disabled:from-gray-600 disabled:to-gray-600 disabled:cursor-not-allowed text-white font-bold p-4 rounded-lg transition-all flex items-center justify-center shadow-lg"
              >
                <Send className="w-5 h-5" />
              </button>
              {messages.length > 0 && (
                <button
                  onClick={exportChat}
                  disabled={isGenerating}
                  className="bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-bold p-4 rounded-lg transition-all flex items-center justify-center"
                  title="Export chat"
                >
                  <Download className="w-5 h-5" />
                </button>
              )}
            </div>
            <div className="flex items-center justify-between mt-2">
              <p className="text-xs text-purple-300">
                Press Enter to send • Shift+Enter for new line
              </p>
              {conversations.length > 0 && (
                <p className="text-xs text-green-300 flex items-center gap-1">
                  <Save className="w-3 h-3" />
                  Auto-saved
                </p>
              )}
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}