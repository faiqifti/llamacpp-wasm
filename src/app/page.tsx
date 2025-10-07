'use client';

import { useState, useRef, useEffect, useMemo } from 'react';
import { 
  Upload, 
  Play, 
  Loader2, 
  Send, 
  Trash2, 
  User, 
  Download, 
  Save, 
  Plus, 
  MessageSquare, 
  Menu, 
  X, 
  Globe, 
  HardDrive, 
  FileText,
  MessageCircle,
  Cpu,
  Brain
} from 'lucide-react';

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
  loadModelFromUrl: (url: string, config: WllamaConfig) => Promise<void>;
  loadModel: (blobs: File[], config: WllamaConfig) => Promise<void>;
  createCompletion: (prompt: string, options: any) => Promise<string>;
}

interface Message {
  role: 'user' | 'assistant';
  content: string;
  timestamp: Date;
  attachments?: Attachment[];
  documentSources?: string[]; // Add this line
}

interface Attachment {
  id: string;
  name: string;
  type: string;
  size: number;
  content: string; // Extracted text content
}

interface Conversation {
  id: string;
  title: string;
  messages: Message[];
  createdAt: Date;
  updatedAt: Date;
}

interface CachedModel {
  url: string;
  size: number;
  name: string;
}

// Add these new interfaces at the top
interface DocumentChunk {
  id: string;
  documentId: string;
  content: string;
  embedding: number[];
  metadata: {
    chunkIndex: number;
    startPos: number;
    endPos: number;
  };
}

interface StoredDocument {
  id: string;
  name: string;
  type: string;
  size: number;
  content: string;
  chunks: DocumentChunk[];
  processedAt: Date;
}

// Add embedding model interface
interface EmbeddingModel {
  generateEmbedding: (text: string) => Promise<number[]>;
  isLoaded: boolean;
}

class EmbeddingProcessor {
  private wllama: any = null;
  private isInitialized = false;
  private initPromise: Promise<void> | null = null;

  private generateSmartMockEmbedding(text: string): number[] {
    const embedding = new Array(384).fill(0);
    const words = text.toLowerCase().split(/\s+/);
    const sentences = text.split(/[.!?]+/);
    
    // Common question words and important terms
    const questionWords = ['what', 'how', 'why', 'when', 'where', 'who', 'which', 'explain', 'describe', 'show'];
    const importantVerbs = ['analyze', 'compare', 'calculate', 'find', 'list', 'identify', 'summarize'];
    
    words.forEach(word => {
      const cleanWord = word.replace(/[^a-z0-9]/g, '');
      if (cleanWord.length < 2) return;
      
      let hash = 0;
      for (let i = 0; i < cleanWord.length; i++) {
        hash = ((hash << 5) - hash) + cleanWord.charCodeAt(i);
        hash = hash & hash;
      }
      
      // Boost important words
      let weight = 0.15;
      if (questionWords.includes(cleanWord)) weight = 0.4;
      if (importantVerbs.includes(cleanWord)) weight = 0.35;
      if (cleanWord.length > 6) weight = 0.25; // Longer words often more specific
      
      // Distribute across embedding
      for (let i = 0; i < 4; i++) {
        const index = Math.abs(hash + i * 7919) % 384;
        embedding[index] = (embedding[index] + weight) % 1.0;
      }
    });
    
    // Document characteristics
    embedding[0] = Math.min(words.length / 150, 1.0);
    embedding[1] = text.includes('?') ? 0.9 : 0.1;
    embedding[2] = Math.min(sentences.length / 15, 1.0);
    
    return embedding;
  }

  async init(): Promise<void> {
    if (this.isInitialized) return;
    if (this.initPromise) return this.initPromise;

    this.initPromise = (async () => {
      try {
        const WllamaModule = await import('@wllama/wllama/esm/index.js');
        const Wllama = WllamaModule.Wllama;

        const CONFIG_PATHS = {
          'single-thread/wllama.wasm': './wllama/esm/single-thread/wllama.wasm',
          'multi-thread/wllama.wasm': './wllama/esm/multi-thread/wllama.wasm',
        };

        this.wllama = new Wllama(CONFIG_PATHS);
        
        const config = {
          n_ctx: 2048,
          n_batch: 512,
          n_threads: navigator.hardwareConcurrency || 4,
          n_gpu_layers: 0,
          use_mlock: false,
          use_mmap: true,
          progressCallback: () => {}
        };

        // const embeddingModelUrl = `https://huggingface.co/ggml-org/embeddinggemma-300M-GGUF/resolve/main/embeddinggemma-300m-q4_0.gguf`;
        const embeddingModelUrl = `https://huggingface.co/ggml-org/gte-small-Q8_0-GGUF/resolve/main/gte-small-q8_0.gguf`;
        
        await this.wllama.loadModelFromUrl(embeddingModelUrl, config);
        
        // Enable embeddings for the main instance
        await this.wllama.setOptions({ embeddings: true });

        this.isInitialized = true;
        
        console.log('Embedding model loaded successfully');
      } catch (error) {
        console.error('Failed to load embedding model:', error);
        this.initPromise = null;
        throw error;
      }
    })();

    return this.initPromise;
  }

  async generateEmbedding(text: string): Promise<number[]> {
    if (!this.isInitialized) {
      await this.init();
    }

    try {
      // Note: You might need to adjust this based on the actual Wllama API
      // Some embedding models might use different methods
      const embedding = await this.wllama.createEmbedding(text);
      return embedding;
    } catch (error) {
      console.error('Embedding generation failed:', error);
      // Fallback to mock embedding
      return this.generateMockEmbedding(text);
    }
  }

  private generateMockEmbedding(text: string): number[] {
    const embedding = new Array(512).fill(0);
    const words = text.toLowerCase().split(/\s+/);
    
    words.forEach(word => {
      let hash = 0;
      for (let i = 0; i < word.length; i++) {
        hash = ((hash << 5) - hash) + word.charCodeAt(i);
        hash |= 0;
      }
      const index = Math.abs(hash) % 512;
      embedding[index] = (embedding[index] + 1) % 1.0;
    });
    
    return embedding;
  }
}

// Add these imports at the top
import { pipeline, env } from '@xenova/transformers';

// Mock embedding function (in production, use a proper embedding model)
const generateEmbedding = async (text: string): Promise<number[]> => {
  // Simple mock embedding - replace with real embedding model
  const embedding = new Array(384).fill(0);
  for (let i = 0; i < Math.min(text.length, 384); i++) {
    embedding[i] = text.charCodeAt(i) / 65535;
  }
  return embedding;
};

const cosineSimilarity = (a: number[], b: number[]): number => {
  const dotProduct = a.reduce((sum, val, i) => sum + val * b[i], 0);
  const magnitudeA = Math.sqrt(a.reduce((sum, val) => sum + val * val, 0));
  const magnitudeB = Math.sqrt(b.reduce((sum, val) => sum + val * val, 0));
  return dotProduct / (magnitudeA * magnitudeB);
};

// Update your DocumentProcessor class to use the real embedding model
// Updated DocumentProcessor class with proper async/await handling
class DocumentProcessor {
  private db: IDBDatabase | null = null;
  private readonly DB_NAME = 'WllamaDocuments';
  private readonly DB_VERSION = 1;
  private readonly STORE_NAME = 'documents';
  private embeddingModel: EmbeddingProcessor;
  private initPromise: Promise<void> | null = null;

  constructor() {
    this.embeddingModel = new EmbeddingProcessor();
  }

  async init(): Promise<void> {
    // Prevent multiple initializations
    if (this.initPromise) {
      return this.initPromise;
    }

    this.initPromise = new Promise((resolve, reject) => {
      const request = indexedDB.open(this.DB_NAME, this.DB_VERSION);

      request.onerror = () => {
        this.initPromise = null;
        reject(request.error);
      };
      
      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains(this.STORE_NAME)) {
          const store = db.createObjectStore(this.STORE_NAME, { keyPath: 'id' });
          store.createIndex('name', 'name', { unique: false });
          store.createIndex('processedAt', 'processedAt', { unique: false });
        }
      };

      request.onblocked = () => {
        this.initPromise = null;
        reject(new Error('IndexedDB request blocked'));
      };
    });

    return this.initPromise;
  }

  async ensureInitialized(): Promise<void> {
    if (!this.db) {
      await this.init();
    }
  }

  async chunkDocument(content: string, chunkSize: number = 500, overlap: number = 50): Promise<string[]> {
    const chunks: string[] = [];
    let start = 0;

    while (start < content.length) {
      let end = start + chunkSize;
      
      // Try to break at sentence end
      const sentenceEnd = content.slice(start, end).lastIndexOf('.');
      if (sentenceEnd > chunkSize * 0.5) {
        end = start + sentenceEnd + 1;
      }

      // Try to break at paragraph end
      const paragraphEnd = content.slice(start, end).lastIndexOf('\n\n');
      if (paragraphEnd > chunkSize * 0.5) {
        end = start + paragraphEnd + 2;
      }

      const chunk = content.slice(start, end).trim();
      if (chunk.length > 0) {
        chunks.push(chunk);
      }

      start = end - overlap;
      if (start >= content.length) break;
    }

    return chunks;
  }

  async processDocument(file: File, content: string): Promise<StoredDocument> {
    await this.ensureInitialized();

    const chunks = await this.chunkDocument(content);
    const documentChunks: DocumentChunk[] = [];

    for (let i = 0; i < chunks.length; i++) {
      try {
        const embedding = await this.embeddingModel.generateEmbedding(chunks[i]);
        documentChunks.push({
          id: `${file.name}-chunk-${i}-${Date.now()}`,
          documentId: file.name,
          content: chunks[i],
          embedding,
          metadata: {
            chunkIndex: i,
            startPos: i * 500,
            endPos: (i * 500) + chunks[i].length
          }
        });
      } catch (error) {
        console.error(`Failed to generate embedding for chunk ${i}:`, error);
        // Continue with other chunks even if one fails
      }
    }

    const document: StoredDocument = {
      id: `${file.name}-${Date.now()}`,
      name: file.name,
      type: file.type,
      size: file.size,
      content: content.substring(0, 1000),
      chunks: documentChunks,
      processedAt: new Date()
    };

    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }

      const transaction = this.db.transaction([this.STORE_NAME], 'readwrite');
      const store = transaction.objectStore(this.STORE_NAME);
      const request = store.put(document);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(document);
    });
  }

  async searchDocuments(query: string, limit: number = 3): Promise<DocumentChunk[]> {
    await this.ensureInitialized();

    const queryEmbedding = await this.embeddingModel.generateEmbedding(query);
    const allChunks: DocumentChunk[] = [];

    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }

      const transaction = this.db.transaction([this.STORE_NAME], 'readonly');
      const store = transaction.objectStore(this.STORE_NAME);
      const request = store.getAll();

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        const documents: StoredDocument[] = request.result;
        
        documents.forEach(doc => {
          allChunks.push(...doc.chunks);
        });

        // Calculate similarity and filter by threshold
        const scoredChunks = allChunks.map(chunk => ({
          chunk,
          score: cosineSimilarity(queryEmbedding, chunk.embedding)
        }));

        // Only return chunks that are actually relevant
        const relevantChunks = scoredChunks
          .filter(item => item.score > 0.3) // Minimum similarity threshold
          .sort((a, b) => b.score - a.score)
          .slice(0, limit)
          .map(item => item.chunk);

        console.log(`Found ${relevantChunks.length} relevant chunks for: "${query}"`);
        resolve(relevantChunks);
      };
    });
  }

  async getAllDocuments(): Promise<StoredDocument[]> {
    await this.ensureInitialized();

    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }

      const transaction = this.db.transaction([this.STORE_NAME], 'readonly');
      const store = transaction.objectStore(this.STORE_NAME);
      const request = store.getAll();

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve(request.result);
    });
  }

  async deleteDocument(documentId: string): Promise<void> {
    await this.ensureInitialized();

    return new Promise((resolve, reject) => {
      if (!this.db) {
        reject(new Error('Database not initialized'));
        return;
      }

      const transaction = this.db.transaction([this.STORE_NAME], 'readwrite');
      const store = transaction.objectStore(this.STORE_NAME);
      const request = store.delete(documentId);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve();
    });
  }
}

class DiskModelManager {
  private db: IDBDatabase | null = null;
  private readonly DB_NAME = 'WllamaModels';
  private readonly STORE_NAME = 'models';

  async init(): Promise<void> {
    return new Promise((resolve, reject) => {
      const request = indexedDB.open(this.DB_NAME, 1);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        this.db = request.result;
        resolve();
      };

      request.onupgradeneeded = (event) => {
        const db = (event.target as IDBOpenDBRequest).result;
        if (!db.objectStoreNames.contains(this.STORE_NAME)) {
          const store = db.createObjectStore(this.STORE_NAME, { keyPath: 'id' });
          store.createIndex('name', 'name', { unique: false });
          store.createIndex('size', 'size', { unique: false });
          store.createIndex('lastAccessed', 'lastAccessed', { unique: false });
        }
      };
    });
  }

  async saveModelChunk(modelId: string, chunkIndex: number, data: ArrayBuffer): Promise<void> {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([this.STORE_NAME], 'readwrite');
      const store = transaction.objectStore(this.STORE_NAME);
      
      const chunk = {
        id: `${modelId}_chunk_${chunkIndex}`,
        modelId,
        chunkIndex,
        data,
        timestamp: new Date()
      };

      const request = store.put(chunk);
      request.onerror = () => reject(request.error);
      request.onsuccess = () => resolve();
    });
  }

  async getModelChunk(modelId: string, chunkIndex: number): Promise<ArrayBuffer | null> {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([this.STORE_NAME], 'readonly');
      const store = transaction.objectStore(this.STORE_NAME);
      const request = store.get(`${modelId}_chunk_${chunkIndex}`);

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        resolve(request.result?.data || null);
      };
    });
  }

  async getCachedModels(): Promise<any[]> {
    if (!this.db) await this.init();

    return new Promise((resolve, reject) => {
      const transaction = this.db!.transaction([this.STORE_NAME], 'readonly');
      const store = transaction.objectStore(this.STORE_NAME);
      const request = store.getAll();

      request.onerror = () => reject(request.error);
      request.onsuccess = () => {
        // Group chunks by modelId
        const chunks = request.result;
        const models: any = {};
        
        chunks.forEach(chunk => {
          if (!models[chunk.modelId]) {
            models[chunk.modelId] = {
              id: chunk.modelId,
              chunks: [],
              size: 0
            };
          }
          models[chunk.modelId].chunks.push(chunk);
          models[chunk.modelId].size += chunk.data.byteLength;
        });

        resolve(Object.values(models));
      };
    });
  }
}

export default function WllamaUI() {
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
  const [showModelManager, setShowModelManager] = useState(false);
  const [modelUrl, setModelUrl] = useState('');
  const [cachedModels, setCachedModels] = useState<CachedModel[]>([]);
  const [loadMethod, setLoadMethod] = useState<'url' | 'file'>('url');
  const [modelFile, setModelFile] = useState<FileList | null>(null);
  const [availableFiles, setAvailableFiles] = useState<string[]>([]);
  const [selectedFile, setSelectedFile] = useState<string>('');
  const [fetchingFiles, setFetchingFiles] = useState(false);
  const [chatTemplate, setChatTemplate] = useState<'gemma' | 'qwen' | 'llama' | 'chatml'>('gemma');
  const [attachedFiles, setAttachedFiles] = useState<File[]>([]);
  const [isProcessingFiles, setIsProcessingFiles] = useState(false);
  const wllamaRef = useRef<Wllama | null>(null);
  const fileInputRef = useRef<HTMLInputElement>(null);
  const documentInputRef = useRef<HTMLInputElement>(null);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const inputRef = useRef<HTMLTextAreaElement>(null);

  const [documentProcessor] = useState(new DocumentProcessor());
  const [processedDocuments, setProcessedDocuments] = useState<StoredDocument[]>([]);

  // Initialize document processor
  // In your WllamaUI component, replace the problematic useEffect with:
  useEffect(() => {
    const initializeApp = async () => {
      try {
        // Initialize document processor
        await documentProcessor.init();
        const docs = await documentProcessor.getAllDocuments();
        setProcessedDocuments(docs);
        
        // Initialize embedding model separately
        setEmbeddingModelStatus('Loading embedding model...');
        try {
          // The embedding model will be initialized when first used
          setEmbeddingModelStatus('Embedding model ready on demand');
        } catch (error) {
          console.error('Failed to initialize embedding model:', error);
          setEmbeddingModelStatus('Embedding model will use fallback');
        }
      } catch (error) {
        console.error('Failed to initialize document processor:', error);
      }
    };

    initializeApp();
  }, [documentProcessor]);

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
    loadCachedModels();
  }, []);

  useEffect(() => {
    if (conversations.length > 0) {
      localStorage.setItem('wllama-conversations', JSON.stringify(conversations));
    }
  }, [conversations]);

  useEffect(() => {
    if (currentConversationId) {
      localStorage.setItem('wllama-last-conversation-id', currentConversationId);
    }
  }, [currentConversationId]);

  const currentConversation = conversations.find(c => c.id === currentConversationId);
  const messages = useMemo(() => currentConversation?.messages || [], [currentConversation]);

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };
  useEffect(() => {
    scrollToBottom();
  }, [messages]);

  const loadCachedModels = async () => {
    try {
      const WllamaModule = await import('@wllama/wllama/esm/index.js');
      const { ModelManager } = WllamaModule;
      const manager = new ModelManager();
      const models = await manager.getModels();
      setCachedModels(models.map((m: any) => ({
        url: m.url,
        size: m.size,
        name: m.url.split('/').pop()?.replace('.gguf', '') || 'Unknown'
      })));
    } catch (err) {
      console.error('Failed to load cached models:', err);
    }
  };

  const fetchRepoFiles = async (repoId: string) => {
    setFetchingFiles(true);
    setAvailableFiles([]);
    setSelectedFile('');
    setError('');

    try {
      // Fetch file list from Hugging Face API
      const response = await fetch(`https://huggingface.co/api/models/${repoId}/tree/main`);
      if (!response.ok) {
        throw new Error('Repository not found or inaccessible');
      }

      const data = await response.json();
      const ggufFiles = data
        .filter((item: any) => item.path.endsWith('.gguf'))
        .map((item: any) => item.path);

      if (ggufFiles.length === 0) {
        setError('No .gguf files found in this repository');
      } else {
        setAvailableFiles(ggufFiles);
        if (ggufFiles.length === 1) {
          setSelectedFile(ggufFiles[0]);
        }
      }
    } catch (err: any) {
      setError('Failed to fetch repository: ' + (err?.message || String(err)));
    } finally {
      setFetchingFiles(false);
    }
  };

  const handleRepoInputChange = (value: string) => {
    setModelUrl(value);

    // Auto-fetch files if input looks like a repo ID
    const repoPattern = /^[a-zA-Z0-9_-]+\/[a-zA-Z0-9_.-]+$/;
    if (repoPattern.test(value)) {
      fetchRepoFiles(value);
    } else {
      setAvailableFiles([]);
      setSelectedFile('');
    }
  };

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
    setAttachedFiles([]);
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
      setAttachedFiles([]);
    }
  };

  const handleFileSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const hasGguf = Array.from(files).some(f => f.name.endsWith('.gguf'));
      if (hasGguf) {
        setModelFile(files);
        setError('');
      } else {
        setError('Please select at least one valid .gguf model file');
        setModelFile(null);
      }
    }
  };

  // Handle document file selection (PDF, CSV, DOC, DOCX)
  const handleDocumentSelect = (e: React.ChangeEvent<HTMLInputElement>) => {
    const files = e.target.files;
    if (files && files.length > 0) {
      const validFiles = Array.from(files).filter(file => {
        const fileType = file.type.toLowerCase();
        const fileName = file.name.toLowerCase();
        return (
          fileType.includes('pdf') ||
          fileType.includes('csv') ||
          fileType.includes('text/csv') ||
          fileName.endsWith('.csv') ||
          fileType.includes('msword') ||
          fileType.includes('wordprocessingml') ||
          fileName.endsWith('.doc') ||
          fileName.endsWith('.docx') ||
          fileType.includes('text/plain') ||
          fileName.endsWith('.txt')
        );
      });

      if (validFiles.length > 0) {
        setAttachedFiles(prev => [...prev, ...validFiles]);
        setError('');
      } else {
        setError('Please select valid PDF, CSV, DOC, DOCX, or TXT files');
      }
    }
  };

  // Remove attached file
  const removeAttachedFile = (index: number) => {
    setAttachedFiles(prev => prev.filter((_, i) => i !== index));
  };

  // Extract text from different file types
  const extractTextFromFile = async (file: File): Promise<string> => {
    return new Promise((resolve, reject) => {
      const fileType = file.type.toLowerCase();
      const fileName = file.name.toLowerCase();

      // For CSV and text files
      if (fileType.includes('csv') || fileType.includes('text/plain') || fileName.endsWith('.txt')) {
        const reader = new FileReader();
        reader.onload = (e) => {
          const content = e.target?.result as string || '';
          // Clean up CSV content - remove extra spaces and normalize
          const cleanedContent = content.split('\n')
            .map(line => line.trim())
            .filter(line => line.length > 0)
            .join('\n');
          resolve(cleanedContent);
        };
        reader.onerror = () => reject(new Error('Failed to read file'));
        reader.readAsText(file);
      }
      // For PDF files - improved extraction
      else if (fileType.includes('pdf') || fileName.endsWith('.pdf')) {
        const reader = new FileReader();
        reader.onload = async (e) => {
          try {
            // Basic PDF text extraction
            const arrayBuffer = e.target?.result as ArrayBuffer;
            if (arrayBuffer) {
              // Convert to text using basic extraction
              const uint8Array = new Uint8Array(arrayBuffer);
              let text = '';
              
              // Simple PDF text extraction (basic approach)
              // This extracts text between BT and ET operators (text objects in PDF)
              const decoder = new TextDecoder('iso-8859-1');
              const pdfText = decoder.decode(uint8Array);
              
              // Extract text between text operators
              const textMatches = pdfText.match(/BT[\s\S]*?ET/g) || [];
              if (textMatches.length > 0) {
                text = textMatches.map(match => {
                  // Extract text content between BT and ET
                  return match.replace(/BT|ET/g, '')
                    .replace(/Td|Tj|TJ|Tm|Tf/g, ' ')
                    .replace(/[\(\)]/g, '')
                    .replace(/\s+/g, ' ')
                    .trim();
                }).join(' ');
              } else {
                // Fallback: try to extract any readable text
                text = pdfText.replace(/[^\x20-\x7E\n\r]/g, ' ')
                  .replace(/\s+/g, ' ')
                  .trim();
              }
              
              resolve(text || `PDF file: ${file.name} - Text extraction limited. For better results, consider converting to text first.`);
            } else {
              resolve(`PDF file: ${file.name} - Could not read file content`);
            }
          } catch (err) {
            console.error('PDF extraction error:', err);
            resolve(`PDF file: ${file.name} - Error extracting text. Consider using a text-based file format.`);
          }
        };
        reader.onerror = () => reject(new Error('Failed to read PDF file'));
        reader.readAsArrayBuffer(file);
      }
      // For DOC/DOCX files
      else if (fileType.includes('msword') || fileType.includes('wordprocessingml') || 
              fileName.endsWith('.doc') || fileName.endsWith('.docx')) {
        resolve(`Word Document: ${file.name}\n\nDOC/DOCX text extraction requires additional libraries. For best results, please save as PDF or TXT format and upload again.`);
      }
      else {
        resolve(`Unsupported file type: ${file.name}. Please use PDF, CSV, TXT, DOC, or DOCX files.`);
      }
    });
  };

  // Process all attached files and extract text
  const processAttachedFiles = async (): Promise<Attachment[]> => {
    if (attachedFiles.length === 0) return [];
  
    setIsProcessingFiles(true);
    setStatus('Processing and indexing documents...');
  
    try {
      const attachments: Attachment[] = [];
      const newlyProcessedDocs: StoredDocument[] = []; // ← Store docs here
  
      for (const file of attachedFiles) {
        try {
          const content = await extractTextFromFile(file);
          const processedDoc = await documentProcessor.processDocument(file, content);
          newlyProcessedDocs.push(processedDoc); // ← Collect docs
  
          attachments.push({
            id: processedDoc.id,
            name: file.name,
            type: file.type,
            size: file.size,
            content: `Document processed and indexed. ${processedDoc.chunks.length} chunks created.`
          });
  
        } catch (err) {
          console.error(`Failed to process file ${file.name}:`, err);
          attachments.push({
            id: Date.now().toString() + Math.random(),
            name: file.name,
            type: file.type,
            size: file.size,
            content: `[Error processing file: ${file.name}]`
          });
        }
      }
  
      // Update state once with all new documents
      setProcessedDocuments(prev => [...prev, ...newlyProcessedDocs]);
      setStatus('');
      return attachments;
      
    } catch (err) {
      setError('Failed to process attached files');
      return [];
    } finally {
      setIsProcessingFiles(false);
    }
  };
  
  const [embeddingModelStatus, setEmbeddingModelStatus] = useState<string>('');

  // Initialize embedding model on component mount
  useEffect(() => {
    const initEmbeddingModel = async () => {
      setEmbeddingModelStatus('Loading embedding model...');
      try {
        await documentProcessor.init();
        setEmbeddingModelStatus('Embedding model ready');
      } catch (error) {
        console.error('Failed to initialize embedding model:', error);
        setEmbeddingModelStatus('Embedding model failed - using fallback');
      }
    };

    initEmbeddingModel();
  }, []);

  // Update the loadModelFromUrl function with better memory management
  const loadModelFromUrl = async (url: string) => {
    setIsLoading(true);
    setError('');
    setLoadProgress(0);
    setStatus('Initializing Wllama...');
  
    try {
      const WllamaModule = await import('@wllama/wllama/esm/index.js');
      const Wllama = WllamaModule.Wllama;
      const { ModelManager } = WllamaModule;
  
      const modelManager = new ModelManager();
      const cachedModels = await modelManager.getModels();
      const existingModel = cachedModels.find(m => m.url === url);
  
      if (existingModel) {
        // Load from disk cache
        setStatus('Loading model from disk cache...');
        wllamaRef.current = new Wllama({
          'single-thread/wllama.wasm': './wllama/esm/single-thread/wllama.wasm',
          'multi-thread/wllama.wasm': './wllama/esm/multi-thread/wllama.wasm',
        });
      
        const config = {
          n_ctx: nCtx,
          n_batch: 256,
          n_threads: navigator.hardwareConcurrency || 4,
          n_gpu_layers: 0,
          use_mlock: false,
          use_mmap: true,
          progressCallback: ({ loaded, total }: ProgressCallback) => {
            const progress = Math.round((loaded / total) * 100);
            setLoadProgress(progress);
            setStatus(`Loading from disk... ${progress}%`);
          },
        };
      
        // Use the cached model's URL to load it
        await wllamaRef.current.loadModelFromUrl(existingModel.url, config);

      } else {
        // Download new model (your original loading logic)
        let fullUrl = url;
        
        if (!url.startsWith('http')) {
          if (availableFiles.length > 0 && !selectedFile) {
            throw new Error('Please select a file from the repository');
          }
          
          if (selectedFile) {
            fullUrl = `https://huggingface.co/${modelUrl}/resolve/main/${selectedFile}`;
          } else {
            if (availableFiles.length > 0) {
              fullUrl = `https://huggingface.co/${modelUrl}/resolve/main/${availableFiles[0]}`;
              setSelectedFile(availableFiles[0]);
            } else {
              throw new Error('No GGUF files found in this repository or repository not found');
            }
          }
        }
  
        wllamaRef.current = new Wllama({
          'single-thread/wllama.wasm': './wllama/esm/single-thread/wllama.wasm',
          'multi-thread/wllama.wasm': './wllama/esm/multi-thread/wllama.wasm',
        });
  
        const progressCallback = ({ loaded, total }: ProgressCallback) => {
          const progressPercentage = Math.round((loaded / total) * 100);
          setLoadProgress(progressPercentage);
          setStatus(`Downloading model... ${progressPercentage}%`);
        };
  
        setStatus('Downloading and caching model...');
  
        const config = {
          n_ctx: nCtx,
          n_batch: 256,
          n_threads: Math.max(2, navigator.hardwareConcurrency - 1),
          n_gpu_layers: 0,
          use_mlock: false,
          use_mmap: true,
          progressCallback,
        };
  
        await wllamaRef.current.loadModelFromUrl(fullUrl, config);
      }
  
      setStatus('Model ready!');
      setShowModelManager(false);
  
      if (conversations.length === 0) {
        createNewConversation();
      }
  
      await loadCachedModels();
    } catch (err: any) {
      setError('Failed to load model: ' + (err?.message || String(err)));
      setStatus('');
      console.error(err);
    } finally {
      setIsLoading(false);
    }
  };

  const loadModelFromFile = async () => {
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
      setShowModelManager(false);

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

  const deleteCachedModel = async (url: string) => {
    try {
      const WllamaModule = await import('@wllama/wllama/esm/index.js');
      const { ModelManager } = WllamaModule;
      const manager = new ModelManager();
      const models = await manager.getModels();
      const model = models.find((m: any) => m.url === url);
      if (model) {
        await model.remove();
        await loadCachedModels();
        setStatus('Model deleted successfully');
      }
    } catch (err: any) {
      setError('Failed to delete model: ' + (err?.message || String(err)));
    }
  };

  // const buildConversationPrompt = (messages: Message[], newUserMessage: string, attachments: Attachment[] = []) => {
  // Add memory optimization to the buildConversationPrompt function
  // const buildConversationPrompt = async (messages: Message[], newUserMessage: string, attachments: Attachment[] = []) => {
  //   let prompt = '';

  //   // Perform semantic search if we have processed documents
  //   if (processedDocuments.length > 0 && newUserMessage.trim()) {
  //     try {
  //       const relevantChunks = await documentProcessor.searchDocuments(newUserMessage, 3);
        
  //       if (relevantChunks.length > 0) {
  //         prompt += "Based on the documents you have access to, here are the most relevant sections:\n\n";
          
  //         relevantChunks.forEach((chunk, index) => {
  //           prompt += `[Document Section ${index + 1} from "${chunk.documentId}"]:\n`;
  //           prompt += chunk.content + '\n\n';
  //         });
          
  //         prompt += "Using the above relevant document sections, please answer the following question:\n\n";
  //       }
  //     } catch (err) {
  //       console.error('Semantic search failed:', err);
  //     }
  //   }

  //   // Add file content for newly attached files
  //   if (attachments.length > 0) {
  //     prompt += "Newly attached files:\n\n";
  //     attachments.forEach(attachment => {
  //       prompt += `--- FILE: ${attachment.name} ---\n`;
  //       const contentPreview = attachment.content.length > 1000 
  //         ? attachment.content.substring(0, 1000) + '... [truncated]'
  //         : attachment.content;
  //       prompt += contentPreview + '\n\n';
  //     });
  //   }

  //    // Rest of your existing prompt building logic...
  //   const recentMessages = messages.slice(-4);
  
  //   // Build conversation history
  //   switch (chatTemplate) {
  //     case 'gemma':
  //       // ✅ Start with system instruction
  //       prompt += `<start_of_turn>system
  //       You are a helpful, respectful, and honest assistant. Always answer as helpfully as possible, while being safe and factual.<end_of_turn>\n`;
      
  //       messages.forEach(msg => {
  //         if (msg.role === 'user') {
  //           prompt += `<start_of_turn>user\n${msg.content}<end_of_turn>\n`;
  //         } else {
  //           prompt += `<start_of_turn>model\n${msg.content}<end_of_turn>\n`;
  //         }
  //       });
  //       prompt += `<start_of_turn>user\n${newUserMessage}<end_of_turn>\n<start_of_turn>model\n`;
  //       break;
  
  //     case 'qwen':
  //       messages.forEach(msg => {
  //         if (msg.role === 'user') {
  //           prompt += `<|im_start|>user\n${msg.content}<|im_end|>\n`;
  //         } else {
  //           prompt += `<|im_start|>assistant\n${msg.content}<|im_end|>\n`;
  //         }
  //       });
  //       prompt += `<|im_start|>user\n${newUserMessage}<|im_end|>\n<|im_start|>assistant\n`;
  //       break;
  
  //     case 'llama':
  //       messages.forEach(msg => {
  //         if (msg.role === 'user') {
  //           prompt += `[INST] ${msg.content} [/INST]\n`;
  //         } else {
  //           prompt += `${msg.content}\n`;
  //         }
  //       });
  //       prompt += `[INST] ${newUserMessage} [/INST]\n`;
  //       break;
  
  //     case 'chatml':
  //       messages.forEach(msg => {
  //         prompt += `<|im_start|>${msg.role}\n${msg.content}<|im_end|>\n`;
  //       });
  //       prompt += `<|im_start|>user\n${newUserMessage}<|im_end|>\n<|im_start|>assistant\n`;
  //       break;
  //   }
  
  //   return prompt;
  // };

  const buildConversationPrompt = async (messages: Message[], newUserMessage: string, attachments: Attachment[] = []) => {
    let prompt = '';
  
    // STEP 1: Always search documents first
    if (processedDocuments.length > 0) {
      try {
        const relevantChunks = await documentProcessor.searchDocuments(newUserMessage, 3);
        
        if (relevantChunks.length > 0) {
          prompt += "You have access to the following documents. Use this information to answer the question. If the answer can be found in these documents, prioritize this information over general knowledge.\n\n";
          prompt += "REFERENCE DOCUMENTS:\n";
          prompt += "====================\n";
          
          relevantChunks.forEach((chunk, index) => {
            prompt += `[DOCUMENT ${index + 1} - ${chunk.documentId}]:\n`;
            prompt += `${chunk.content}\n\n`;
          });
          
          prompt += "QUESTION: " + newUserMessage + "\n\n";
          prompt += "INSTRUCTIONS:\n";
          prompt += "- Answer based on the reference documents above\n";
          prompt += "- If the documents contain the answer, use them as the primary source\n";
          prompt += "- Only use general knowledge if the documents don't contain the answer\n";
          prompt += "- Be precise and cite information from the documents when possible\n";
          prompt += "ANSWER: ";
          
          return prompt; // Return early - documents take priority
        }
      } catch (err) {
        console.error('Document search failed:', err);
      }
    }
  
    // STEP 2: Only use general knowledge if no relevant documents found
    prompt += "Question: " + newUserMessage + "\n\n";
    prompt += "Answer based on your general knowledge: ";
  
    return prompt;
  };

  const sendMessage = async () => {
    if (!wllamaRef.current) {
      setError('Please load a model first');
      return;
    }
  
    if (!input.trim() && attachedFiles.length === 0) {
      setError('Please enter a message or attach files');
      return;
    }
  
    if (!currentConversationId) {
      createNewConversation();
      return;
    }
  
    const userMessage = input.trim() || 
      (attachedFiles.length > 0 
        ? `Please analyze the content of the attached file${attachedFiles.length > 1 ? 's' : ''} and provide a summary or answer questions about it.` 
        : '');
    
    setInput('');
    setIsGenerating(true);
    setError('');
  
    // Process attachments first
    const attachments = await processAttachedFiles();
  
    const newUserMessage: Message = {
      role: 'user',
      content: userMessage,
      timestamp: new Date(),
      attachments: attachments.length > 0 ? attachments : undefined
    };
  
    // Add user message first
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
  
    if (messages.length === 0) {
      updateConversationTitle(currentConversationId, userMessage || `Chat with ${attachments.length} file${attachments.length > 1 ? 's' : ''}`);
    }
  
    // Clear attached files after processing
    setAttachedFiles([]);
  
    // SINGLE TRY BLOCK - removed the nested one
    try {
      // Use the current messages (including the one we just added)
      const currentConv = conversations.find(c => c.id === currentConversationId);
      const messagesForPrompt = currentConv ? [...currentConv.messages, newUserMessage] : [newUserMessage];
      
      // SEARCH DOCUMENTS FIRST - moved here
      let documentSources: string[] = [];
      let formattedPrompt = '';
      
      if (processedDocuments.length > 0) {
        const relevantChunks = await documentProcessor.searchDocuments(userMessage, 3);
        if (relevantChunks.length > 0) {
          // documentSources = [...new Set(relevantChunks.map(chunk => chunk.documentId))];
          documentSources = relevantChunks.map(chunk => chunk.documentId);
        }
      }
      
      formattedPrompt = await buildConversationPrompt(
        messagesForPrompt.slice(0, -1),
        userMessage, 
        attachments
      );
  
      // Get the index for the assistant's message
      const assistantMessageIndex = messages.length + 1;
  
      // Add placeholder for assistant response WITH documentSources
      setConversations(prev => prev.map(conv => {
        if (conv.id === currentConversationId) {
          return {
            ...conv,
            messages: [...conv.messages, {
              role: 'assistant',
              content: '',
              timestamp: new Date(),
              documentSources: documentSources.length > 0 ? documentSources : undefined // Add here
            }],
            updatedAt: new Date()
          };
        }
        return conv;
      }));
  
    } catch (err: any) {
      setError('Failed to generate response: ' + (err?.message || String(err)));
      console.error('Generation error:', err);
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

    const chatText = currentConversation.messages.map(msg => {
      let messageText = `[${msg.timestamp.toLocaleString()}] ${msg.role.toUpperCase()}: ${msg.content}`;
      if (msg.attachments && msg.attachments.length > 0) {
        messageText += `\nAttachments: ${msg.attachments.map(a => a.name).join(', ')}`;
        msg.attachments.forEach(attachment => {
          messageText += `\n--- ${attachment.name} ---\n${attachment.content}\n`;
        });
      }
      return messageText;
    }).join('\n\n');

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

  const [diskUsage, setDiskUsage] = useState<{used: number, quota: number} | null>(null);

  useEffect(() => {
    const checkStorage = async () => {
      if ('storage' in navigator && 'estimate' in navigator.storage) {
        const estimation = await navigator.storage.estimate();
        setDiskUsage({
          used: estimation.usage || 0,
          quota: estimation.quota || 0
        });
      }
    };
    checkStorage();
  }, []);

  return (
    <div className="min-h-screen bg-gradient-to-br from-slate-900 via-purple-900 to-slate-900 flex">
      {/* Sidebar */}
      <div className={`${sidebarOpen ? 'w-64' : 'w-0'} transition-all duration-300 bg-slate-950/50 backdrop-blur-lg border-r border-white/10 flex flex-col overflow-hidden`}>
        <div className="p-4 border-b border-white/10">
          <h2 className="text-white font-bold text-lg mb-3">Wllama</h2>
          <button
            onClick={createNewConversation}
            disabled={!wllamaRef.current}
            className="w-full bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2 mb-2"
          >
            <Plus className="w-4 h-4" />
            New conversation
          </button>
          <button
            onClick={() => setShowModelManager(true)}
            className="w-full bg-blue-600 hover:bg-blue-700 text-white font-semibold py-2 px-4 rounded-lg transition-colors flex items-center justify-center gap-2"
          >
            <HardDrive className="w-4 h-4" />
            Manage models
          </button>
        </div>

        <div className="flex-1 overflow-y-auto p-2">
          {conversations.map(conv => (
            <div
              key={conv.id}
              className={`group relative mb-1 p-3 rounded-lg cursor-pointer transition-colors ${currentConversationId === conv.id
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

      {/* Model Manager Modal */}
      {showModelManager && (
        <div className="fixed inset-0 bg-black/50 backdrop-blur-sm z-50 flex items-center justify-center p-4">
          <div className="bg-slate-900 rounded-2xl border border-white/20 max-w-2xl w-full max-h-[80vh] overflow-y-auto">
            <div className="p-6 border-b border-white/10 flex items-center justify-between sticky top-0 bg-slate-900">
              <h2 className="text-2xl font-bold text-white">Manage Models</h2>
              <button
                onClick={() => setShowModelManager(false)}
                className="text-white hover:bg-white/10 p-2 rounded-lg transition-colors"
              >
                <X className="w-5 h-5" />
              </button>
            </div>

            <div className="p-6">
              {/* Load Method Tabs */}
              <div className="flex gap-2 mb-6">
                <button
                  onClick={() => setLoadMethod('url')}
                  className={`flex-1 py-2 px-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 ${loadMethod === 'url'
                    ? 'bg-purple-600 text-white'
                    : 'bg-white/10 text-purple-200 hover:bg-white/20'
                    }`}
                >
                  <Globe className="w-4 h-4" />
                  From URL
                </button>
                <button
                  onClick={() => setLoadMethod('file')}
                  className={`flex-1 py-2 px-4 rounded-lg font-semibold transition-colors flex items-center justify-center gap-2 ${loadMethod === 'file'
                    ? 'bg-purple-600 text-white'
                    : 'bg-white/10 text-purple-200 hover:bg-white/20'
                    }`}
                >
                  <Upload className="w-4 h-4" />
                  From File
                </button>
              </div>

              {/* Load from URL */}
              {loadMethod === 'url' && (
                <div className="mb-6">
                  <label className="block text-white font-semibold mb-3">
                    Hugging Face Repository or Direct URL
                  </label>
                  <input
                    type="text"
                    value={modelUrl}
                    onChange={(e) => handleRepoInputChange(e.target.value)}
                    placeholder="e.g., ggml-org/gemma-3-270m-it-GGUF"
                    className="w-full bg-white/10 border border-white/20 text-white placeholder-gray-400 rounded-lg p-3 mb-3 focus:outline-none focus:ring-2 focus:ring-purple-500"
                  />

                  {fetchingFiles && (
                    <div className="flex items-center gap-2 text-purple-300 mb-3">
                      <Loader2 className="w-4 h-4 animate-spin" />
                      <span className="text-sm">Fetching repository files...</span>
                    </div>
                  )}

                  {availableFiles.length > 0 && (
                    <div className="mb-3">
                      <label className="block text-purple-200 text-sm mb-2">
                        Select a GGUF file ({availableFiles.length} available)
                      </label>
                      <select
                        value={selectedFile}
                        onChange={(e) => setSelectedFile(e.target.value)}
                        className="w-full bg-white/10 border border-white/20 text-white rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-purple-500"
                      >
                        <option value="">Choose a file...</option>
                        {availableFiles.map((file) => (
                          <option key={file} value={file}>
                            {file}
                          </option>
                        ))}
                      </select>
                    </div>
                  )}

                  <button
                    onClick={() => loadModelFromUrl(modelUrl)}
                    disabled={(!selectedFile && availableFiles.length > 0) || isLoading || !modelUrl}
                    className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        Loading... {loadProgress}%
                      </>
                    ) : (
                      <>
                        <Download className="w-5 h-5" />
                        {availableFiles.length > 0 ? 'Download & Load Selected' : 'Download & Load Model'}
                      </>
                    )}
                  </button>

                  <div className="mt-3 p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                    <p className="text-blue-200 text-xs">
                      <strong>Examples:</strong><br />
                      • ggml-org/gemma-3-270m-it-GGUF (Gemma)<br />
                      • Qwen/Qwen2.5-0.5B-Instruct-GGUF (Qwen)<br />
                      • TheBloke/Llama-2-7B-Chat-GGUF (Llama)<br />
                      • Or paste a direct .gguf URL
                    </p>
                  </div>
                </div>
              )}

              {/* Load from File */}
              {loadMethod === 'file' && (
                <div className="mb-6">
                  <label className="block text-white font-semibold mb-3">
                    Select Local .gguf File
                  </label>
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
                    className="w-full bg-purple-600 hover:bg-purple-700 text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2 mb-3"
                  >
                    <Upload className="w-5 h-5" />
                    Choose File
                  </button>
                  {modelFile && (
                    <p className="text-green-300 text-sm mb-3">
                      Selected: {modelFile.length} file(s)
                    </p>
                  )}
                  <button
                    onClick={loadModelFromFile}
                    disabled={!modelFile || isLoading}
                    className="w-full bg-green-600 hover:bg-green-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white font-semibold py-3 px-6 rounded-lg transition-colors flex items-center justify-center gap-2"
                  >
                    {isLoading ? (
                      <>
                        <Loader2 className="w-5 h-5 animate-spin" />
                        Loading... {loadProgress}%
                      </>
                    ) : (
                      <>
                        <Play className="w-5 h-5" />
                        Load Model
                      </>
                    )}
                  </button>
                </div>
              )}

              {/* Cached Models */}
              <div className="mt-8">
                <h3 className="text-white font-semibold mb-3">Cached Models</h3>
                {cachedModels.length === 0 ? (
                  <p className="text-gray-400 text-sm">No cached models found.</p>
                ) : (
                  <div className="space-y-2">
                    {cachedModels.map((model) => (
                      <div
                        key={model.url}
                        className="flex items-center justify-between p-3 bg-white/5 rounded-lg border border-white/10"
                      >
                        <div className="flex-1 min-w-0">
                          <p className="text-white text-sm truncate">{model.name}</p>
                          <p className="text-purple-300 text-xs">
                            {(model.size / 1024 / 1024).toFixed(1)} MB
                          </p>
                        </div>
                        <button
                          onClick={() => deleteCachedModel(model.url)}
                          className="text-red-400 hover:text-red-300 p-1 transition-colors"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    ))}
                  </div>
                )}
              </div>

              {/* ADD DISK USAGE HERE */}
              {diskUsage && (
                <div className="mt-4 p-3 bg-blue-500/10 border border-blue-500/30 rounded-lg">
                  <p className="text-blue-200 text-xs">
                    Storage: {(diskUsage.used / 1024 / 1024).toFixed(1)}MB / 
                    {(diskUsage.quota / 1024 / 1024).toFixed(1)}MB used
                    (${((diskUsage.used / diskUsage.quota) * 100).toFixed(1)}%)
                  </p>
                </div>
              )}

              {/* Processed Documents Section - THIS IS CORRECTLY PLACED */}
              <div className="mt-8">
                <h3 className="text-white font-semibold mb-3">Processed Documents ({processedDocuments.length})</h3>
                <div className="space-y-2 max-h-60 overflow-y-auto">
                  {processedDocuments.length === 0 ? (
                    <p className="text-purple-300 text-sm">No documents processed yet</p>
                  ) : (
                    processedDocuments.map((doc) => (
                      <div
                        key={doc.id}
                        className="bg-white/5 border border-white/10 rounded-lg p-3 flex items-center justify-between"
                      >
                        <div className="flex-1">
                          <p className="text-white text-sm font-medium">{doc.name}</p>
                          <p className="text-purple-300 text-xs">
                            {doc.chunks.length} chunks • {(doc.size / 1024).toFixed(1)} KB
                          </p>
                        </div>
                        <button
                          onClick={() => {
                            documentProcessor.deleteDocument(doc.id);
                            setProcessedDocuments(prev => prev.filter(d => d.id !== doc.id));
                          }}
                          className="text-red-400 hover:text-red-300 p-1 transition-colors"
                        >
                          <Trash2 className="w-4 h-4" />
                        </button>
                      </div>
                    ))
                  )}
                </div>
              </div>
            </div>
          </div>
        </div>
      )}

      {/* Main Chat Area */}
      <div className="flex-1 flex flex-col">
        {/* Header */}
        <header className="bg-slate-900/50 backdrop-blur-lg border-b border-white/10 p-4 flex items-center justify-between">
          <div className="flex items-center gap-4">
            <button
              onClick={() => setSidebarOpen(!sidebarOpen)}
              className="text-white hover:bg-white/10 p-2 rounded-lg transition-colors"
            >
              {sidebarOpen ? <X className="w-5 h-5" /> : <Menu className="w-5 h-5" />}
            </button>
            <h1 className="text-white font-bold text-xl">Wllama Chat</h1>
            {embeddingModelStatus && (
              <span className={`text-xs px-2 py-1 rounded-full ${
                embeddingModelStatus.includes('ready') 
                  ? 'bg-green-500/20 text-green-300' 
                  : embeddingModelStatus.includes('failed')
                  ? 'bg-red-500/20 text-red-300'
                  : 'bg-blue-500/20 text-blue-300'
              }`}>
                {embeddingModelStatus}
              </span>
            )}
          </div>

          <div className="flex items-center gap-3">
            {status && (
              <div className="text-purple-300 text-sm flex items-center gap-2">
                <Loader2 className="w-4 h-4 animate-spin" />
                {status}
              </div>
            )}
            {currentConversation && (
              <button
                onClick={exportChat}
                className="text-white hover:bg-white/10 p-2 rounded-lg transition-colors"
                title="Export chat"
              >
                <Save className="w-5 h-5" />
              </button>
            )}
          </div>
        </header>

        {/* Chat Messages */}
        <div className="flex-1 overflow-y-auto p-4 space-y-6">
          {!wllamaRef.current ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center text-white max-w-md">
                <HardDrive className="w-16 h-16 mx-auto mb-4 text-purple-400" />
                <h3 className="text-xl font-bold mb-2">No Model Loaded</h3>
                <p className="text-purple-200 mb-4">
                  Please load a model to start chatting. You can download from Hugging Face or load a local file.
                </p>
                <button
                  onClick={() => setShowModelManager(true)}
                  className="bg-purple-600 hover:bg-purple-700 text-white font-semibold py-2 px-6 rounded-lg transition-colors"
                >
                  Load Model
                </button>
              </div>
            </div>
          ) : messages.length === 0 ? (
            <div className="flex items-center justify-center h-full">
              <div className="text-center text-white max-w-md">
                <MessageSquare className="w-16 h-16 mx-auto mb-4 text-purple-400" />
                <h3 className="text-xl font-bold mb-2">Start a Conversation</h3>
                <p className="text-purple-200">
                  Send a message to begin chatting with your loaded model.
                </p>
              </div>
            </div>
          ) : (
            messages.map((message, index) => (
              <div key={index} className={`flex gap-4 ${message.role === 'user' ? 'flex-row-reverse' : 'flex-row'}`}>
                <div className={`w-8 h-8 rounded-full flex items-center justify-center flex-shrink-0 ${
                  message.role === 'user' ? 'bg-purple-600' : 'bg-blue-600'
                }`}>
                  {message.role === 'user' ? (
                    <User className="w-4 h-4 text-white" />
                  ) : (
                    <MessageCircle className="w-4 h-4 text-white" />
                  )}
                </div>
                <div className={`flex-1 max-w-3xl ${message.role === 'user' ? 'text-right' : 'text-left'}`}>
                  {/* Show document sources for assistant messages */}
                  {message.role === 'assistant' && message.documentSources && (
                    <div className="mb-2 p-2 bg-green-500/20 border border-green-500/30 rounded-lg">
                      <p className="text-green-300 text-xs">
                        📚 Based on: {message.documentSources.join(', ')}
                      </p>
                    </div>
                  )}
                  
                  <div className={`inline-block rounded-2xl px-4 py-2 ${
                    message.role === 'user' ? 'bg-purple-600 text-white' : 'bg-white/10 text-white'
                  }`}>
                    <p className="whitespace-pre-wrap">{message.content}</p>
                  </div>
                </div>
              </div>
            ))
          )}
          <div ref={messagesEndRef} />
        </div>

        {/* File Attachments Preview */}
        {attachedFiles.length > 0 && (
          <div className="px-4 py-2 border-t border-white/10 bg-slate-800/50">
            <div className="flex flex-wrap gap-2">
              {attachedFiles.map((file, index) => (
                <div
                  key={index}
                  className="flex items-center gap-2 bg-purple-600/30 border border-purple-500/50 rounded-lg px-3 py-2 text-sm text-white"
                >
                  <FileText className="w-4 h-4" />
                  <span className="max-w-xs truncate">{file.name}</span>
                  <button
                    onClick={() => removeAttachedFile(index)}
                    className="text-red-300 hover:text-red-200 transition-colors"
                  >
                    <X className="w-4 h-4" />
                  </button>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* Input Area */}
        <div className="border-t border-white/10 p-4 bg-slate-900/50 backdrop-blur-lg">
          {error && (
            <div className="mb-3 p-3 bg-red-500/20 border border-red-500/50 text-red-200 rounded-lg text-sm">
              {error}
            </div>
          )}

          <div className="flex gap-2">
            {/* File Upload Button */}
            <input
              ref={documentInputRef}
              type="file"
              accept=".pdf,.csv,.doc,.docx,.txt"
              multiple
              onChange={handleDocumentSelect}
              className="hidden"
            />
            <button
              onClick={() => documentInputRef.current?.click()}
              disabled={isProcessingFiles || isGenerating}
              className="flex-shrink-0 bg-blue-600 hover:bg-blue-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white p-3 rounded-lg transition-colors"
              title="Attach PDF, CSV, DOC, DOCX, or TXT files"
            >
              <FileText className="w-5 h-5" />
            </button>

            {/* Chat Template Selector */}
            <select
              value={chatTemplate}
              onChange={(e) => setChatTemplate(e.target.value as any)}
              className="flex-shrink-0 bg-white/10 border border-white/20 text-white rounded-lg p-3 focus:outline-none focus:ring-2 focus:ring-purple-500"
            >
              <option value="gemma">Gemma</option>
              <option value="qwen">Qwen</option>
              <option value="llama">Llama</option>
              <option value="chatml">ChatML</option>
            </select>

            <div className="flex-1 relative">
              <textarea
                ref={inputRef}
                value={input}
                onChange={(e) => setInput(e.target.value)}
                onKeyPress={handleKeyPress}
                placeholder={wllamaRef.current ? "Type your message... (Shift+Enter for new line)" : "Load a model to start chatting..."}
                disabled={!wllamaRef.current || isGenerating || isProcessingFiles}
                className="w-full bg-white/10 border border-white/20 text-white placeholder-gray-400 rounded-lg p-3 pr-12 focus:outline-none focus:ring-2 focus:ring-purple-500 disabled:opacity-50 resize-none"
                rows={1}
                style={{ minHeight: '3rem', maxHeight: '12rem' }}
              />
            </div>

            <button
              onClick={sendMessage}
              disabled={!wllamaRef.current || isGenerating || isProcessingFiles || (!input.trim() && attachedFiles.length === 0)}
              className="flex-shrink-0 bg-purple-600 hover:bg-purple-700 disabled:bg-gray-600 disabled:cursor-not-allowed text-white p-3 rounded-lg transition-colors"
            >
              {isGenerating ? (
                <Loader2 className="w-5 h-5 animate-spin" />
              ) : (
                <Send className="w-5 h-5" />
              )}
            </button>
          </div>

          {/* Processing indicator */}
          {isProcessingFiles && (
            <div className="mt-2 flex items-center gap-2 text-purple-300 text-sm">
              <Loader2 className="w-4 h-4 animate-spin" />
              Processing {attachedFiles.length} file(s)...
            </div>
          )}
        </div>
      </div>
    </div>
  );
}