export interface ChatMessage {
  id: string;
  message: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

export interface ChunkData {
  content: string;
  source: string;
  category: string;
  topic: string;
}

export interface ChatSource {
  text: string;
  score: number;
  chunk_id: number;
}

export interface ChatResponse {
  success: boolean;
  message: string;
  sources?: ChatSource[];
  chunks?: ChunkData[];
  prompt?: string;
  retrieved_chunks?: ChunkData[];
  error?: string;
}

class ChatService {
  private baseUrl = 'http://localhost:8000';

  async getResponse(message: string, history: ChatMessage[] = []): Promise<ChatResponse> {
    try {
      // Convert history to backend format
      const backendHistory = history.map(msg => ({
        role: msg.sender,
        content: msg.message
      }));

      const response = await fetch(`${this.baseUrl}/chat`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ 
          message: message,
          history: backendHistory
        }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Failed to parse error response' }));
        throw new Error(errorData.detail || errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      // Map backend context (string[]) to frontend ChatSource structure
      const sources: ChatSource[] = (data.context || []).map((text: string, index: number) => ({
        text: text,
        score: 0,
        chunk_id: index
      }));

      // Get chunks data from backend
      const chunks: ChunkData[] = data.chunks || [];
      const prompt: string = data.prompt || '';
      const retrieved_chunks: ChunkData[] = data.retrieved_chunks || [];

      // Log retrieved chunks to console
      console.log('%c📦 Retrieved Chunks:', 'color: #4CAF50; font-weight: bold; font-size: 14px;');
      console.table(chunks.map((c, i) => ({
        '#': i + 1,
        source: c.source,
        category: c.category,
        topic: c.topic,
        content: c.content.substring(0, 100) + '...'
      })));

      // Log all retrieved chunks (before filtering)
      console.log('%c📚 All Retrieved Chunks (before final selection):', 'color: #2196F3; font-weight: bold; font-size: 14px;');
      console.table(retrieved_chunks.map((c, i) => ({
        '#': i + 1,
        source: c.source,
        category: c.category,
        topic: c.topic,
        content: c.content.substring(0, 100) + '...'
      })));

      // Log the prompt sent to LLM
      console.log('%c📝 Final Prompt to LLM:', 'color: #FF9800; font-weight: bold; font-size: 14px;');
      console.log(prompt);

      return {
        success: true,
        message: data.response,
        sources: sources,
        chunks: chunks,
        prompt: prompt,
        retrieved_chunks: retrieved_chunks,
      };
    } catch (error) {
      console.error('Chat service error:', error);
      return {
        success: false,
        message: "Sorry, I'm having trouble connecting to the backend. Please make sure it's running and try again.",
        error: error instanceof Error ? error.message : 'Unknown error',
      };
    }
  }
}

export const chatService = new ChatService();