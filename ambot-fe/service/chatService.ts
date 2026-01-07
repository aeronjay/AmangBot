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
  bart_output?: string;
}

export interface StreamCallbacks {
  onToken: (token: string) => void;
  onMetadata?: (metadata: { chunks: ChunkData[]; retrieved_chunks: ChunkData[]; prompt: string; bart_output?: string }) => void;
  onDone: () => void;
  onError: (error: string) => void;
}

class ChatService {
  private baseUrl = import.meta.env.VITE_API_URL || 'http://localhost:8000';

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

      // Log Question and Retrieved Chunks
      if (localStorage.getItem('ADMIN') === '1') {
        console.log('%c❓ Question:', 'color: #2196F3; font-weight: bold; font-size: 14px;');
        console.log(message);
        
        console.log('All chunks got:', retrieved_chunks);
        console.log('Passed to LLM:', chunks);
      }

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

  async streamResponse(message: string, history: ChatMessage[] = [], callbacks: StreamCallbacks): Promise<void> {
    try {
      // Convert history to backend format
      const backendHistory = history.map(msg => ({
        role: msg.sender,
        content: msg.message
      }));

      const response = await fetch(`${this.baseUrl}/chat/stream`, {
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

      const reader = response.body?.getReader();
      if (!reader) {
        throw new Error('No response body');
      }

      const decoder = new TextDecoder();
      let buffer = '';

      while (true) {
        const { done, value } = await reader.read();
        
        if (done) {
          break;
        }

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split('\n');
        buffer = lines.pop() || '';

        for (const line of lines) {
          if (line.startsWith('data: ')) {
            try {
              const data = JSON.parse(line.slice(6));
              
              if (data.type === 'metadata') {
                // Log Question and Retrieved Chunks
                if (localStorage.getItem('ADMIN') === '1') {
                  console.log('%c❓ Question:', 'color: #2196F3; font-weight: bold; font-size: 14px;');
                  console.log(message);

                  console.log('All chunks got:', data.retrieved_chunks);
                  console.log('Passed to LLM:', data.chunks);
                  console.log('BART OUTPUT:', data.bart_output);
                }

                if (callbacks.onMetadata) {
                  callbacks.onMetadata({
                    chunks: data.chunks,
                    retrieved_chunks: data.retrieved_chunks,
                    prompt: data.prompt,
                    bart_output: data.bart_output
                  });
                }
              } else if (data.type === 'token') {
                callbacks.onToken(data.content);
              } else if (data.type === 'done') {
                callbacks.onDone();
              }
            } catch (e) {
              console.error('Error parsing SSE data:', e);
            }
          }
        }
      }
    } catch (error) {
      console.error('Chat stream error:', error);
      callbacks.onError(error instanceof Error ? error.message : 'Unknown error');
    }
  }
}

export const chatService = new ChatService();