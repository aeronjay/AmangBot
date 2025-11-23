export interface ChatMessage {
  id: string;
  message: string;
  sender: 'user' | 'bot';
  timestamp: Date;
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

      return {
        success: true,
        message: data.response,
        sources: sources,
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