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
  private baseUrl = 'http://127.0.0.1:5000';

  async getResponse(message: string): Promise<ChatResponse> {
    try {
      const response = await fetch(`${this.baseUrl}/ask`, {
        method: 'POST',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ question: message }),
      });

      if (!response.ok) {
        const errorData = await response.json().catch(() => ({ error: 'Failed to parse error response' }));
        throw new Error(errorData.error || `HTTP error! status: ${response.status}`);
      }

      const data = await response.json();

      return {
        success: true,
        message: data.answer,
        sources: data.sources,
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