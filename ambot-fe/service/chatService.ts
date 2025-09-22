export interface ChatMessage {
  id: string;
  message: string;
  sender: 'user' | 'bot';
  timestamp: Date;
}

export interface ChatResponse {
  success: boolean;
  message: string;
  error?: string;
}

class ChatService {
  private baseUrl = 'https://mybackend.com';

  async getResponse(message: string): Promise<ChatResponse> {
    
    try {
      await new Promise(resolve => setTimeout(resolve, 1000));
      
      return {
        success: true,
        message: "Haha tanginamo gago ka pinagsasabimo"
      };
      
    } catch (error) {
      console.error('Chat service error:', error);
      return {
        success: false,
        message: "Sorry, I'm having trouble connecting right now. Please try again later.",
        error: error instanceof Error ? error.message : 'Unknown error'
      };
    }
  }
}

export const chatService = new ChatService();