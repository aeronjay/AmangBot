import type { ChatMessage } from '../service/chatService';

interface MessageBubbleProps {
  message: ChatMessage;
  darkMode: boolean;
}

function MessageBubble({ message, darkMode }: MessageBubbleProps) {
  const isUser = message.sender === 'user';
  
  return (
    <div className={`flex ${isUser ? 'justify-end' : 'justify-start'} mb-4`}>
      {!isUser && (
        <img 
          src="Ambot.png" 
          alt="AmBot" 
          className="w-8 h-8 rounded-full mr-3 mt-1 flex-shrink-0"
        />
      )}
      <div 
        className={`max-w-[80%] px-4 py-2 rounded-lg ${
          isUser 
            ? `${darkMode ? 'bg-yellow-600 text-white' : 'bg-yellow-400 text-gray-900'} rounded-br-sm` 
            : `${darkMode ? 'bg-gray-700 text-gray-100' : 'bg-gray-200 text-gray-900'} rounded-bl-sm`
        } shadow-sm transition-colors duration-200`}
      >
        <p className="text-sm leading-relaxed whitespace-pre-wrap break-words">
          {message.message}
        </p>
      </div>
    </div>
  );
}

export default MessageBubble;