import type { ChatMessage } from '../service/chatService';

interface MessageBubbleProps {
  message: ChatMessage;
  darkMode: boolean;
}

function MessageBubble({ message, darkMode }: MessageBubbleProps) {
  const isUser = message.sender === 'user';

  const formatLine = (text: string) => {
    // Handle bullet points - keep them as plain text to avoid merging with bold logic
    let content = text;
    let prefix = '';
    
    // Check for lines starting with "* " or "- " (bullets)
    const bulletMatch = text.match(/^(\s*[\*-]\s+)(.*)/);
    if (bulletMatch) {
      prefix = bulletMatch[1];
      content = bulletMatch[2];
    }

    // Split by **text** or *text* (non-greedy to handle multiple on one line)
    // We prioritize ** matching over *
    const parts = content.split(/(\*\*.*?\*\*|\*.*?\*)/g);
    
    const formattedContent = parts.map((part, index) => {
      // Check for **bold**
      if (part.startsWith('**') && part.endsWith('**') && part.length >= 4) {
        return <strong key={index}>{part.slice(2, -2)}</strong>;
      }
      // Check for *bold* (requested by user)
      if (part.startsWith('*') && part.endsWith('*') && part.length >= 2) {
        return <strong key={index}>{part.slice(1, -1)}</strong>;
      }
      return part;
    });

    return (
      <>
        {prefix}
        {formattedContent}
      </>
    );
  };

  const formatMessage = (text: string) => {
    // Process line by line to correctly handle bullets and newlines
    return text.split('\n').map((line, i) => (
      <div key={i} className="min-h-[1.2em]">
        {formatLine(line)}
      </div>
    ));
  };
  
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
        <div className="text-sm leading-relaxed break-words">
          {formatMessage(message.message)}
        </div>
      </div>
    </div>
  );
}

export default MessageBubble;