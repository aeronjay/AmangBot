interface TypingAnimationProps {
  darkMode: boolean;
}

function TypingAnimation({ darkMode }: TypingAnimationProps) {
  return (
    <div className="flex items-center space-x-1">
      <div className="flex space-x-1">
        <div className={`w-2 h-2 ${darkMode ? 'bg-gray-400' : 'bg-gray-600'} rounded-full animate-bounce`} style={{ animationDelay: '0ms' }}></div>
        <div className={`w-2 h-2 ${darkMode ? 'bg-gray-400' : 'bg-gray-600'} rounded-full animate-bounce`} style={{ animationDelay: '150ms' }}></div>
        <div className={`w-2 h-2 ${darkMode ? 'bg-gray-400' : 'bg-gray-600'} rounded-full animate-bounce`} style={{ animationDelay: '300ms' }}></div>
      </div>
      <span className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-600'} ml-2`}>
        AmBot is typing...
      </span>
    </div>
  );
}

export default TypingAnimation;