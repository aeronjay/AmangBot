
import { useState, useEffect, useRef } from 'react';
import { getRoleDisplayName, type UserRole } from '../src/utils/roleStorage';
import { chatService, type ChatMessage } from '../service/chatService';
import MessageBubble from './MessageBubble';
import TypingAnimation from './TypingAnimation';

interface ChatInterfaceProps {
  userRole: UserRole | null;
  onOpenSettings: () => void;
  darkMode: boolean;
}

const CHAT_HISTORY_KEY = 'ambot-chat-history';

function ChatInterface({ userRole, onOpenSettings, darkMode }: ChatInterfaceProps) {
  const [messages, setMessages] = useState<ChatMessage[]>([]);
  const [inputValue, setInputValue] = useState('');
  const [isTyping, setIsTyping] = useState(false);
  const [showClearConfirm, setShowClearConfirm] = useState(false);
  const messagesEndRef = useRef<HTMLDivElement>(null);
  const chatBodyRef = useRef<HTMLDivElement>(null);
  const textareaRef = useRef<HTMLTextAreaElement>(null);

  // Auto-resize textarea
  useEffect(() => {
    if (textareaRef.current) {
      textareaRef.current.style.height = 'auto';
      const scrollHeight = textareaRef.current.scrollHeight;
      textareaRef.current.style.height = `${Math.min(scrollHeight, 120)}px`;
      
      // Only show scrollbar if content exceeds max height
      if (scrollHeight > 120) {
        textareaRef.current.style.overflowY = 'auto';
      } else {
        textareaRef.current.style.overflowY = 'hidden';
      }
    }
  }, [inputValue]);

  // Auto-scroll to bottom when new messages are added
  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  // Save messages to localStorage whenever they change
  useEffect(() => {
    if (messages.length > 0) {
      localStorage.setItem(CHAT_HISTORY_KEY, JSON.stringify(messages));
    }
  }, [messages]);

  // Load chat history from localStorage on mount
  useEffect(() => {
    const savedHistory = localStorage.getItem(CHAT_HISTORY_KEY);
    if (savedHistory) {
      try {
        const parsedHistory = JSON.parse(savedHistory);
        // Convert timestamp strings back to Date objects
        const historyWithDates = parsedHistory.map((msg: ChatMessage) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }));
        setMessages(historyWithDates);
      } catch (error) {
        console.error('Failed to parse chat history:', error);
        addWelcomeMessage();
      }
    } else {
      addWelcomeMessage();
    }
  }, []);

  const addWelcomeMessage = () => {
    const welcomeMessage: ChatMessage = {
      id: 'welcome-1',
      message: "Hello! I'm AmBot, your school assistant. How can I help you today?",
      sender: 'bot',
      timestamp: new Date()
    };
    setMessages([welcomeMessage]);
  };

  const handleNewChat = () => {
    setShowClearConfirm(true);
  };

  const confirmClearChat = () => {
    localStorage.removeItem(CHAT_HISTORY_KEY);
    const welcomeMessage: ChatMessage = {
      id: `welcome-${Date.now()}`,
      message: "Hello! I'm AmBot, your school assistant. How can I help you today?",
      sender: 'bot',
      timestamp: new Date()
    };
    setMessages([welcomeMessage]);
    setShowClearConfirm(false);
  };

  const handleSendMessage = async () => {
    if (!inputValue.trim() || isTyping) return;

    const userMessage: ChatMessage = {
      id: `user-${Date.now()}`,
      message: inputValue.trim(),
      sender: 'user',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, userMessage]);
    const currentInput = inputValue.trim();
    setInputValue('');
    setIsTyping(true);

    // Create a placeholder bot message that will be updated with streamed content
    const botMessageId = `bot-${Date.now()}`;
    const botMessage: ChatMessage = {
      id: botMessageId,
      message: '',
      sender: 'bot',
      timestamp: new Date()
    };

    setMessages(prev => [...prev, botMessage]);

    try {
      await chatService.streamResponse(currentInput, messages, {
        onToken: (token: string) => {
          setMessages(prev => 
            prev.map(msg => {
              if (msg.id === botMessageId) {
                const newMessage = msg.message + token;
                return { ...msg, message: msg.message === '' ? newMessage.trimStart() : newMessage };
              }
              return msg;
            })
          );
        },
        onMetadata: (metadata) => {
          // Metadata is already logged in the service
        },
        onDone: () => {
          setIsTyping(false);
        },
        onError: (error: string) => {
          console.error('Stream error:', error);
          setMessages(prev => 
            prev.map(msg => 
              msg.id === botMessageId 
                ? { ...msg, message: "Sorry, I'm having trouble connecting right now. Please try again later." }
                : msg
            )
          );
          setIsTyping(false);
        }
      });
    } catch (error) {
      setMessages(prev => 
        prev.map(msg => 
          msg.id === botMessageId 
            ? { ...msg, message: "Sorry, I'm having trouble connecting right now. Please try again later." }
            : msg
        )
      );
      setIsTyping(false);
    }
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      handleSendMessage();
    }
  };

    return (
        <div className={`chat-interface w-full sm:w-[90%] md:w-[70%] lg:w-[50%] xl:w-[40%] mx-auto my-4 h-[95vh] ${darkMode ? 'bg-gray-800' : 'bg-gray-50'} rounded-lg flex flex-col shadow-lg transition-colors duration-200`}>

            <div className={`chat-header flex items-center justify-between p-4 ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-200'} border-b transition-colors duration-200`}>
                <div className="flex items-center gap-3">
                    <img src="Ambot.png" alt="Ambot Img" className="w-10 h-10 rounded-lg"/>
                    <div>
                        <h1 className={`text-2xl font-semibold ${darkMode ? 'text-red-400' : 'text-red-900'}`}>Amang Bot</h1>
                        <div className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>
                          {userRole ? getRoleDisplayName(userRole) : 'Loading...'}
                        </div>
                    </div>
                </div>
                <div className="flex items-center gap-2">
                    <button 
                      onClick={handleNewChat}
                      className={`p-1 ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-100'} rounded transition-colors duration-200`}
                      title="Clear Chat History"
                    >
                        <svg className={`w-5 h-5 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                        </svg>
                    </button>
                    <button 
                      onClick={onOpenSettings}
                      className={`p-1 ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-100'} rounded transition-colors duration-200`}
                      title="Settings"
                    >
                        <svg className={`w-5 h-5 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.50 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                        </svg>
                    </button>
                </div>
            </div>

            
            <div ref={chatBodyRef} className="chat-body flex-1 overflow-y-auto p-4 space-y-4">
                {messages.map(message => (
                    message.message ? (
                        <MessageBubble 
                            key={message.id} 
                            message={message} 
                            darkMode={darkMode} 
                        />
                    ) : null
                ))}
                {isTyping && messages[messages.length - 1]?.message === '' && (
                    <div className="flex justify-start mb-4">
                        <img 
                            src="Ambot.png" 
                            alt="AmBot" 
                            className="w-8 h-8 rounded-full mr-3 mt-1 flex-shrink-0"
                        />
                        <div className={`max-w-[80%] px-4 py-2 rounded-lg rounded-bl-sm ${
                            darkMode ? 'bg-gray-700' : 'bg-gray-200'
                        } shadow-sm transition-colors duration-200`}>
                            <TypingAnimation darkMode={darkMode} />
                        </div>
                    </div>
                )}
                <div ref={messagesEndRef} />
            </div>

            
            <div className={`chat-footer p-4 ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-200'} border-t transition-colors duration-200`}>
                <div className="flex items-end gap-2">
                    <textarea 
                        ref={textareaRef}
                        value={inputValue}
                        onChange={(e) => setInputValue(e.target.value)}
                        onKeyDown={handleKeyDown}
                        placeholder="Type your message here..." 
                        rows={1}
                        className={`flex-1 px-4 py-2 border ${darkMode ? 'border-gray-500 bg-gray-600 text-gray-100 placeholder-gray-400 focus:ring-red-400' : 'border-gray-300 bg-white text-gray-900 placeholder-gray-500 focus:ring-red-900'} rounded-lg focus:outline-none focus:ring-2 focus:border-transparent transition-colors duration-200 resize-none overflow-hidden min-h-[42px] max-h-[120px]`}
                        disabled={isTyping}
                    />
                    <button 
                        onClick={handleSendMessage}
                        disabled={!inputValue.trim() || isTyping}
                        className={`p-2 ${
                            !inputValue.trim() || isTyping 
                                ? `${darkMode ? 'bg-gray-600 text-gray-400' : 'bg-gray-300 text-gray-500'} cursor-not-allowed` 
                                : `${darkMode ? 'bg-red-700 hover:bg-red-600' : 'bg-red-900 hover:bg-red-800'} text-white`
                        } rounded-lg transition-colors duration-200`}
                    >
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                        </svg>
                    </button>
                </div>
            </div>

            {showClearConfirm && (
                <div 
                    className="fixed inset-0 z-50 flex items-center justify-center p-4"
                    style={{backgroundColor: darkMode ? 'rgba(0, 0, 0, 0.8)' : 'rgba(255, 255, 255, 0.8)'}}
                >
                    <div className={`${darkMode ? 'bg-gray-800 text-white' : 'bg-white text-gray-900'} p-6 rounded-lg shadow-xl max-w-sm w-full mx-4`}>
                        <h3 className="text-lg font-semibold mb-2">Clear Chat History?</h3>
                        <p className={`mb-6 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>
                            Are you sure you want to clear all chat history? This action cannot be undone.
                        </p>
                        <div className="flex justify-end gap-3">
                            <button
                                onClick={() => setShowClearConfirm(false)}
                                className={`px-4 py-2 rounded-lg ${darkMode ? 'hover:bg-gray-700' : 'hover:bg-gray-100'} transition-colors`}
                            >
                                Cancel
                            </button>
                            <button
                                onClick={confirmClearChat}
                                className="px-4 py-2 bg-red-600 text-white rounded-lg hover:bg-red-700 transition-colors"
                            >
                                Clear Chat
                            </button>
                        </div>
                    </div>
                </div>
            )}
        </div>
    );
}

export default ChatInterface;
