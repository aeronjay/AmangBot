
import { getRoleDisplayName, type UserRole } from '../src/utils/roleStorage';

interface ChatInterfaceProps {
  userRole: UserRole | null;
  onRoleChange: () => void;
}

function ChatInterface({ userRole, onRoleChange }: ChatInterfaceProps) {

    return (
        <div className="chat-interface w-full sm:w-[90%] md:w-[70%] lg:w-[50%] xl:w-[40%] mx-auto my-4 h-[95vh] bg-gray-50 rounded-lg flex flex-col">

            <div className="chat-header flex items-center justify-between p-4 bg-white border-b border-gray-200">
                <div className="flex items-center gap-3">
                    <img src="Ambot.png" alt="Ambot Img" className="w-10 h-10 rounded-lg"/>
                    <div>
                        <h1 className="text-2xl font-semibold text-red-900">Amang Bot</h1>
                        <div className="text-sm text-gray-500">
                          {userRole ? getRoleDisplayName(userRole) : 'Loading...'}
                        </div>
                    </div>
                </div>
                <button 
                  onClick={onRoleChange}
                  className="p-1 hover:bg-gray-100 rounded"
                  title="Change Role"
                >
                    <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M10.325 4.317c.426-1.756 2.924-1.756 3.35 0a1.724 1.724 0 002.573 1.066c1.543-.94 3.31.826 2.37 2.37a1.724 1.724 0 001.065 2.572c1.756.426 1.756 2.924 0 3.35a1.724 1.724 0 00-1.066 2.573c.94 1.543-.826 3.31-2.37 2.37a1.724 1.724 0 00-2.572 1.065c-.426 1.756-2.924 1.756-3.35 0a1.724 1.724 0 00-2.573-1.066c-1.543.94-3.31-.826-2.37-2.37a1.724 1.724 0 00-1.065-2.572c-1.756-.426-1.756-2.924 0-3.35a1.724 1.724 0 001.066-2.573c-.94-1.543.826-3.31 2.37-2.37.996.608 2.296.07 2.572-1.065z" />
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                    </svg>
                </button>
            </div>

            
            <div className="chat-body flex-1 overflow-y-auto p-4 space-y-4">
                
            </div>

            
            <div className="chat-footer p-4 bg-white border-t border-gray-200">
                <div className="flex items-center gap-2">
                    <input 
                        type="text" 
                        placeholder="Type your message here..." 
                        className="flex-1 px-4 py-2 border border-gray-300 rounded-lg focus:outline-none focus:ring-2 focus:ring-red-900 focus:border-transparent"
                    />
                    <button className="p-2 bg-red-900 text-white rounded-lg hover:bg-red-800 transition-colors">
                        <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 19l9 2-9-18-9 18 9-2zm0 0v-8" />
                        </svg>
                    </button>
                </div>
            </div>
        </div>
    );
}

export default ChatInterface;
