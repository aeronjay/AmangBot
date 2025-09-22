import React, { useRef } from 'react';

interface RoleSelectionModalProps {
  isOpen: boolean;
  onRoleSelect: (role: string) => void;
  onClose: () => void;
}

const RoleSelectionModal: React.FC<RoleSelectionModalProps> = ({ isOpen, onRoleSelect, onClose }) => {
  const modalRef = useRef<HTMLDivElement>(null);

  if (!isOpen) return null;

  const handleRoleClick = (role: string) => {
    onRoleSelect(role);
    onClose();
  };

  const handleCloseClick = () => {
    // Set default role to inquiring/future student when closing manually
    onRoleSelect('inquiring-student');
    onClose();
  };

  const roles = [
    {
      id: 'current-student',
      title: 'Current Student',
      icon: (
        <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M16 7a4 4 0 11-8 0 4 4 0 018 0zM12 14a7 7 0 00-7 7h14a7 7 0 00-7-7z" />
        </svg>
      )
    },
    {
      id: 'inquiring-student',
      title: 'Inquiring/Future Student',
      icon: (
        <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M17 20h5v-2a3 3 0 00-5.356-1.857M17 20H7m10 0v-2c0-.656-.126-1.283-.356-1.857M7 20H2v-2a3 3 0 015.356-1.857M7 20v-2c0-.656.126-1.283.356-1.857m0 0a5.002 5.002 0 019.288 0M15 7a3 3 0 11-6 0 3 3 0 016 0zm6 3a2 2 0 11-4 0 2 2 0 014 0zM7 10a2 2 0 11-4 0 2 2 0 014 0z" />
        </svg>
      )
    },
    {
      id: 'teacher',
      title: 'Teacher',
      icon: (
        <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 6.253v13m0-13C10.832 5.477 9.246 5 7.5 5S4.168 5.477 3 6.253v13C4.168 18.477 5.754 18 7.5 18s3.332.477 4.5 1.253m0-13C13.168 5.477 14.754 5 16.5 5c1.746 0 3.332.477 4.5 1.253v13C19.832 18.477 18.246 18 16.5 18c-1.746 0-3.332.477-4.5 1.253" />
        </svg>
      )
    },
    {
      id: 'school-admin',
      title: 'School Admin',
      icon: (
        <svg className="w-6 h-6 text-red-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
          <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 21V5a2 2 0 00-2-2H7a2 2 0 00-2 2v16m14 0h2m-2 0h-5m-9 0H3m2 0h5M9 7h1m-1 4h1m4-4h1m-1 4h1m-5 10v-5a1 1 0 011-1h2a1 1 0 011 1v5m-4 0h4" />
        </svg>
      )
    }
  ];

  return (
    <div className="fixed inset-0 flex items-center justify-center z-50 p-4" style={{backgroundColor: 'rgba(255, 255, 255, 0.8)'}}>
      <div ref={modalRef} className="bg-white rounded-lg shadow-xl max-w-md w-full relative">
        {/* Close button */}
        <button
          onClick={handleCloseClick}
          className="absolute top-4 right-4 text-gray-400 hover:text-gray-600 transition-colors"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
          </svg>
        </button>
        
        {/* Header */}
        <div className="text-center p-6 border-b border-gray-200">
          <div className="mb-4">
            <img src="Ambot.png" alt="AmBot" className="w-12 h-12 mx-auto rounded-lg" />
          </div>
          <h1 className="text-2xl font-bold text-red-900 mb-2">AmBot</h1>
          <h2 className="text-lg font-semibold text-gray-800">
            Select your role
          </h2>
        </div>

        {/* Role Selection Grid */}
        <div className="p-6">
          <div className="grid grid-cols-2 gap-3">
            {roles.map((role) => (
              <button
                key={role.id}
                onClick={() => handleRoleClick(role.id)}
                className="p-4 border border-gray-200 rounded-lg hover:border-red-300 hover:bg-red-50 transition-all duration-200 text-center group"
              >
                <div className="flex flex-col items-center space-y-2">
                  <div className="flex items-center justify-center w-10 h-10 bg-red-100 rounded-lg group-hover:bg-red-200 transition-colors">
                    {role.icon}
                  </div>
                  <h3 className="text-sm font-semibold text-gray-800">
                    {role.title}
                  </h3>
                </div>
              </button>
            ))}
          </div>
        </div>
      </div>
    </div>
  );
};

export default RoleSelectionModal;