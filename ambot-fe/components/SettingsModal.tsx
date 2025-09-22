import React from 'react';
import { getRoleDisplayName, type UserRole } from '../src/utils/roleStorage';

interface SettingsModalProps {
  isOpen: boolean;
  onClose: () => void;
  userRole: UserRole | null;
  onRoleChange: (role: UserRole) => void;
  darkMode: boolean;
  onDarkModeToggle: () => void;
  fontSize: 'Small' | 'Medium' | 'Large';
  onFontSizeChange: (size: 'Small' | 'Medium' | 'Large') => void;
}

function SettingsModal({
  isOpen,
  onClose,
  userRole,
  onRoleChange,
  darkMode,
  onDarkModeToggle,
  fontSize,
  onFontSizeChange
}: SettingsModalProps) {
  if (!isOpen) return null;

  const roles: UserRole[] = ['current-student', 'inquiring-student', 'teacher', 'school-admin'];

  const handleBackdropClick = (e: React.MouseEvent) => {
    if (e.target === e.currentTarget) {
      onClose();
    }
  };

  return (
    <div 
      className="fixed inset-0 flex items-center justify-center z-50 p-4"
      style={{backgroundColor: 'rgba(255, 255, 255, 0.8)'}}
      onClick={handleBackdropClick}
    >
      <div className="bg-white rounded-lg shadow-xl max-w-md w-full relative max-h-[90vh] overflow-y-auto">
        {/* Header */}
        <div className="flex items-center justify-between p-4 border-b border-gray-200">
          <h2 className="text-lg font-semibold text-gray-800">Settings</h2>
          <button
            onClick={onClose}
            className="text-gray-500 hover:text-gray-700 text-xl font-bold"
          >
            ×
          </button>
        </div>

        {/* Content */}
        <div className="p-4 space-y-6">
          {/* User Role */}
          <div>
            <label className="block text-sm font-medium text-gray-700 mb-2">
              User Role
            </label>
            <select
              value={userRole || 'current-student'}
              onChange={(e) => onRoleChange(e.target.value as UserRole)}
              className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-red-900 focus:border-transparent bg-white"
            >
              {roles.map((role) => (
                <option key={role} value={role}>
                  {getRoleDisplayName(role)}
                </option>
              ))}
            </select>
          </div>

          {/* Appearance */}
          <div>
            <h3 className="text-sm font-medium text-gray-700 mb-3">Appearance</h3>
            
            {/* Dark Mode */}
            <div className="flex items-center justify-between mb-4">
              <span className="text-sm text-gray-600">Dark Mode</span>
              <label className="relative inline-flex items-center cursor-pointer">
                <input
                  type="checkbox"
                  checked={darkMode}
                  onChange={onDarkModeToggle}
                  className="sr-only peer"
                />
                <div className="w-11 h-6 bg-gray-200 peer-focus:outline-none peer-focus:ring-4 peer-focus:ring-red-300 rounded-full peer peer-checked:after:translate-x-full peer-checked:after:border-white after:content-[''] after:absolute after:top-[2px] after:left-[2px] after:bg-white after:border-gray-300 after:border after:rounded-full after:h-5 after:w-5 after:transition-all peer-checked:bg-red-600"></div>
              </label>
            </div>

            {/* Font Size */}
            <div>
              <label className="block text-sm text-gray-600 mb-2">Font Size</label>
              <select
                value={fontSize}
                onChange={(e) => onFontSizeChange(e.target.value as 'Small' | 'Medium' | 'Large')}
                className="w-full px-3 py-2 border border-gray-300 rounded-md focus:outline-none focus:ring-2 focus:ring-red-900 focus:border-transparent bg-white"
              >
                <option value="Small">Small</option>
                <option value="Medium">Medium</option>
                <option value="Large">Large</option>
              </select>
            </div>
          </div>

          {/* Chat History */}
          <div>
            <h3 className="text-sm font-medium text-gray-700 mb-3">Chat History</h3>
            
            {/* Reset Chat Button */}
            <button
              disabled
              className="w-full mb-3 px-4 py-2 bg-red-800 text-white rounded-md opacity-50 cursor-not-allowed"
            >
              Reset Chat
            </button>

            {/* Report Chat Button */}
            <button
              disabled
              className="w-full px-4 py-2 border border-red-600 text-red-600 rounded-md opacity-50 cursor-not-allowed flex items-center justify-center gap-2"
            >
              <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 9v2m0 4h.01m-6.938 4h13.856c1.54 0 2.502-1.667 1.732-2.5L13.732 4c-.77-.833-1.964-.833-2.732 0L4.082 16.5c-.77.833.192 2.5 1.732 2.5z" />
              </svg>
              Report Chat to Admin
            </button>
          </div>
        </div>

        {/* Footer */}
        <div className="p-4 border-t border-gray-200">
          <button
            onClick={onClose}
            className="w-full px-4 py-2 bg-gray-100 text-gray-700 rounded-md hover:bg-gray-200 transition-colors"
          >
            Close
          </button>
        </div>
      </div>
    </div>
  );
}

export default SettingsModal;