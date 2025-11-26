import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const AdminDashboard: React.FC = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [darkMode, setDarkMode] = useState(false);

  useEffect(() => {
    // Load dark mode setting from localStorage
    const storedDarkMode = localStorage.getItem('ambot_dark_mode');
    if (storedDarkMode !== null) {
      setDarkMode(JSON.parse(storedDarkMode));
    }
  }, []);

  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  const handleDarkModeToggle = () => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);
    localStorage.setItem('ambot_dark_mode', JSON.stringify(newDarkMode));
  };

  const handleLogout = async () => {
    await logout();
    navigate('/admin');
  };

  return (
    <div className={`h-screen ${darkMode ? 'bg-gray-900' : 'bg-gray-100'} flex items-center justify-center transition-colors duration-200`}>
      <div className={`w-full sm:w-[90%] md:w-[70%] lg:w-[50%] xl:w-[40%] mx-auto my-4 h-[95vh] ${darkMode ? 'bg-gray-800' : 'bg-gray-50'} rounded-lg flex flex-col shadow-lg transition-colors duration-200`}>
        
        {/* Header */}
        <div className={`flex items-center justify-between p-4 ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-200'} border-b rounded-t-lg transition-colors duration-200`}>
          <div className="flex items-center gap-3">
            <img src="/Ambot.png" alt="Ambot Img" className="w-10 h-10 rounded-lg"/>
            <div>
              <h1 className={`text-2xl font-semibold ${darkMode ? 'text-red-400' : 'text-red-900'}`}>Admin Dashboard</h1>
              <div className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-500'}`}>
                {user?.email}
              </div>
            </div>
          </div>
          <div className="flex items-center gap-2">
            {/* Dark Mode Toggle */}
            <button 
              onClick={handleDarkModeToggle}
              className={`p-2 ${darkMode ? 'hover:bg-gray-600' : 'hover:bg-gray-100'} rounded transition-colors duration-200`}
              title={darkMode ? 'Light Mode' : 'Dark Mode'}
            >
              {darkMode ? (
                <svg className="w-5 h-5 text-yellow-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 3v1m0 16v1m9-9h-1M4 12H3m15.364 6.364l-.707-.707M6.343 6.343l-.707-.707m12.728 0l-.707.707M6.343 17.657l-.707.707M16 12a4 4 0 11-8 0 4 4 0 018 0z" />
                </svg>
              ) : (
                <svg className="w-5 h-5 text-gray-600" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M20.354 15.354A9 9 0 018.646 3.646 9.003 9.003 0 0012 21a9.003 9.003 0 008.354-5.646z" />
                </svg>
              )}
            </button>
            {/* Logout Button */}
            <button 
              onClick={handleLogout}
              className={`px-4 py-2 ${darkMode ? 'bg-red-700 hover:bg-red-600' : 'bg-red-900 hover:bg-red-800'} text-white rounded-lg transition-colors duration-200 text-sm font-medium`}
            >
              Logout
            </button>
          </div>
        </div>

        {/* Main Content */}
        <div className="flex-1 overflow-y-auto p-6">
          <div className="grid gap-4">
            {/* Stats Cards */}
            <div className="grid grid-cols-1 md:grid-cols-3 gap-4">
              <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-white'} shadow transition-colors duration-200`}>
                <div className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Total Conversations</div>
                <div className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>--</div>
              </div>
              <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-white'} shadow transition-colors duration-200`}>
                <div className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Active Users</div>
                <div className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>--</div>
              </div>
              <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-white'} shadow transition-colors duration-200`}>
                <div className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Response Rate</div>
                <div className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>--</div>
              </div>
            </div>

            {/* Quick Actions */}
            <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-white'} shadow transition-colors duration-200`}>
              <h2 className={`text-lg font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>Quick Actions</h2>
              <div className="flex flex-wrap gap-2">
                <button className={`px-4 py-2 rounded-lg ${darkMode ? 'bg-gray-600 hover:bg-gray-500 text-gray-200' : 'bg-gray-100 hover:bg-gray-200 text-gray-700'} transition-colors duration-200`}>
                  View Logs
                </button>
                <button className={`px-4 py-2 rounded-lg ${darkMode ? 'bg-gray-600 hover:bg-gray-500 text-gray-200' : 'bg-gray-100 hover:bg-gray-200 text-gray-700'} transition-colors duration-200`}>
                  Manage Users
                </button>
                <button className={`px-4 py-2 rounded-lg ${darkMode ? 'bg-gray-600 hover:bg-gray-500 text-gray-200' : 'bg-gray-100 hover:bg-gray-200 text-gray-700'} transition-colors duration-200`}>
                  Settings
                </button>
              </div>
            </div>

            {/* Recent Activity */}
            <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-white'} shadow transition-colors duration-200`}>
              <h2 className={`text-lg font-semibold mb-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>Recent Activity</h2>
              <div className={`text-center py-8 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                <svg className="w-12 h-12 mx-auto mb-2 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 5H7a2 2 0 00-2 2v12a2 2 0 002 2h10a2 2 0 002-2V7a2 2 0 00-2-2h-2M9 5a2 2 0 002 2h2a2 2 0 002-2M9 5a2 2 0 012-2h2a2 2 0 012 2" />
                </svg>
                <p>No recent activity</p>
              </div>
            </div>
          </div>
        </div>

        {/* Footer */}
        <div className={`p-4 ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-200'} border-t rounded-b-lg transition-colors duration-200`}>
          <div className={`text-center text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            AmBot Admin Panel • Logged in as {user?.email}
          </div>
        </div>
      </div>
    </div>
  );
};

export default AdminDashboard;
