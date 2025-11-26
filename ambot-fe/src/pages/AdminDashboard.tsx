import React, { useState, useEffect } from 'react';
import { useNavigate } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

interface InfoFile {
  id: number;
  name: string;
  source: string;
  size: string;
  lastModified: string;
  status: string;
}

const AdminDashboard: React.FC = () => {
  const { user, logout } = useAuth();
  const navigate = useNavigate();
  const [darkMode, setDarkMode] = useState(false);
  const [infoFiles, setInfoFiles] = useState<InfoFile[]>([]);
  const [showAddModal, setShowAddModal] = useState(false);
  const [newInfoName, setNewInfoName] = useState('');
  const [newInfoContent, setNewInfoContent] = useState('');
  const [selectedInfo, setSelectedInfo] = useState<InfoFile | null>(null);
  const [showViewModal, setShowViewModal] = useState(false);

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

  const handleAddInfo = () => {
    if (newInfoName.trim()) {
      const newInfo: InfoFile = {
        id: Date.now(),
        name: newInfoName.endsWith('.txt') ? newInfoName : `${newInfoName}.txt`,
        source: 'AdminDB',
        size: `${Math.round(newInfoContent.length / 1024) || 1} KB`,
        lastModified: new Date().toISOString().split('T')[0],
        status: 'pending',
      };
      setInfoFiles([...infoFiles, newInfo]);
      setNewInfoName('');
      setNewInfoContent('');
      setShowAddModal(false);
    }
  };

  const handleDeleteInfo = (id: number) => {
    setInfoFiles(infoFiles.filter(info => info.id !== id));
  };

  const handleViewInfo = (info: InfoFile) => {
    setSelectedInfo(info);
    setShowViewModal(true);
  };

  return (
    <div className={`h-screen ${darkMode ? 'bg-gray-900' : 'bg-gray-100'} flex items-center justify-center transition-colors duration-200`}>
      <div className={`w-full sm:w-[95%] md:w-[85%] lg:w-[75%] xl:w-[65%] mx-auto my-4 h-[95vh] ${darkMode ? 'bg-gray-800' : 'bg-gray-50'} rounded-lg flex flex-col shadow-lg transition-colors duration-200`}>
        
        {/* Header */}
        <div className={`flex items-center justify-between p-4 ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-200'} border-b rounded-t-lg transition-colors duration-200`}>
            <div className="flex items-center gap-3">
            <img src="/Ambot.png" alt="Ambot Img" className="w-10 h-10 rounded-lg"/>
            <div>
              <h1 className={`text-2xl font-semibold ${darkMode ? 'text-red-400' : 'text-red-900'}`}>Info Manager</h1>
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
            {/* Stats Card */}
            <div className="grid grid-cols-1 gap-4">
              <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-white'} shadow transition-colors duration-200`}>
                <div className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Total Information Files</div>
                <div className={`text-2xl font-bold ${darkMode ? 'text-white' : 'text-gray-900'}`}>{infoFiles.length}</div>
              </div>
            </div>

            {/* Info Table */}
            <div className={`p-4 rounded-lg ${darkMode ? 'bg-gray-700' : 'bg-white'} shadow transition-colors duration-200`}>
              <div className="flex items-center justify-between mb-4">
                <h2 className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>Information Files</h2>
                <button
                  // onClick={() => setShowAddModal(true)}
                  className={`px-4 py-2 rounded-lg font-medium transition-colors duration-200 ${
                    darkMode ? 'bg-green-700 hover:bg-green-600 text-white' : 'bg-green-600 hover:bg-green-700 text-white'
                  }`}
                >
                  <div className="flex items-center gap-2">
                    <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                      <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 4v16m8-8H4" />
                    </svg>
                    Add Info
                  </div>
                </button>
              </div>

              {/* Info Table */}
              <div className="overflow-x-auto">
                <table className="w-full">
                  <thead>
                    <tr className={`${darkMode ? 'border-gray-600' : 'border-gray-200'} border-b`}>
                      <th className={`text-left py-3 px-4 font-medium ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>Name</th>
                      <th className={`text-left py-3 px-4 font-medium ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>Size</th>
                      <th className={`text-left py-3 px-4 font-medium ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>Last Modified</th>
                      <th className={`text-left py-3 px-4 font-medium ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>Status</th>
                      <th className={`text-left py-3 px-4 font-medium ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>Actions</th>
                    </tr>
                  </thead>
                  <tbody>
                    {infoFiles.length === 0 ? (
                      <tr>
                        <td colSpan={5} className={`text-center py-8 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
                          <svg className="w-12 h-12 mx-auto mb-2 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 13h6m-3-3v6m5 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                          </svg>
                          <p>No information files found</p>
                          <p className="text-sm mt-1">Click "Add Info" to add a new information file</p>
                        </td>
                      </tr>
                    ) : (
                      infoFiles.map((info) => (
                        <tr key={info.id} className={`${darkMode ? 'border-gray-600 hover:bg-gray-600' : 'border-gray-200 hover:bg-gray-50'} border-b transition-colors duration-200`}>
                          <td className={`py-3 px-4 ${darkMode ? 'text-white' : 'text-gray-900'}`}>
                            <div className="flex items-center gap-2">
                              <svg className={`w-4 h-4 ${darkMode ? 'text-gray-400' : 'text-gray-500'}`} fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M9 12h6m-6 4h6m2 5H7a2 2 0 01-2-2V5a2 2 0 012-2h5.586a1 1 0 01.707.293l5.414 5.414a1 1 0 01.293.707V19a2 2 0 01-2 2z" />
                              </svg>
                              {info.name}
                            </div>
                          </td>
                          <td className={`py-3 px-4 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>{info.size}</td>
                          <td className={`py-3 px-4 ${darkMode ? 'text-gray-300' : 'text-gray-600'}`}>{info.lastModified}</td>
                          <td className={`py-3 px-4`}>
                            <span className={`px-2 py-1 rounded text-xs font-medium ${
                              info.status === 'active' 
                                ? darkMode ? 'bg-green-900 text-green-200' : 'bg-green-100 text-green-800'
                                : darkMode ? 'bg-yellow-900 text-yellow-200' : 'bg-yellow-100 text-yellow-800'
                            }`}>
                              {info.status}
                            </span>
                          </td>
                          <td className={`py-3 px-4`}>
                            <div className="flex items-center gap-2">
                              <button
                                onClick={() => handleViewInfo(info)}
                                className={`p-1.5 rounded transition-colors duration-200 ${darkMode ? 'hover:bg-gray-500 text-gray-300' : 'hover:bg-gray-200 text-gray-600'}`}
                                title="View"
                              >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 12a3 3 0 11-6 0 3 3 0 016 0z" />
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M2.458 12C3.732 7.943 7.523 5 12 5c4.478 0 8.268 2.943 9.542 7-1.274 4.057-5.064 7-9.542 7-4.477 0-8.268-2.943-9.542-7z" />
                                </svg>
                              </button>
                              <button
                                onClick={() => handleDeleteInfo(info.id)}
                                className={`p-1.5 rounded transition-colors duration-200 ${darkMode ? 'hover:bg-red-900 text-red-400' : 'hover:bg-red-100 text-red-600'}`}
                                title="Delete"
                              >
                                <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M19 7l-.867 12.142A2 2 0 0116.138 21H7.862a2 2 0 01-1.995-1.858L5 7m5 4v6m4-6v6m1-10V4a1 1 0 00-1-1h-4a1 1 0 00-1 1v3M4 7h16" />
                                </svg>
                              </button>
                            </div>
                          </td>
                        </tr>
                      ))
                    )}
                  </tbody>
                </table>
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

      {/* Add Info Modal */}
      {showAddModal && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className={`w-full max-w-lg mx-4 rounded-lg shadow-xl ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
            <div className={`flex items-center justify-between p-4 border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
              <h2 className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>Add New Information</h2>
              <button
                onClick={() => setShowAddModal(false)}
                className={`p-1 rounded ${darkMode ? 'hover:bg-gray-700 text-gray-400' : 'hover:bg-gray-100 text-gray-500'}`}
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="p-4">
              <div className="mb-4">
                <label className={`block text-sm font-medium mb-2 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                  File Name
                </label>
                <input
                  type="text"
                  value={newInfoName}
                  onChange={(e) => setNewInfoName(e.target.value)}
                  placeholder="e.g., NewInfo.txt"
                  className={`w-full px-3 py-2 rounded-lg border ${
                    darkMode 
                      ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400' 
                      : 'bg-white border-gray-300 text-gray-900 placeholder-gray-400'
                  } focus:outline-none focus:ring-2 focus:ring-red-500`}
                />
              </div>
              <div className="mb-4">
                <label className={`block text-sm font-medium mb-2 ${darkMode ? 'text-gray-300' : 'text-gray-700'}`}>
                  Content
                </label>
                <textarea
                  value={newInfoContent}
                  onChange={(e) => setNewInfoContent(e.target.value)}
                  placeholder="Enter the information content here..."
                  rows={8}
                  className={`w-full px-3 py-2 rounded-lg border ${
                    darkMode 
                      ? 'bg-gray-700 border-gray-600 text-white placeholder-gray-400' 
                      : 'bg-white border-gray-300 text-gray-900 placeholder-gray-400'
                  } focus:outline-none focus:ring-2 focus:ring-red-500 resize-none`}
                />
              </div>
              <div className={`text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'} mb-4`}>
                <svg className="w-4 h-4 inline mr-1" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M13 16h-1v-4h-1m1-4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                This file will be saved as a .txt file
              </div>
            </div>
            <div className={`flex justify-end gap-2 p-4 border-t ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
              <button
                onClick={() => setShowAddModal(false)}
                className={`px-4 py-2 rounded-lg ${darkMode ? 'bg-gray-600 hover:bg-gray-500 text-gray-200' : 'bg-gray-100 hover:bg-gray-200 text-gray-700'} transition-colors duration-200`}
              >
                Cancel
              </button>
              <button
                onClick={handleAddInfo}
                disabled={!newInfoName.trim()}
                className={`px-4 py-2 rounded-lg ${
                  newInfoName.trim()
                    ? darkMode ? 'bg-green-700 hover:bg-green-600 text-white' : 'bg-green-600 hover:bg-green-700 text-white'
                    : darkMode ? 'bg-gray-600 text-gray-400 cursor-not-allowed' : 'bg-gray-200 text-gray-400 cursor-not-allowed'
                } transition-colors duration-200`}
              >
                Add Info
              </button>
            </div>
          </div>
        </div>
      )}

      {/* View Info Modal */}
      {showViewModal && selectedInfo && (
        <div className="fixed inset-0 bg-black bg-opacity-50 flex items-center justify-center z-50">
          <div className={`w-full max-w-lg mx-4 rounded-lg shadow-xl ${darkMode ? 'bg-gray-800' : 'bg-white'}`}>
            <div className={`flex items-center justify-between p-4 border-b ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
              <h2 className={`text-lg font-semibold ${darkMode ? 'text-white' : 'text-gray-900'}`}>Information Details</h2>
              <button
                onClick={() => setShowViewModal(false)}
                className={`p-1 rounded ${darkMode ? 'hover:bg-gray-700 text-gray-400' : 'hover:bg-gray-100 text-gray-500'}`}
              >
                <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                  <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M6 18L18 6M6 6l12 12" />
                </svg>
              </button>
            </div>
            <div className="p-4">
              <div className="space-y-3">
                <div>
                  <span className={`text-sm font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Name:</span>
                  <p className={`${darkMode ? 'text-white' : 'text-gray-900'}`}>{selectedInfo.name}</p>
                </div>
                <div>
                  <span className={`text-sm font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Size:</span>
                  <p className={`${darkMode ? 'text-white' : 'text-gray-900'}`}>{selectedInfo.size}</p>
                </div>
                <div>
                  <span className={`text-sm font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Last Modified:</span>
                  <p className={`${darkMode ? 'text-white' : 'text-gray-900'}`}>{selectedInfo.lastModified}</p>
                </div>
                <div>
                  <span className={`text-sm font-medium ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>Status:</span>
                  <p>
                    <span className={`px-2 py-1 rounded text-xs font-medium ${
                      selectedInfo.status === 'active' 
                        ? darkMode ? 'bg-green-900 text-green-200' : 'bg-green-100 text-green-800'
                        : darkMode ? 'bg-yellow-900 text-yellow-200' : 'bg-yellow-100 text-yellow-800'
                    }`}>
                      {selectedInfo.status}
                    </span>
                  </p>
                </div>
              </div>
            </div>
            <div className={`flex justify-end p-4 border-t ${darkMode ? 'border-gray-700' : 'border-gray-200'}`}>
              <button
                onClick={() => setShowViewModal(false)}
                className={`px-4 py-2 rounded-lg ${darkMode ? 'bg-gray-600 hover:bg-gray-500 text-gray-200' : 'bg-gray-100 hover:bg-gray-200 text-gray-700'} transition-colors duration-200`}
              >
                Close
              </button>
            </div>
          </div>
        </div>
      )}
    </div>
  );
};

export default AdminDashboard;
