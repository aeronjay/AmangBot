import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../context/AuthContext';

const AdminLogin: React.FC = () => {
  const [email, setEmail] = useState('');
  const [password, setPassword] = useState('');
  const [error, setError] = useState('');
  const [isLoading, setIsLoading] = useState(false);
  const [darkMode, setDarkMode] = useState(false);
  
  const { login, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();

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

  // Redirect if already authenticated
  React.useEffect(() => {
    if (isAuthenticated) {
      const from = location.state?.from?.pathname || '/admindashboard';
      navigate(from, { replace: true });
    }
  }, [isAuthenticated, navigate, location]);

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    setError('');
    setIsLoading(true);

    try {
      await login(email, password);
      const from = location.state?.from?.pathname || '/admindashboard';
      navigate(from, { replace: true });
    } catch (err) {
      setError(err instanceof Error ? err.message : 'Login failed. Please try again.');
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className={`h-screen ${darkMode ? 'bg-gray-900' : 'bg-gray-100'} flex items-center justify-center transition-colors duration-200 px-4`}>
      <div className={`w-full sm:w-[90%] md:w-[450px] ${darkMode ? 'bg-gray-800' : 'bg-gray-50'} rounded-lg shadow-lg transition-colors duration-200`}>
        
        {/* Header */}
        <div className={`flex flex-col items-center p-6 ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-200'} border-b rounded-t-lg transition-colors duration-200`}>
          <img src="/Ambot.png" alt="Ambot Img" className="w-16 h-16 rounded-lg mb-3"/>
          <h1 className={`text-2xl font-semibold ${darkMode ? 'text-red-400' : 'text-red-900'}`}>Admin Login</h1>
          <p className={`text-sm ${darkMode ? 'text-gray-300' : 'text-gray-500'} mt-1`}>Sign in to access the dashboard</p>
        </div>

        {/* Login Form */}
        <div className="p-6">
          <form onSubmit={handleSubmit} className="space-y-5">
            {error && (
              <div className={`${darkMode ? 'bg-red-900/30 border-red-500' : 'bg-red-50 border-red-500'} border-l-4 p-4 rounded transition-colors duration-200`}>
                <div className="flex items-center">
                  <svg className="w-5 h-5 text-red-500 mr-2" fill="currentColor" viewBox="0 0 20 20">
                    <path fillRule="evenodd" d="M18 10a8 8 0 11-16 0 8 8 0 0116 0zm-7 4a1 1 0 11-2 0 1 1 0 012 0zm-1-9a1 1 0 00-1 1v4a1 1 0 102 0V6a1 1 0 00-1-1z" clipRule="evenodd" />
                  </svg>
                  <p className={`${darkMode ? 'text-red-300' : 'text-red-700'} text-sm`}>{error}</p>
                </div>
              </div>
            )}

            <div>
              <label htmlFor="email" className={`block text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'} mb-2`}>
                Email Address or Username
              </label>
              <input
                id="email"
                type="text"
                value={email}
                onChange={(e) => setEmail(e.target.value)}
                required
                className={`w-full px-4 py-3 border ${darkMode ? 'border-gray-500 bg-gray-600 text-gray-100 placeholder-gray-400 focus:ring-red-400' : 'border-gray-300 bg-white text-gray-900 placeholder-gray-500 focus:ring-red-900'} rounded-lg focus:outline-none focus:ring-2 focus:border-transparent transition-colors duration-200`}
                placeholder="admin@earist.edu.ph or username"
                disabled={isLoading}
              />
            </div>

            <div>
              <label htmlFor="password" className={`block text-sm font-medium ${darkMode ? 'text-gray-300' : 'text-gray-700'} mb-2`}>
                Password
              </label>
              <input
                id="password"
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                className={`w-full px-4 py-3 border ${darkMode ? 'border-gray-500 bg-gray-600 text-gray-100 placeholder-gray-400 focus:ring-red-400' : 'border-gray-300 bg-white text-gray-900 placeholder-gray-500 focus:ring-red-900'} rounded-lg focus:outline-none focus:ring-2 focus:border-transparent transition-colors duration-200`}
                placeholder="••••••••"
                disabled={isLoading}
              />
            </div>

            <button
              type="submit"
              disabled={isLoading}
              className={`w-full ${darkMode ? 'bg-red-700 hover:bg-red-600' : 'bg-red-900 hover:bg-red-800'} text-white font-semibold py-3 px-4 rounded-lg transition-colors duration-200 flex items-center justify-center disabled:opacity-50 disabled:cursor-not-allowed`}
            >
              {isLoading ? (
                <>
                  <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                    <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                    <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                  </svg>
                  Signing in...
                </>
              ) : (
                'Sign In'
              )}
            </button>
          </form>

          <div className="mt-6 text-center">
            <a href="/" className={`text-sm ${darkMode ? 'text-red-400 hover:text-red-300' : 'text-red-900 hover:text-red-700'} transition-colors duration-200`}>
              ← Back to Chat
            </a>
          </div>
        </div>

        {/* Footer */}
        <div className={`p-4 ${darkMode ? 'bg-gray-700 border-gray-600' : 'bg-white border-gray-200'} border-t rounded-b-lg transition-colors duration-200`}>
          <p className={`text-center text-sm ${darkMode ? 'text-gray-400' : 'text-gray-500'}`}>
            © 2025 EARIST AmangBot. All rights reserved.
          </p>
        </div>
      </div>
    </div>
  );
};

export default AdminLogin;
