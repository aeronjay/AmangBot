
import { useState, useEffect } from 'react';
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom';
import './index.css';
import ChatInterface from '../components/chatInterface';
import RoleSelectionModal from '../components/RoleSelectionModal';
import SettingsModal from '../components/SettingsModal';
import { hasUserRole, setUserRole, getUserRole, type UserRole } from './utils/roleStorage';
import { AuthProvider } from './context/AuthContext';
import ProtectedRoute from './components/ProtectedRoute';
import AdminLogin from './pages/AdminLogin';
import AdminDashboard from './pages/AdminDashboard';
import AmbotScene from './components/AmbotModel';

function ChatPage() {
  const [showRoleModal, setShowRoleModal] = useState(false);
  const [showSettingsModal, setShowSettingsModal] = useState(false);
  const [userRole, setCurrentUserRole] = useState<UserRole | null>(null);
  const [darkMode, setDarkMode] = useState(false);
  const [fontSize, setFontSize] = useState<'Small' | 'Medium' | 'Large'>('Medium');
  const [botState, setBotState] = useState<string>('idle');

  useEffect(() => {
    // Check if user has a role stored in localStorage
    if (!hasUserRole()) {
      setShowRoleModal(true);
    } else {
      const storedRole = getUserRole();
      setCurrentUserRole(storedRole);
    }

    // Load settings from localStorage
    const storedDarkMode = localStorage.getItem('ambot_dark_mode');
    if (storedDarkMode !== null) {
      setDarkMode(JSON.parse(storedDarkMode));
    }

    const storedFontSize = localStorage.getItem('ambot_font_size');
    if (storedFontSize) {
      setFontSize(storedFontSize as 'Small' | 'Medium' | 'Large');
    }
  }, []);

  // Apply dark mode class to document element
  useEffect(() => {
    if (darkMode) {
      document.documentElement.classList.add('dark');
    } else {
      document.documentElement.classList.remove('dark');
    }
  }, [darkMode]);

  const handleRoleSelect = (role: string) => {
    const selectedRole = role as UserRole;
    setUserRole(selectedRole);
    setCurrentUserRole(selectedRole);
    setShowRoleModal(false);
  };

  const handleCloseModal = () => {
    setShowRoleModal(false);
  };

  const handleOpenSettings = () => {
    setShowSettingsModal(true);
  };

  const handleCloseSettings = () => {
    setShowSettingsModal(false);
  };

  const handleSettingsRoleChange = (role: UserRole) => {
    setUserRole(role);
    setCurrentUserRole(role);
  };

  const handleDarkModeToggle = () => {
    const newDarkMode = !darkMode;
    setDarkMode(newDarkMode);
    localStorage.setItem('ambot_dark_mode', JSON.stringify(newDarkMode));
  };

  const handleFontSizeChange = (size: 'Small' | 'Medium' | 'Large') => {
    setFontSize(size);
    localStorage.setItem('ambot_font_size', size);
  };

  const getFontSizeClass = () => {
    switch (fontSize) {
      case 'Small': return 'font-size-small';
      case 'Large': return 'font-size-large';
      default: return 'font-size-medium';
    }
  };

  return (
    <div className={getFontSizeClass()}>
      <div className={`h-screen ${darkMode ? 'bg-gray-900' : 'bg-gray-100'} flex items-center justify-center transition-colors duration-200 overflow-hidden`}>
        <div className="flex w-full max-w-[1800px] h-[95vh] gap-6 px-4 justify-center">
          <div className="flex-1 h-full flex justify-center max-w-5xl">
            <ChatInterface 
              userRole={userRole} 
              onOpenSettings={handleOpenSettings}
              darkMode={darkMode}
              className="w-full h-full"
              onBotStateChange={setBotState}
            />
          </div>
          <div className="hidden xl:flex w-[500px] h-full items-center justify-center bg-transparent">
            <AmbotScene botState={botState} />
          </div>
        </div>
      </div>
      
      <RoleSelectionModal 
        isOpen={showRoleModal} 
        onRoleSelect={handleRoleSelect}
        onClose={handleCloseModal}
      />

      <SettingsModal
        isOpen={showSettingsModal}
        onClose={handleCloseSettings}
        userRole={userRole}
        onRoleChange={handleSettingsRoleChange}
        darkMode={darkMode}
        onDarkModeToggle={handleDarkModeToggle}
        fontSize={fontSize}
        onFontSizeChange={handleFontSizeChange}
      />
    </div>
  );
}

function App() {
  return (
    <AuthProvider>
      <Router>
        <Routes>
          <Route path="/" element={<ChatPage />} />
          <Route path="/admin" element={<AdminLogin />} />
          <Route
            path="/admindashboard"
            element={
              <ProtectedRoute>
                <AdminDashboard />
              </ProtectedRoute>
            }
          />
        </Routes>
      </Router>
    </AuthProvider>
  );
}

export default App;
