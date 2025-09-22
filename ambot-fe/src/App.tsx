
import { useState, useEffect } from 'react';
import './index.css';
import ChatInterface from '../components/chatInterface';
import RoleSelectionModal from '../components/RoleSelectionModal';
import { hasUserRole, setUserRole, getUserRole, type UserRole } from './utils/roleStorage';

function App() {
  const [showRoleModal, setShowRoleModal] = useState(false);
  const [userRole, setCurrentUserRole] = useState<UserRole | null>(null);

  useEffect(() => {
    // Check if user has a role stored in localStorage
    if (!hasUserRole()) {
      setShowRoleModal(true);
    } else {
      const storedRole = getUserRole();
      setCurrentUserRole(storedRole);
    }
  }, []);

  const handleRoleSelect = (role: string) => {
    const selectedRole = role as UserRole;
    setUserRole(selectedRole);
    setCurrentUserRole(selectedRole);
    setShowRoleModal(false);
  };

  const handleRoleChange = () => {
    setShowRoleModal(true);
  };

  const handleCloseModal = () => {
    setShowRoleModal(false);
  };

  return (
    <>
      <div className='h-screen bg-gray-100 flex items-center justify-center'>
        <ChatInterface userRole={userRole} onRoleChange={handleRoleChange} />
      </div>
      
      <RoleSelectionModal 
        isOpen={showRoleModal} 
        onRoleSelect={handleRoleSelect}
        onClose={handleCloseModal}
      />
    </>
  );
}

export default App;
