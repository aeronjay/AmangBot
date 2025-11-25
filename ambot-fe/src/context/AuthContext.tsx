import React, { createContext, useContext, useState, useEffect, type ReactNode } from 'react';
import authService from '../../service/authService';
import type { UserResponse } from '../../service/authService';

interface AuthContextType {
  user: UserResponse | null;
  isAuthenticated: boolean;
  isLoading: boolean;
  login: (email: string, password: string) => Promise<void>;
  logout: () => Promise<void>;
  checkAuth: () => Promise<boolean>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [user, setUser] = useState<UserResponse | null>(null);
  const [isLoading, setIsLoading] = useState(true);

  const checkAuth = async (): Promise<boolean> => {
    try {
      const tokenData = await authService.verifyToken();
      if (tokenData && tokenData.valid) {
        const userData = await authService.getCurrentUser();
        setUser(userData);
        return true;
      }
      setUser(null);
      return false;
    } catch {
      setUser(null);
      return false;
    }
  };

  useEffect(() => {
    const initAuth = async () => {
      setIsLoading(true);
      await checkAuth();
      setIsLoading(false);
    };
    initAuth();
  }, []);

  const login = async (email: string, password: string): Promise<void> => {
    await authService.login({ email, password });
    const userData = await authService.getCurrentUser();
    setUser(userData);
  };

  const logout = async (): Promise<void> => {
    await authService.logout();
    setUser(null);
  };

  const value: AuthContextType = {
    user,
    isAuthenticated: !!user,
    isLoading,
    login,
    logout,
    checkAuth,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};

export const useAuth = (): AuthContextType => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

export default AuthContext;
