// Local storage utilities for user role management

export const ROLE_STORAGE_KEY = 'ambot_user_role';

export type UserRole = 'current-student' | 'inquiring-student' | 'teacher' | 'school-admin';

/**
 * Get the user's role from localStorage
 * @returns The stored role or null if no role is stored
 */
export const getUserRole = (): UserRole | null => {
  try {
    const role = localStorage.getItem(ROLE_STORAGE_KEY);
    return role as UserRole;
  } catch (error) {
    console.error('Error reading role from localStorage:', error);
    return null;
  }
};

/**
 * Save the user's role to localStorage
 * @param role The role to save
 */
export const setUserRole = (role: UserRole): void => {
  try {
    localStorage.setItem(ROLE_STORAGE_KEY, role);
  } catch (error) {
    console.error('Error saving role to localStorage:', error);
  }
};

/**
 * Check if a user role is stored in localStorage
 * @returns true if a role exists, false otherwise
 */
export const hasUserRole = (): boolean => {
  return getUserRole() !== null;
};

/**
 * Clear the user's role from localStorage
 */
export const clearUserRole = (): void => {
  try {
    localStorage.removeItem(ROLE_STORAGE_KEY);
  } catch (error) {
    console.error('Error clearing role from localStorage:', error);
  }
};

/**
 * Get a human-readable display name for a role
 * @param role The role to get the display name for
 * @returns The display name for the role
 */
export const getRoleDisplayName = (role: UserRole): string => {
  const roleMap: Record<UserRole, string> = {
    'current-student': 'Current Student',
    'inquiring-student': 'Inquiring/Future Student',
    'teacher': 'Teacher',
    'school-admin': 'School Admin'
  };
  
  return roleMap[role] || role;
};