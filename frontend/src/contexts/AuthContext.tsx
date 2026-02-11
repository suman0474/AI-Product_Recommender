import React, { createContext, useContext, useState, useEffect, ReactNode } from 'react';
import { checkAuth, login as apiLogin, logout as apiLogout, signup as apiSignup } from '../components/AIRecommender/api';
import { UserCredentials } from '../components/AIRecommender/types';
import { useToast } from '../hooks/use-toast';
import { getSessionManager } from '../services/SessionManager';

// UPDATED User interface to include role and status
interface User {
  username: string;
  name: string;
  firstName?: string;
  lastName?: string;
  email: string;
  role: "admin" | "user";        // change to union type for safety
  status: "pending" | "active" | "rejected";  // 'pending', 'active', or 'rejected'
  companyName?: string;
  location?: string;
  strategyInterest?: string;
  documentFileId?: string;
}

interface AuthContextType {
  isAuthenticated: boolean;
  isLoading: boolean;
  user: User | null;
  isAdmin: boolean;
  userSessionId: string | null;  // NEW: Unified session ID for entire login session
  refreshUser: () => Promise<void>;
  login: (credentials: Omit<UserCredentials, 'email'>) => Promise<void>;
  signup: (credentials: UserCredentials) => Promise<void>;
  logout: () => Promise<void>;
}

const AuthContext = createContext<AuthContextType | undefined>(undefined);

// Storage key for user session ID (persists across page refreshes)
const USER_SESSION_ID_KEY = 'user_session_id';

// Helper function to generate a new user session ID
const generateUserSessionId = (): string => {
  return `session_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`;
};

export const useAuth = () => {
  const context = useContext(AuthContext);
  if (context === undefined) {
    throw new Error('useAuth must be used within an AuthProvider');
  }
  return context;
};

interface AuthProviderProps {
  children: ReactNode;
}

export const AuthProvider: React.FC<AuthProviderProps> = ({ children }) => {
  const [isAuthenticated, setIsAuthenticated] = useState(false);
  const [isLoading, setIsLoading] = useState(true);
  const [user, setUser] = useState<User | null>(null);
  // NEW: User session ID - same across all screens for entire login session
  const [userSessionId, setUserSessionId] = useState<string | null>(() => {
    // Try to restore from localStorage on mount
    return localStorage.getItem(USER_SESSION_ID_KEY);
  });
  const { toast } = useToast();

  const checkAuthStatus = async () => {
    try {
      const authData = await checkAuth();
      setIsAuthenticated(!!authData);
      if (authData) {
        // Type cast to the new User interface
        setUser(authData.user as User);

        // Ensure user session ID exists if authenticated
        if (!localStorage.getItem(USER_SESSION_ID_KEY)) {
          const newSessionId = generateUserSessionId();
          localStorage.setItem(USER_SESSION_ID_KEY, newSessionId);
          setUserSessionId(newSessionId);
          console.log('[AUTH] Generated new user session ID on auth check:', newSessionId);
        }

        // Ensure thread session exists for authenticated user
        const sessionManager = getSessionManager();
        if (!sessionManager.getCurrentSession()) {
          await sessionManager.getOrCreateSession(authData.user.username || authData.user.email);
          console.log('[AUTH] Thread session restored for authenticated user');
        }
      } else {
        // Not authenticated - clear session ID
        localStorage.removeItem(USER_SESSION_ID_KEY);
        setUserSessionId(null);
      }
    } catch (error) {
      setIsAuthenticated(false);
      setUser(null);
      localStorage.removeItem(USER_SESSION_ID_KEY);
      setUserSessionId(null);
    } finally {
      setIsLoading(false);
    }
  };

  useEffect(() => {
    checkAuthStatus();
  }, []);

  const refreshUser = async () => {
    await checkAuthStatus();
  };

  const login = async (credentials: Omit<UserCredentials, 'email'>) => {
    try {
      setIsLoading(true);
      const response = await apiLogin(credentials);

      // CLEAR ALL OLD STATE to ensure fresh session on every login
      // Clear localStorage
      localStorage.clear();

      // Clear sessionStorage
      sessionStorage.clear();

      // Clear IndexedDB project state
      try {
        const DBDeleteRequest = window.indexedDB.deleteDatabase("project_page_db");
        DBDeleteRequest.onerror = () => {
          console.error("Error deleting project database on login.");
        };
        DBDeleteRequest.onsuccess = () => {
          console.log("Project database deleted for fresh start on login");
        };
      } catch (e) {
        console.error("Failed to delete project IndexedDB on login", e);
      }

      // Clear chat database
      try {
        const chatDBDeleteRequest = window.indexedDB.deleteDatabase("chat_db");
        chatDBDeleteRequest.onerror = () => {
          console.error("Error deleting chat database on login.");
        };
        chatDBDeleteRequest.onsuccess = () => {
          console.log("Chat database deleted for fresh start on login");
        };
      } catch (e) {
        console.error("Failed to delete chat IndexedDB on login", e);
      }

      // NEW: Generate and store user session ID for the entire login session
      const newUserSessionId = generateUserSessionId();
      localStorage.setItem(USER_SESSION_ID_KEY, newUserSessionId);
      setUserSessionId(newUserSessionId);
      console.log('[AUTH] Generated new user session ID on login:', newUserSessionId);

      setIsAuthenticated(true);
      // Set the user with the new role and status
      setUser(response.user as User);

      // Create thread session for UI-managed thread system
      const sessionManager = getSessionManager();
      await sessionManager.getOrCreateSession(response.user.username || response.user.email);
      console.log('[AUTH] Thread session created for user:', response.user.username);

      toast({
        title: "Success",
        description: "Successfully logged in!",
      });
    } catch (error: any) {
      const errorMessage = error.response?.data?.error || error.message || "Invalid credentials";
      let description = errorMessage;
      if (error.response?.status === 403) {
        // Status 403 means user is not active (pending or rejected)
        if (errorMessage.toLowerCase().includes("pending")) {
          description = "Your account is pending admin approval.";
        } else if (errorMessage.toLowerCase().includes("rejected")) {
          description = "Your account has been rejected. Please contact support.";
        }
      }
      toast({
        title: "Login Failed",
        description,
        variant: "destructive",
      });
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const signup = async (credentials: UserCredentials) => {
    try {
      setIsLoading(true);
      const response = await apiSignup(credentials);
      // The signup message now indicates admin approval is needed
      toast({
        title: "Success",
        description: response.message || "Account created successfully! Awaiting admin approval.",
      });
    } catch (error: any) {
      toast({
        title: "Signup Failed",
        description: error.message || "Failed to create account",
        variant: "destructive",
      });
      throw error;
    } finally {
      setIsLoading(false);
    }
  };

  const logout = async () => {
    try {
      setIsLoading(true);
      await apiLogout();

      // Clear all local state
      setIsAuthenticated(false);
      setUser(null);

      // NEW: Clear user session ID on logout
      localStorage.removeItem(USER_SESSION_ID_KEY);
      setUserSessionId(null);
      console.log('[AUTH] Cleared user session ID on logout');

      // CLEAR ALL LOCAL STORAGE to ensure fresh session on next login
      localStorage.clear();

      // CLEAR ALL SESSION STORAGE (including lastSourcePage, etc.)
      sessionStorage.clear();

      // Clear IndexedDB project state
      try {
        const DBDeleteRequest = window.indexedDB.deleteDatabase("project_page_db");
        DBDeleteRequest.onerror = () => {
          console.error("Error deleting project database on logout.");
        };
        DBDeleteRequest.onsuccess = () => {
          console.log("Project database deleted successfully on logout");
        };
      } catch (e) {
        console.error("Failed to delete project IndexedDB on logout", e);
      }

      // Clear chat database
      try {
        const chatDBDeleteRequest = window.indexedDB.deleteDatabase("chat_db");
        chatDBDeleteRequest.onerror = () => {
          console.error("Error deleting chat database on logout.");
        };
        chatDBDeleteRequest.onsuccess = () => {
          console.log("Chat database deleted successfully on logout");
        };
      } catch (e) {
        console.error("Failed to delete chat IndexedDB on logout", e);
      }

      // End thread session (clears if not saved)
      const sessionManager = getSessionManager();
      sessionManager.endSession();
      console.log('[AUTH] Thread session ended');

      toast({
        title: "Success",
        description: "Successfully logged out!",
      });
    } catch (error: any) {
      toast({
        title: "Logout Failed",
        description: error.message || "Failed to logout",
        variant: "destructive",
      });
    } finally {
      setIsLoading(false);
    }
  };

  const value = {
    isAuthenticated,
    isLoading,
    user,
    isAdmin: user?.role === "admin",   // added helper
    userSessionId,  // NEW: Expose user session ID to all components
    refreshUser,
    login,
    signup,
    logout,
  };

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
};
