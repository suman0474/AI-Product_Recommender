/**
 * ThreadContext.tsx
 *
 * React Context for managing thread and session state across the application
 * Provides hooks and utilities for all components to access current thread context
 */

import React, { createContext, useContext, useCallback, useEffect, useState } from 'react';
import { SessionManager, UserSession, SubThread, SessionMetadata } from '../services/SessionManager';

export interface ThreadContextType {
  // Session Management
  currentSession: UserSession | undefined;
  sessionMetadata: SessionMetadata | undefined;

  // Thread Management
  mainThreadId: string | undefined;
  activeSubThreadId: string | undefined;
  subThreads: Map<string, SubThread> | undefined;

  // Actions
  createSession: (userId: string, zone?: string) => UserSession | null;
  endSession: () => boolean;
  createSubThread: (
    workflowType: 'instrument_identifier' | 'solution' | 'product_search' | 'grounded_chat'
  ) => SubThread | null;
  addItemThread: (
    subThreadId: string,
    itemNumber: number,
    itemName: string,
    itemType: 'instrument' | 'accessory'
  ) => string | null;
  setActiveSubThread: (subThreadId: string) => boolean;
  closeSubThread: (subThreadId: string) => boolean;
  getItemThreads: (subThreadId: string) => Map<number, string> | null;
  restoreSession: (sessionId: string) => UserSession | null;
  updateWindowCount: (count: number) => void;

  // Utilities
  getAllActiveSessions: () => SessionMetadata[];
  clearAllSessions: () => void;
  isSessionActive: boolean;
}

const ThreadContext = createContext<ThreadContextType | undefined>(undefined);

export interface ThreadProviderProps {
  children: React.ReactNode;
  onSessionCreated?: (session: UserSession) => void;
  onSessionEnded?: (sessionId: string) => void;
  onSubThreadCreated?: (subThread: SubThread) => void;
}

/**
 * ThreadProvider Component
 * Wraps application and provides thread context to all components
 */
export const ThreadProvider: React.FC<ThreadProviderProps> = ({
  children,
  onSessionCreated,
  onSessionEnded,
  onSubThreadCreated,
}) => {
  const sessionManager = SessionManager.getInstance();
  const [currentSession, setCurrentSession] = useState<UserSession | undefined>();
  const [sessionMetadata, setSessionMetadata] = useState<SessionMetadata | undefined>();
  const [forceUpdate, setForceUpdate] = useState(0);

  // Sync state with session manager changes
  const updateState = useCallback(() => {
    const session = sessionManager.getCurrentSession();
    setCurrentSession(session);

    if (session) {
      const metadata = sessionManager.getSessionMetadata(session.sessionId);
      setSessionMetadata(metadata);
    } else {
      setSessionMetadata(undefined);
    }
  }, [sessionManager]);

  // Create session
  const handleCreateSession = useCallback(
    (userId: string, zone?: string): UserSession | null => {
      const session = sessionManager.createSession(userId, zone);
      updateState();

      if (session && onSessionCreated) {
        onSessionCreated(session);
      }

      return session;
    },
    [sessionManager, updateState, onSessionCreated]
  );

  // End session
  const handleEndSession = useCallback((): boolean => {
    const session = sessionManager.getCurrentSession();
    const sessionId = session?.sessionId;

    const success = sessionManager.endSession();

    if (success) {
      updateState();

      if (sessionId && onSessionEnded) {
        onSessionEnded(sessionId);
      }
    }

    return success;
  }, [sessionManager, updateState, onSessionEnded]);

  // Create sub-thread
  const handleCreateSubThread = useCallback(
    (
      workflowType: 'instrument_identifier' | 'solution' | 'product_search' | 'grounded_chat'
    ): SubThread | null => {
      const subThread = sessionManager.createSubThread(workflowType);
      updateState();

      if (subThread && onSubThreadCreated) {
        onSubThreadCreated(subThread);
      }

      return subThread;
    },
    [sessionManager, updateState, onSubThreadCreated]
  );

  // Add item thread
  const handleAddItemThread = useCallback(
    (
      subThreadId: string,
      itemNumber: number,
      itemName: string,
      itemType: 'instrument' | 'accessory'
    ): string | null => {
      const itemThreadId = sessionManager.addItemThreadToSubThread(
        subThreadId,
        itemNumber,
        itemName,
        itemType
      );
      updateState();
      return itemThreadId;
    },
    [sessionManager, updateState]
  );

  // Set active sub-thread
  const handleSetActiveSubThread = useCallback(
    (subThreadId: string): boolean => {
      const success = sessionManager.setActiveSubThread(subThreadId);
      updateState();
      return success;
    },
    [sessionManager, updateState]
  );

  // Close sub-thread
  const handleCloseSubThread = useCallback(
    (subThreadId: string): boolean => {
      const success = sessionManager.closeSubThread(subThreadId);
      updateState();
      return success;
    },
    [sessionManager, updateState]
  );

  // Get item threads
  const handleGetItemThreads = useCallback(
    (subThreadId: string): Map<number, string> | null => {
      return sessionManager.getItemThreadsInSubThread(subThreadId);
    },
    [sessionManager]
  );

  // Restore session
  const handleRestoreSession = useCallback(
    (sessionId: string): UserSession | null => {
      const session = sessionManager.restoreSession(sessionId);
      updateState();
      return session;
    },
    [sessionManager, updateState]
  );

  // Update window count
  const handleUpdateWindowCount = useCallback(
    (count: number): void => {
      sessionManager.updateWindowCount(count);
      updateState();
    },
    [sessionManager, updateState]
  );

  // Get all active sessions
  const handleGetAllActiveSessions = useCallback((): SessionMetadata[] => {
    return sessionManager.getAllActiveSessions();
  }, [sessionManager]);

  // Clear all sessions
  const handleClearAllSessions = useCallback((): void => {
    sessionManager.clearAllSessions();
    updateState();
  }, [sessionManager, updateState]);

  // Initialize on mount
  useEffect(() => {
    updateState();
  }, [updateState]);

  // Track active sub-threads for UI updates
  useEffect(() => {
    const interval = setInterval(() => {
      setForceUpdate(prev => prev + 1);
    }, 1000);

    return () => clearInterval(interval);
  }, []);

  const value: ThreadContextType = {
    // State
    currentSession,
    sessionMetadata,
    mainThreadId: currentSession?.mainThreadId,
    activeSubThreadId: currentSession?.activeSubThreadId,
    subThreads: currentSession?.subThreads,

    // Actions
    createSession: handleCreateSession,
    endSession: handleEndSession,
    createSubThread: handleCreateSubThread,
    addItemThread: handleAddItemThread,
    setActiveSubThread: handleSetActiveSubThread,
    closeSubThread: handleCloseSubThread,
    getItemThreads: handleGetItemThreads,
    restoreSession: handleRestoreSession,
    updateWindowCount: handleUpdateWindowCount,

    // Utilities
    getAllActiveSessions: handleGetAllActiveSessions,
    clearAllSessions: handleClearAllSessions,
    isSessionActive: !!currentSession,
  };

  return <ThreadContext.Provider value={value}>{children}</ThreadContext.Provider>;
};

/**
 * useThread Hook
 * Use this in any component to access thread context
 */
export const useThread = (): ThreadContextType => {
  const context = useContext(ThreadContext);

  if (context === undefined) {
    throw new Error('useThread must be used within a ThreadProvider');
  }

  return context;
};

/**
 * useSessionMetadata Hook
 * Get current session metadata
 */
export const useSessionMetadata = (): SessionMetadata | undefined => {
  const { sessionMetadata } = useThread();
  return sessionMetadata;
};

/**
 * useActiveSubThread Hook
 * Get currently active sub-thread
 */
export const useActiveSubThread = (): SubThread | undefined => {
  const { currentSession, activeSubThreadId } = useThread();

  if (!currentSession || !activeSubThreadId) {
    return undefined;
  }

  return currentSession.subThreads.get(activeSubThreadId);
};

/**
 * useThreadIds Hook
 * Get all relevant thread IDs for API calls
 */
export const useThreadIds = () => {
  const { mainThreadId, activeSubThreadId, currentSession } = useThread();

  return {
    mainThreadId,
    subThreadId: activeSubThreadId,
    sessionId: currentSession?.sessionId,
    zone: currentSession?.zone,
  };
};

export default ThreadContext;
