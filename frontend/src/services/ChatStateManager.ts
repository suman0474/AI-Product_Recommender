/**
 * ChatStateManager.ts
 * 
 * Manages complete chat/conversation state persistence using IndexedDB
 * Ensures conversations continue seamlessly after page refresh
 */

import React from 'react';
import { useScreenPersistence } from '../hooks/use-screen-persistence';

/**
 * Complete Chat State Interface
 */
export interface ChatState {
    // Conversation thread data
    threadId: string;
    messages: Array<{
        id: string;
        role: 'user' | 'assistant' | 'system';
        content: string;
        timestamp: string;
        metadata?: Record<string, any>;
    }>;

    // Current workflow state
    currentWorkflow?: 'instrument_identifier' | 'solution' | 'product_search' | 'engenie_chat';
    currentStep?: string;

    // Data collected during conversation
    collectedData: {
        productType?: string;
        identifiedInstruments?: any[];
        identifiedAccessories?: any[];
        mandatoryRequirements?: Record<string, any>;
        optionalRequirements?: Record<string, any>;
        advancedParameters?: any[];
        analysisResults?: any;
        schema?: any;
    };

    // UI state
    isLoading?: boolean;
    inputValue?: string;
    selectedItems?: any[];

    // Metadata
    createdAt: string;
    lastUpdatedAt: string;
    sessionId: string;
}

/**
 * Hook for managing chat state with IndexedDB persistence
 */
export function useChatStatePersistence(tabId: string) {
    const stateRef = React.useRef<ChatState>({
        threadId: '',
        messages: [],
        collectedData: {},
        createdAt: new Date().toISOString(),
        lastUpdatedAt: new Date().toISOString(),
        sessionId: ''
    });

    const { saveState, loadState, clearState } = useScreenPersistence<ChatState>(
        stateRef,
        {
            dbName: 'AIProduct Recommender_ChatState',
            storeName: 'chatStates',
            key: `chat_${tabId}`,
            backupKey: `chatBackup_${tabId}`,
            enableAutoSave: true,
            autoSaveIntervalMs: 10000, // Save every 10 seconds

            // Transform for LocalStorage backup (save lighter version)
            transformForBackup: (state) => ({
                threadId: state.threadId,
                messagesCount: state.messages.length,
                lastMessage: state.messages[state.messages.length - 1],
                currentWorkflow: state.currentWorkflow,
                productType: state.collectedData.productType,
                sessionId: state.sessionId,
                lastUpdatedAt: state.lastUpdatedAt
            }),

            // Restore Date objects after loading
            onLoad: (state) => {
                return {
                    ...state,
                    messages: state.messages || [],
                    collectedData: state.collectedData || {},
                };
            }
        }
    );

    return {
        stateRef,
        saveState,
        loadState,
        clearState
    };
}

/**
 * Utility function to update chat state safely
 */
export function updateChatState(
    stateRef: React.MutableRefObject<ChatState>,
    updates: Partial<ChatState>
): void {
    stateRef.current = {
        ...stateRef.current,
        ...updates,
        lastUpdatedAt: new Date().toISOString()
    };
}

/**
 * Utility to add message to chat
 */
export function addMessage(
    stateRef: React.MutableRefObject<ChatState>,
    role: 'user' | 'assistant' | 'system',
    content: string,
    metadata?: Record<string, any>
): void {
    const message = {
        id: `msg_${Date.now()}_${Math.random().toString(36).substr(2, 9)}`,
        role,
        content,
        timestamp: new Date().toISOString(),
        metadata
    };

    stateRef.current = {
        ...stateRef.current,
        messages: [...stateRef.current.messages, message],
        lastUpdatedAt: new Date().toISOString()
    };
}

/**
 * Utility to update collected data
 */
export function updateCollectedData(
    stateRef: React.MutableRefObject<ChatState>,
    data: Partial<ChatState['collectedData']>
): void {
    stateRef.current = {
        ...stateRef.current,
        collectedData: {
            ...stateRef.current.collectedData,
            ...data
        },
        lastUpdatedAt: new Date().toISOString()
    };
}

export default {
    useChatStatePersistence,
    updateChatState,
    addMessage,
    updateCollectedData
};
