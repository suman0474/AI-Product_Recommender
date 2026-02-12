
import { useRef, useEffect, useCallback } from 'react';

// Define the hook's configuration interface
interface UseScreenPersistenceOptions<T> {
    dbName: string;
    storeName: string;
    key: string;
    backupKey: string;
    enableAutoSave?: boolean;
    autoSaveIntervalMs?: number;
    // Optional transformer for the fail-safe backup (e.g. to save a lighter version to LocalStorage)
    transformForBackup?: (state: T) => any;
    // Optional transformer after loading (e.g. to restore Date objects)
    onLoad?: (state: any) => T;
}

export function useScreenPersistence<T>(
    stateRef: React.MutableRefObject<T>,
    options: UseScreenPersistenceOptions<T>
) {
    const {
        dbName,
        storeName,
        key,
        backupKey,
        enableAutoSave = true,
        autoSaveIntervalMs = 30000,
        transformForBackup,
        onLoad
    } = options;

    // Open IndexedDB
    const openDB = useCallback((): Promise<IDBDatabase> => {
        return new Promise((resolve, reject) => {
            const request = indexedDB.open(dbName, 1);
            request.onerror = () => reject(request.error);
            request.onsuccess = () => resolve(request.result);
            request.onupgradeneeded = (event) => {
                const db = (event.target as IDBOpenDBRequest).result;
                if (!db.objectStoreNames.contains(storeName)) {
                    db.createObjectStore(storeName, { keyPath: 'id' });
                }
            };
        });
    }, [dbName, storeName]);

    // Save state to IndexedDB with LocalStorage backup
    const saveState = useCallback(async () => {
        const currentState = stateRef.current;
        if (!currentState) return;

        // 1. Save to IndexedDB (Async, Full State)
        try {
            const db = await openDB();
            const transaction = db.transaction(storeName, 'readwrite');
            const store = transaction.objectStore(storeName);

            await new Promise<void>((resolve, reject) => {
                const request = store.put({ id: key, ...currentState });
                request.onsuccess = () => resolve();
                request.onerror = () => reject(request.error);
            });
            db.close();
        } catch (e) {
            console.warn(`[PERSISTENCE] Failed to save to IndexedDB (${dbName}):`, e);
        }

        // 2. Save to LocalStorage (Sync, Backup/Light State)
        try {
            const stateToBackup = transformForBackup
                ? transformForBackup(currentState)
                : currentState;

            localStorage.setItem(backupKey, JSON.stringify({
                ...stateToBackup,
                _savedAt: new Date().toISOString()
            }));
        } catch (e) {
            console.warn(`[PERSISTENCE] LocalStorage backup failed (${backupKey}):`, e);
        }
    }, [openDB, storeName, key, backupKey, transformForBackup, stateRef]);

    // Load state from IndexedDB
    const loadState = useCallback(async (): Promise<T | null> => {
        try {
            const db = await openDB();
            const transaction = db.transaction(storeName, 'readonly');
            const store = transaction.objectStore(storeName);

            const result: any = await new Promise((resolve, reject) => {
                const request = store.get(key);
                request.onsuccess = () => resolve(request.result);
                request.onerror = () => reject(request.error);
            });

            db.close();

            // Return processed result if found
            if (result) {
                // Remove the 'id' key that we added for IndexedDB
                const { id, ...data } = result;
                return onLoad ? onLoad(data) : data as T;
            }

            // Fallback: Check LocalStorage if IndexedDB is empty (e.g. after clearing browser data but not local storage)
            const backup = localStorage.getItem(backupKey);
            if (backup) {
                console.log('[PERSISTENCE] Recovering from LocalStorage backup');
                const parsed = JSON.parse(backup);
                return onLoad ? onLoad(parsed) : parsed as T;
            }

            return null;
        } catch (e) {
            console.warn(`[PERSISTENCE] Failed to load from IndexedDB (${dbName}):`, e);
            return null;
        }
    }, [openDB, storeName, key, backupKey, onLoad]);

    // Clear state
    const clearState = useCallback(async () => {
        try {
            const db = await openDB();
            const transaction = db.transaction(storeName, 'readwrite');
            const store = transaction.objectStore(storeName);

            await new Promise<void>((resolve, reject) => {
                const request = store.delete(key);
                request.onsuccess = () => resolve();
                request.onerror = () => reject(request.error);
            });
            db.close();

            localStorage.removeItem(backupKey);
        } catch (e) {
            console.warn(`[PERSISTENCE] Failed to clear IndexedDB (${dbName}):`, e);
        }
    }, [openDB, storeName, key, backupKey]);

    // Setup Auto-save and BeforeUnload listeners
    useEffect(() => {
        const handleBeforeUnload = () => {
            saveState();
        };

        window.addEventListener('beforeunload', handleBeforeUnload);

        let intervalId: NodeJS.Timeout | null = null;
        if (enableAutoSave) {
            intervalId = setInterval(() => {
                saveState();
            }, autoSaveIntervalMs);
        }

        return () => {
            window.removeEventListener('beforeunload', handleBeforeUnload);
            if (intervalId) {
                clearInterval(intervalId);
            }
            // Optional: Save one last time on unmount? 
            // Usually dangerous for 'unmount' because it might save cleared state if navigating away
            // Ideally 'beforeunload' covers tab close, and navigation logic should handle manual saves if needed.
            // But for safety against crashes, the interval is key.
            handleBeforeUnload();
        };
    }, [saveState, enableAutoSave, autoSaveIntervalMs]);

    return { saveState, loadState, clearState };
}
