
import { useState, useEffect, useRef, useCallback } from 'react';
import { BASE_URL } from '@/components/AIRecommender/api';

interface QueuedRequest {
    fieldName: string;
    productType: string;
    contextValue?: string;
    priority: 'high' | 'low';
    resolve: (desc: string) => void;
    reject: (err: any) => void;
}

interface StoredDescription {
    description: string;
    timestamp: number;
}

const CONCURRENCY_LIMIT = 2; // Keep low to avoid browser blocking
const CACHE_TTL = 1000 * 60 * 60; // 1 hour

export const useFieldDescriptions = (productType: string) => {
    const [descriptions, setDescriptions] = useState<Record<string, string>>({});
    const processingRef = useRef(false);
    const queueRef = useRef<QueuedRequest[]>([]);
    const activeRequestsRef = useRef(0);
    const mountedRef = useRef(true);

    // Load from SessionStorage on init
    useEffect(() => {
        mountedRef.current = true;
        try {
            const cacheKey = `field_desc_cache_${productType}`;
            const stored = sessionStorage.getItem(cacheKey);
            if (stored) {
                const parsed = JSON.parse(stored);
                // Simple validation/expiry check could go here
                setDescriptions(parsed);
            }
        } catch (e) {
            console.warn("Failed to load field descriptions cache", e);
        }

        return () => {
            mountedRef.current = false;
        };
    }, [productType]);

    // Save to SessionStorage on change
    useEffect(() => {
        if (Object.keys(descriptions).length > 0) {
            const cacheKey = `field_desc_cache_${productType}`;
            sessionStorage.setItem(cacheKey, JSON.stringify(descriptions));
        }
    }, [descriptions, productType]);

    const processQueue = useCallback(async () => {
        if (!mountedRef.current) return;
        if (activeRequestsRef.current >= CONCURRENCY_LIMIT) return;
        if (queueRef.current.length === 0) return;

        // Sort queue: High priority first
        queueRef.current.sort((a, b) => {
            if (a.priority === b.priority) return 0;
            return a.priority === 'high' ? -1 : 1;
        });

        const request = queueRef.current.shift();
        if (!request) return;

        activeRequestsRef.current++;

        // Mark field as "loading" (optional, for UI)
        // setDescriptions(prev => ({ ...prev, [request.fieldName]: '...' }));

        try {
            const response = await fetch(`${BASE_URL}/api/describe_field`, {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({
                    field_name: request.fieldName,
                    product_type: request.productType,
                    context_value: request.contextValue
                })
            });

            if (!response.ok) throw new Error("API Failed");

            const data = await response.json();
            if (data.success && data.description) {
                if (mountedRef.current) {
                    setDescriptions(prev => ({
                        ...prev,
                        [request.fieldName]: data.description
                    }));
                    request.resolve(data.description);
                }
            } else {
                if (mountedRef.current) {
                    // Fallback or error state
                    request.resolve("");
                }
            }
        } catch (error) {
            console.error(`Failed to fetch description for ${request.fieldName}`, error);
            if (mountedRef.current) request.reject(error);
        } finally {
            activeRequestsRef.current--;
            processQueue(); // Process next
        }
    }, [productType]);

    const getDescription = useCallback((fieldName: string, contextValue?: string, priority: 'high' | 'low' = 'low') => {
        // Return existing if found
        if (descriptions[fieldName] && descriptions[fieldName] !== '...') return descriptions[fieldName];

        // Return if check for pending/loading logic (omitted for simplicity, but could add "pending" state)

        // Queue new request
        return new Promise<string>((resolve, reject) => {
            queueRef.current.push({
                fieldName,
                productType,
                contextValue,
                priority,
                resolve,
                reject
            });
            processQueue();
        });
    }, [descriptions, productType, processQueue]);

    // Specialized handler for "Hover" interaction (High Priority)
    const fetchOnHover = useCallback((fieldName: string, contextValue?: string) => {
        if (descriptions[fieldName]) return; // Already have it
        getDescription(fieldName, contextValue, 'high');
    }, [getDescription, descriptions]);

    return {
        descriptions,
        getDescription,
        fetchOnHover
    };
};
