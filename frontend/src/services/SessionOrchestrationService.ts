/**
 * SessionOrchestrationService.ts
 *
 * Backend API client for session lifecycle orchestration.
 * Syncs frontend session state with backend SessionOrchestrator.
 *
 * Features:
 * - Start session (login)
 * - Heartbeat (keep-alive, every 5 minutes)
 * - End session (logout)
 * - Session statistics
 */

import axios, { AxiosInstance } from 'axios';

export interface StartSessionRequest {
    user_id: string;
    main_thread_id: string;
    is_saved?: boolean;
    zone?: string;
    metadata?: Record<string, any>;
}

export interface StartSessionResponse {
    success: boolean;
    data?: {
        session_id: string;
        main_thread_id: string;
        created_at: string;
        message: string;
    };
    error?: string;
}

export interface HeartbeatResponse {
    success: boolean;
    data?: {
        main_thread_id: string;
        last_activity: string;
        message: string;
    };
    error?: string;
}

export interface ValidateSessionResponse {
    success: boolean;
    data?: {
        valid: boolean;
        session_id?: string;
        user_id?: string;
        created_at?: string;
        last_activity?: string;
        is_active?: boolean;
        inactive_minutes?: number;
        reason?: string;
    };
    error?: string;
}

export interface SessionStats {
    active_sessions: number;
    active_users: number;
    total_workflows: number;
    sessions: Record<string, any>;
}

export interface SessionDetails {
    user_id: string;
    main_thread_id: string;
    created_at: string;
    last_activity: string;
    workflow_count: number;
    active: boolean;
    is_saved: boolean;
    request_count: number;
    zone: string;
}

/**
 * SessionOrchestrationService
 * Singleton service for backend session API communication
 */
export class SessionOrchestrationService {
    private static instance: SessionOrchestrationService;
    private axiosInstance: AxiosInstance;
    private heartbeatInterval: NodeJS.Timeout | null = null;
    private heartbeatIntervalMs: number = 5 * 60 * 1000; // 5 minutes
    private currentMainThreadId: string | null = null;

    private constructor(baseURL: string = import.meta.env.VITE_API_URL || '') {
        this.axiosInstance = axios.create({
            baseURL,
            timeout: 10000,
            headers: {
                'Content-Type': 'application/json',
            },
        });

        // Log responses for debugging
        this.axiosInstance.interceptors.response.use(
            (response) => {
                console.log(`[SESSION_API] ${response.config.method?.toUpperCase()} ${response.config.url}:`, response.data);
                return response;
            },
            (error) => {
                console.error(`[SESSION_API] Error:`, error.response?.data || error.message);
                return Promise.reject(error);
            }
        );
    }

    public static getInstance(): SessionOrchestrationService {
        if (!SessionOrchestrationService.instance) {
            SessionOrchestrationService.instance = new SessionOrchestrationService();
        }
        return SessionOrchestrationService.instance;
    }

    // ==========================================================================
    // SESSION LIFECYCLE
    // ==========================================================================

    /**
     * Start Session
     * Called on user login to register session with backend
     */
    public async startSession(request: StartSessionRequest): Promise<StartSessionResponse> {
        try {
            const response = await this.axiosInstance.post<StartSessionResponse>(
                '/api/agentic/sessions/start',
                request
            );

            if (response.data.success) {
                this.currentMainThreadId = request.main_thread_id;
                this.startHeartbeat();
                console.log(`[SESSION_ORCHESTRATION] Session started: ${request.main_thread_id}`);
            }

            return response.data;
        } catch (error: any) {
            console.error('[SESSION_ORCHESTRATION] Failed to start session:', error);
            return {
                success: false,
                error: error.response?.data?.error || error.message,
            };
        }
    }

    /**
     * Send Heartbeat
     * Called every 5 minutes to keep session alive
     */
    public async heartbeat(mainThreadId?: string): Promise<HeartbeatResponse> {
        const threadId = mainThreadId || this.currentMainThreadId;

        if (!threadId) {
            console.warn('[SESSION_ORCHESTRATION] No main_thread_id for heartbeat');
            return { success: false, error: 'No main_thread_id' };
        }

        try {
            const response = await this.axiosInstance.post<HeartbeatResponse>(
                '/api/agentic/sessions/heartbeat',
                { main_thread_id: threadId }
            );

            return response.data;
        } catch (error: any) {
            console.error('[SESSION_ORCHESTRATION] Heartbeat failed:', error);
            return {
                success: false,
                error: error.response?.data?.error || error.message,
            };
        }
    }

    /**
     * Validate Session
     * Check if a session exists and is still valid on the backend
     * Use this before reusing a stored session ID
     */
    public async validateSession(mainThreadId: string): Promise<ValidateSessionResponse> {
        try {
            const response = await this.axiosInstance.get<ValidateSessionResponse>(
                `/api/agentic/sessions/${mainThreadId}/validate`
            );

            if (response.data.success && response.data.data?.valid) {
                console.log(`[SESSION_ORCHESTRATION] Session valid: ${mainThreadId}`);
            } else {
                console.warn(
                    `[SESSION_ORCHESTRATION] Session invalid: ${mainThreadId} - ` +
                    `${response.data.data?.reason || 'Unknown reason'}`
                );
            }

            return response.data;
        } catch (error: any) {
            console.error('[SESSION_ORCHESTRATION] Session validation failed:', error);

            // Handle specific status codes
            if (error.response?.status === 404) {
                return {
                    success: false,
                    data: {
                        valid: false,
                        reason: 'Session not found'
                    }
                };
            }

            if (error.response?.status === 410) {
                return {
                    success: false,
                    data: {
                        valid: false,
                        reason: error.response.data?.data?.reason || 'Session expired'
                    }
                };
            }

            return {
                success: false,
                error: error.response?.data?.error || error.message,
                data: {
                    valid: false,
                    reason: 'Validation request failed'
                }
            };
        }
    }

    /**
     * End Session
     * Called on user logout to clean up backend session
     */
    public async endSession(mainThreadId?: string): Promise<boolean> {
        const threadId = mainThreadId || this.currentMainThreadId;

        if (!threadId) {
            console.warn('[SESSION_ORCHESTRATION] No main_thread_id for end session');
            return false;
        }

        try {
            this.stopHeartbeat();

            const response = await this.axiosInstance.post(
                '/api/agentic/sessions/end',
                { main_thread_id: threadId }
            );

            if (response.data.success) {
                this.currentMainThreadId = null;
                console.log(`[SESSION_ORCHESTRATION] Session ended: ${threadId}`);
            }

            return response.data.success;
        } catch (error: any) {
            console.error('[SESSION_ORCHESTRATION] Failed to end session:', error);
            return false;
        }
    }

    // ==========================================================================
    // HEARTBEAT MANAGEMENT
    // ==========================================================================

    /**
     * Start Heartbeat Interval
     * Automatically sends heartbeat every 5 minutes
     */
    public startHeartbeat(): void {
        if (this.heartbeatInterval) {
            console.log('[SESSION_ORCHESTRATION] Heartbeat already running');
            return;
        }

        this.heartbeatInterval = setInterval(() => {
            this.heartbeat().catch((error) => {
                console.error('[SESSION_ORCHESTRATION] Heartbeat interval error:', error);
            });
        }, this.heartbeatIntervalMs);

        console.log(`[SESSION_ORCHESTRATION] Heartbeat started (${this.heartbeatIntervalMs / 1000}s interval)`);
    }

    /**
     * Stop Heartbeat Interval
     */
    public stopHeartbeat(): void {
        if (this.heartbeatInterval) {
            clearInterval(this.heartbeatInterval);
            this.heartbeatInterval = null;
            console.log('[SESSION_ORCHESTRATION] Heartbeat stopped');
        }
    }

    /**
     * Set Heartbeat Interval (in milliseconds)
     */
    public setHeartbeatInterval(intervalMs: number): void {
        this.heartbeatIntervalMs = intervalMs;

        // Restart if already running
        if (this.heartbeatInterval) {
            this.stopHeartbeat();
            this.startHeartbeat();
        }
    }

    // ==========================================================================
    // SESSION QUERIES
    // ==========================================================================

    /**
     * Get Session Statistics
     * Admin endpoint for monitoring
     */
    public async getStats(): Promise<SessionStats | null> {
        try {
            const response = await this.axiosInstance.get<{ success: boolean; data: SessionStats }>(
                '/api/agentic/sessions/stats'
            );

            return response.data.success ? response.data.data : null;
        } catch (error: any) {
            console.error('[SESSION_ORCHESTRATION] Failed to get stats:', error);
            return null;
        }
    }

    /**
     * Get Session Details
     */
    public async getSession(mainThreadId: string): Promise<SessionDetails | null> {
        try {
            const response = await this.axiosInstance.get<{ success: boolean; data: { session: SessionDetails } }>(
                `/api/agentic/sessions/${mainThreadId}`
            );

            return response.data.success ? response.data.data.session : null;
        } catch (error: any) {
            console.error('[SESSION_ORCHESTRATION] Failed to get session:', error);
            return null;
        }
    }

    /**
     * Get Current Main Thread ID
     */
    public getCurrentMainThreadId(): string | null {
        return this.currentMainThreadId;
    }

    /**
     * Set Current Main Thread ID (for manual override)
     */
    public setCurrentMainThreadId(mainThreadId: string | null): void {
        this.currentMainThreadId = mainThreadId;
    }

    /**
     * Check if Session is Active
     */
    public isSessionActive(): boolean {
        return this.currentMainThreadId !== null && this.heartbeatInterval !== null;
    }
}

// Export singleton instance
export const sessionOrchestrationService = SessionOrchestrationService.getInstance();

export default SessionOrchestrationService;
