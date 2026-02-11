/**
 * InstanceOrchestrationService.ts
 *
 * Backend API client for workflow instance orchestration.
 * Enables deduplication and instance tracking via backend InstanceManager.
 *
 * Features:
 * - Check for existing instances (deduplication)
 * - Get instance summary for session
 * - Get active instances
 * - Instance statistics
 */

import axios, { AxiosInstance } from 'axios';

export interface InstanceByTriggerRequest {
    session_id: string;
    workflow_type: string;
    parent_workflow_id: string;
    trigger_source: string;
}

export interface InstanceDetails {
    instance_id: string;
    thread_id: string;
    workflow_type: string;
    parent_workflow_id: string | null;
    parent_workflow_type: string | null;
    trigger_source: string;
    main_thread_id: string;
    status: 'created' | 'running' | 'completed' | 'error' | 'cancelled';
    priority: 'low' | 'normal' | 'high' | 'critical';
    request_count: number;
    error_count: number;
    created_at: string;
    started_at: string | null;
    completed_at: string | null;
    last_activity: string;
    error_message: string | null;
    has_result: boolean;
    metadata: Record<string, any>;
}

export interface InstanceByTriggerResponse {
    success: boolean;
    data?: {
        exists: boolean;
        instance: InstanceDetails | null;
    };
    error?: string;
}

export interface InstanceSummary {
    session_id: string;
    pools: Record<string, {
        workflow_type: string;
        parent_workflow_id: string;
        instance_count: number;
        active_count: number;
        instances: InstanceDetails[];
    }>;
    total_instances: number;
}

export interface InstanceStats {
    total_sessions_with_instances: number;
    total_pools: number;
    total_instances: number;
    active_instances: number;
    completed_instances: number;
    error_instances: number;
    lifetime_created: number;
    lifetime_completed: number;
    lifetime_errored: number;
}

/**
 * InstanceOrchestrationService
 * Singleton service for backend instance API communication
 */
export class InstanceOrchestrationService {
    private static instance: InstanceOrchestrationService;
    private axiosInstance: AxiosInstance;

    private constructor(baseURL: string = import.meta.env.VITE_API_URL || 'http://localhost:5000') {
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
                console.log(`[INSTANCE_API] ${response.config.method?.toUpperCase()} ${response.config.url}:`, response.data);
                return response;
            },
            (error) => {
                console.error(`[INSTANCE_API] Error:`, error.response?.data || error.message);
                return Promise.reject(error);
            }
        );
    }

    public static getInstance(): InstanceOrchestrationService {
        if (!InstanceOrchestrationService.instance) {
            InstanceOrchestrationService.instance = new InstanceOrchestrationService();
        }
        return InstanceOrchestrationService.instance;
    }

    // ==========================================================================
    // DEDUPLICATION
    // ==========================================================================

    /**
     * Check if Instance Exists by Trigger
     * This is the KEY METHOD for preventing duplicate workflow instances
     *
     * Usage:
     *   const result = await instanceService.checkExistingInstance({
     *     session_id: mainThreadId,
     *     workflow_type: 'product_search',
     *     parent_workflow_id: parentWorkflowId,
     *     trigger_source: 'item_1'
     *   });
     *
     *   if (result.exists) {
     *     // RERUN - use existing instance
     *     console.log('Using existing instance:', result.instance);
     *   } else {
     *     // NEW - proceed with creating workflow
     *   }
     */
    public async checkExistingInstance(
        request: InstanceByTriggerRequest
    ): Promise<{ exists: boolean; instance: InstanceDetails | null }> {
        try {
            const response = await this.axiosInstance.post<InstanceByTriggerResponse>(
                '/api/agentic/instances/by-trigger',
                request
            );

            if (response.data.success && response.data.data) {
                return {
                    exists: response.data.data.exists,
                    instance: response.data.data.instance,
                };
            }

            return { exists: false, instance: null };
        } catch (error: any) {
            console.error('[INSTANCE_ORCHESTRATION] Failed to check existing instance:', error);
            return { exists: false, instance: null };
        }
    }

    // ==========================================================================
    // INSTANCE QUERIES
    // ==========================================================================

    /**
     * Get Instance by ID
     */
    public async getInstanceById(instanceId: string): Promise<InstanceDetails | null> {
        try {
            const response = await this.axiosInstance.get<{ success: boolean; data: InstanceDetails }>(
                `/api/agentic/instances/${instanceId}`
            );

            return response.data.success ? response.data.data : null;
        } catch (error: any) {
            console.error('[INSTANCE_ORCHESTRATION] Failed to get instance:', error);
            return null;
        }
    }

    /**
     * Get Instance Summary for Session
     */
    public async getInstanceSummary(
        sessionId: string,
        workflowType?: string
    ): Promise<InstanceSummary | null> {
        try {
            const params: Record<string, string> = {};
            if (workflowType) {
                params.workflow_type = workflowType;
            }

            const response = await this.axiosInstance.get<{ success: boolean; data: InstanceSummary }>(
                `/api/agentic/instances/summary/${sessionId}`,
                { params }
            );

            return response.data.success ? response.data.data : null;
        } catch (error: any) {
            console.error('[INSTANCE_ORCHESTRATION] Failed to get instance summary:', error);
            return null;
        }
    }

    /**
     * Get Active Instances for Session
     */
    public async getActiveInstances(sessionId: string): Promise<InstanceDetails[]> {
        try {
            const response = await this.axiosInstance.get<{
                success: boolean;
                data: { instances: InstanceDetails[]; count: number };
            }>(`/api/agentic/instances/active/${sessionId}`);

            return response.data.success ? response.data.data.instances : [];
        } catch (error: any) {
            console.error('[INSTANCE_ORCHESTRATION] Failed to get active instances:', error);
            return [];
        }
    }

    /**
     * Get Instance Statistics
     * Admin endpoint for monitoring
     */
    public async getStats(): Promise<InstanceStats | null> {
        try {
            const response = await this.axiosInstance.get<{ success: boolean; data: InstanceStats }>(
                '/api/agentic/instances/stats'
            );

            return response.data.success ? response.data.data : null;
        } catch (error: any) {
            console.error('[INSTANCE_ORCHESTRATION] Failed to get stats:', error);
            return null;
        }
    }
}

// Export singleton instance
export const instanceOrchestrationService = InstanceOrchestrationService.getInstance();

export default InstanceOrchestrationService;
