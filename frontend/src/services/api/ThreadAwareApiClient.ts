

import axios, { AxiosInstance, AxiosRequestConfig, InternalAxiosRequestConfig } from 'axios';
import { SessionManager } from '../SessionManager';

export interface ThreadContext {
  mainThreadId: string;
  subThreadId?: string;
  itemThreadId?: string;
  sessionId: string;
  zone?: string;
}

export interface ApiRequestOptions extends AxiosRequestConfig {
  includeThreadContext?: boolean;
  useItemThreadId?: boolean;
}

export class ThreadAwareApiClient {
  private axiosInstance: AxiosInstance;
  private sessionManager: SessionManager;
  private baseURL: string;

  constructor(baseURL: string = import.meta.env.VITE_API_URL || '') {
    this.baseURL = baseURL;
    this.sessionManager = SessionManager.getInstance();

    this.axiosInstance = axios.create({
      baseURL,
      timeout: 30000,
      headers: {
        'Content-Type': 'application/json',
      },
    });

    // Add request interceptor to inject thread context
    this.axiosInstance.interceptors.request.use(
      (config) => this.injectThreadContext(config),
      (error) => Promise.reject(error)
    );

    // Add response interceptor for error handling
    this.axiosInstance.interceptors.response.use(
      (response) => response,
      (error) => {
        console.error('[API] Request failed:', error.response?.data || error.message);
        return Promise.reject(error);
      }
    );
  }

  /**
   * Get Current Thread Context
   * Returns all relevant thread IDs from current session
   */
  private getThreadContext(): ThreadContext | null {
    const session = this.sessionManager.getCurrentSession();

    if (!session) {
      console.warn('[API] No active session - thread context unavailable');
      return null;
    }

    return {
      mainThreadId: session.mainThreadId,
      subThreadId: session.activeSubThreadId,
      sessionId: session.sessionId,
      zone: session.zone,
    };
  }

  /**
   * Inject Thread Context into Request
   * Automatically adds thread IDs to request body/headers
   */
  private injectThreadContext(config: InternalAxiosRequestConfig): InternalAxiosRequestConfig {
    const threadContext = this.getThreadContext();

    if (!threadContext) {
      console.warn('[API] Thread context not available for request');
      return config;
    }

    // Add to request body for POST/PUT requests
    if (['post', 'put', 'patch'].includes(config.method?.toLowerCase() || '')) {
      config.data = {
        ...config.data,
        main_thread_id: threadContext.mainThreadId,
        workflow_thread_id: threadContext.subThreadId,
        session_id: threadContext.sessionId,
        zone: threadContext.zone,
      };
    }

    // Add to query params for GET requests
    if (config.method?.toLowerCase() === 'get') {
      config.params = {
        ...config.params,
        main_thread_id: threadContext.mainThreadId,
        workflow_thread_id: threadContext.subThreadId,
        session_id: threadContext.sessionId,
      };
    }

    // Also add to headers for reference
    if (config.headers) {
      config.headers['X-Main-Thread-ID'] = threadContext.mainThreadId;
      config.headers['X-Workflow-Thread-ID'] = threadContext.subThreadId || '';
      config.headers['X-Session-ID'] = threadContext.sessionId;
      config.headers['X-Zone'] = threadContext.zone || 'DEFAULT';
    }

    console.log('[API] Injected thread context:', {
      mainThreadId: threadContext.mainThreadId,
      subThreadId: threadContext.subThreadId,
      sessionId: threadContext.sessionId,
    });

    return config;
  }

  // ========================================================================
  // WORKFLOW API CALLS
  // ========================================================================

  /**
   * Run Instrument Identifier Workflow
   * Creates a sub-thread for this workflow
   */
  async runInstrumentIdentifier(requirements: string): Promise<any> {
    // Create sub-thread for this workflow
    const subThread = this.sessionManager.createSubThread('instrument_identifier');

    if (!subThread) {
      throw new Error('Failed to create sub-thread for instrument_identifier');
    }

    const response = await this.axiosInstance.post('/api/agentic/instrument-identifier', {
      requirements,
      // Thread context auto-injected by interceptor
    });

    // Process response and create item threads
    if (response.data?.data?.response_data?.items) {
      const items = response.data.data.response_data.items;

      items.forEach((item: any) => {
        this.sessionManager.addItemThreadToSubThread(
          subThread.subThreadId,
          item.number,
          item.name,
          item.type
        );
      });
    }

    return response.data;
  }

  /**
   * Run Solution Workflow
   * Creates a sub-thread for this workflow
   */
  async runSolution(solutionDescription: string): Promise<any> {
    // Create sub-thread for this workflow
    const subThread = this.sessionManager.createSubThread('solution');

    if (!subThread) {
      throw new Error('Failed to create sub-thread for solution');
    }

    const response = await this.axiosInstance.post('/api/agentic/solution', {
      user_input: solutionDescription,
      // Thread context auto-injected by interceptor
    });

    // Process response and create item threads
    if (response.data?.data?.response_data?.items) {
      const items = response.data.data.response_data.items;

      items.forEach((item: any) => {
        this.sessionManager.addItemThreadToSubThread(
          subThread.subThreadId,
          item.number,
          item.name,
          item.type
        );
      });
    }

    return response.data;
  }

  /**
   * Search Products
   * Creates a new product_search sub-thread for each search
   */
  async searchProducts(
    parentWorkflowThreadId: string,
    itemNumber: number,
    itemName: string,
    requirements: any
  ): Promise<any> {
    // Get the item thread ID from parent
    const itemThreads = this.sessionManager.getItemThreadsInSubThread(parentWorkflowThreadId);
    const itemThreadId = itemThreads?.get(itemNumber);

    if (!itemThreadId) {
      throw new Error(`Item thread not found for item ${itemNumber}`);
    }

    // Create a new product_search sub-thread
    const productSearchSubThread = this.sessionManager.createProductSearchSubThread(
      parentWorkflowThreadId,
      itemNumber,
      itemThreadId
    );

    if (!productSearchSubThread) {
      throw new Error('Failed to create product search sub-thread');
    }

    const response = await this.axiosInstance.post('/api/agentic/product-search', {
      item_number: itemNumber,
      item_name: itemName,
      item_thread_id: itemThreadId,
      parent_workflow_thread_id: parentWorkflowThreadId,
      requirements,
      // Thread context auto-injected by interceptor (will use new product_search sub-thread)
    });

    return {
      ...response.data,
      productSearchSubThreadId: productSearchSubThread.subThreadId,
      itemThreadId,
    };
  }

  /**
   * Select Product
   * Updates the product search thread with selected product
   */
  async selectProduct(
    productSearchSubThreadId: string,
    itemNumber: number,
    selectedProduct: any
  ): Promise<any> {
    // Set this as the active sub-thread
    this.sessionManager.setActiveSubThread(productSearchSubThreadId);

    const response = await this.axiosInstance.post('/api/agentic/select-product', {
      item_number: itemNumber,
      selected_product: selectedProduct,
      // Thread context auto-injected by interceptor
    });

    return response.data;
  }

  /**
   * Get Thread Tree
   * Retrieve the complete thread hierarchy
   */
  async getThreadTree(): Promise<any> {
    const session = this.sessionManager.getCurrentSession();

    if (!session) {
      throw new Error('No active session');
    }

    const response = await this.axiosInstance.get(
      `/api/agentic/threads/${session.mainThreadId}/tree`,
      {
        params: {
          session_id: session.sessionId,
        },
      }
    );

    return response.data;
  }

  /**
   * Get Item State
   * Retrieve full persistent state for an item
   */
  async getItemState(itemThreadId: string): Promise<any> {
    const response = await this.axiosInstance.get(
      `/api/agentic/threads/${itemThreadId}/state`
    );

    return response.data;
  }

  /**
   * Update Item State
   * Update specific fields in item state
   */
  async updateItemState(itemThreadId: string, updates: any): Promise<any> {
    const response = await this.axiosInstance.put(
      `/api/agentic/threads/${itemThreadId}/state`,
      updates
    );

    return response.data;
  }

  /**
   * Generic GET request with thread context
   */
  async get(endpoint: string, config?: ApiRequestOptions): Promise<any> {
    return this.axiosInstance.get(endpoint, config);
  }

  /**
   * Generic POST request with thread context
   */
  async post(endpoint: string, data?: any, config?: ApiRequestOptions): Promise<any> {
    return this.axiosInstance.post(endpoint, data, config);
  }

  /**
   * Generic PUT request with thread context
   */
  async put(endpoint: string, data?: any, config?: ApiRequestOptions): Promise<any> {
    return this.axiosInstance.put(endpoint, data, config);
  }

  /**
   * Generic DELETE request with thread context
   */
  async delete(endpoint: string, config?: ApiRequestOptions): Promise<any> {
    return this.axiosInstance.delete(endpoint, config);
  }
}

// Create singleton instance
export const apiClient = new ThreadAwareApiClient();

export default apiClient;
