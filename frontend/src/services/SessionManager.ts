/**
 * SessionManager.ts
 *
 * Manages complete session lifecycle for AIPR application
 * - Session creation at login
 * - Main thread ID generation
 * - Workflow and sub-thread tracking
 * - Session persistence
 * - Multi-window support (100+ concurrent users)
 */

import { v4 as uuidv4 } from 'uuid';
import { SessionOrchestrationService, sessionOrchestrationService } from './SessionOrchestrationService';

// Workflow type aligned with backend WorkflowTarget enum
export type WorkflowType =
  | 'engenie_chat'           // WorkflowTarget.ENGENIE_CHAT
  | 'solution'               // WorkflowTarget.SOLUTION_WORKFLOW  
  | 'instrument_identifier'  // WorkflowTarget.INSTRUMENT_IDENTIFIER
  | 'out_of_domain'          // WorkflowTarget.OUT_OF_DOMAIN
  | null;

export interface SubThread {
  subThreadId: string;
  workflowType: 'instrument_identifier' | 'solution' | 'product_search' | 'grounded_chat';
  createdAt: string;
  status: 'active' | 'completed' | 'archived';
  itemThreads: Map<number, string>; // Maps item number to item thread ID
}

export interface UserSession {
  sessionId: string;
  userId: string;
  mainThreadId: string;
  zone?: string;
  loginTime: string;
  lastActivityTime: string;
  subThreads: Map<string, SubThread>; // Map of subThreadId -> SubThread
  activeSubThreadId?: string; // Currently active sub-thread
  windowCount: number;
  // Workflow state management
  currentWorkflow: WorkflowType;
  workflowLockedAt?: string;
}

export interface SessionMetadata {
  sessionId: string;
  userId: string;
  mainThreadId: string;
  subThreadCount: number;
  activeWindowCount: number;
  createdAt: string;
  lastUpdated: string;
}

/**
 * SessionManager Class
 * Handles all session and thread creation/management on UI side
 */
export class SessionManager {
  private static instance: SessionManager;
  private sessions: Map<string, UserSession> = new Map();
  private sessionStorageKey = 'aipr_sessions';
  private currentSessionId?: string;
  private storageType: 'localStorage' | 'sessionStorage' = 'localStorage';
  private orchestrationService: SessionOrchestrationService;
  private sessionCreationPromise: Promise<UserSession> | null = null; // Track in-flight creation
  private tabInstanceId: string; // Unique ID for this browser tab

  // Workflow TTL (10 minutes)
  private readonly WORKFLOW_TTL_MS = 10 * 60 * 1000;

  private constructor() {
    this.orchestrationService = sessionOrchestrationService;

    // Initialize tab instance ID (unique per browser tab)
    this.tabInstanceId = this.getOrCreateTabInstanceId();

    this.loadSessionsFromStorage();

    // Cross-tab synchronization
    if (typeof window !== 'undefined') {
      window.addEventListener('storage', (e: StorageEvent) => {
        if (e.key === this.sessionStorageKey) {
          console.log('[SESSION] Cross-tab sync detected, reloading sessions...');
          this.loadSessionsFromStorage();
        }
      });
    }
  }

  public static getInstance(): SessionManager {
    if (!SessionManager.instance) {
      SessionManager.instance = new SessionManager();
    }
    return SessionManager.instance;
  }

  /**
   * GET OR CREATE TAB INSTANCE ID
   * Generates a unique ID for this browser tab, persisted in sessionStorage.
   * This enables tab-isolated persistence for EnGenie Chat.
   */
  private getOrCreateTabInstanceId(): string {
    if (typeof window === 'undefined') return 'server';

    const TAB_ID_KEY = 'aipr_tab_instance_id';
    let tabId = sessionStorage.getItem(TAB_ID_KEY);

    if (!tabId) {
      tabId = `tab_${uuidv4().split('-')[0]}_${Date.now()}`;
      sessionStorage.setItem(TAB_ID_KEY, tabId);
      console.log(`[SESSION] Created new tab instance: ${tabId}`);
    } else {
      // Edge case: Detect very old tab IDs (>1 hour) and regenerate
      // This handles browser tab duplication edge cases
      const now = Date.now();
      const storedTime = parseInt(tabId.split('_').pop() || '0');
      if (now - storedTime > 3600000) {
        console.log('[SESSION] Tab ID too old, regenerating...');
        tabId = `tab_${uuidv4().split('-')[0]}_${Date.now()}`;
        sessionStorage.setItem(TAB_ID_KEY, tabId);
        console.log(`[SESSION] Regenerated tab instance: ${tabId}`);
      } else {
        console.log(`[SESSION] Using existing tab instance: ${tabId}`);
      }
    }

    return tabId;
  }

  /**
   * GET TAB INSTANCE ID
   * Public method to access current tab's unique identifier.
   * Used by EnGenieChat to create tab-specific persistence keys.
   */
  public getTabInstanceId(): string {
    return this.tabInstanceId;
  }

  /**
   * GET OR CREATE SESSION (RECOMMENDED)
   * Prevents duplicate session creation with built-in deduplication
   * Use this instead of createSession() to avoid race conditions
   */
  public async getOrCreateSession(userId: string, zone?: string): Promise<UserSession> {
    // 1. Check if we already have a local session for this user
    const existingSession = this.findSessionByUserId(userId);
    if (existingSession) {
      // Validate session with backend before reusing
      console.log(`[SESSION] Found existing local session for ${userId}, validating...`);
      try {
        const validation = await this.orchestrationService.validateSession(
          existingSession.mainThreadId
        );

        if (validation.success && validation.data?.valid) {
          console.log(`[SESSION] ✓ Session validated, reusing for user ${userId}`);
          this.currentSessionId = existingSession.sessionId;
          return existingSession;
        } else {
          console.warn(
            `[SESSION] ✗ Session invalid (${validation.data?.reason}), creating new`
          );
          // Remove invalid session
          this.sessions.delete(existingSession.sessionId);
          this.persistSessionsToStorage();
        }
      } catch (error) {
        console.error('[SESSION] Validation failed, assuming session invalid:', error);
        // Remove session on validation error
        this.sessions.delete(existingSession.sessionId);
        this.persistSessionsToStorage();
      }
    }

    // 2. Check if session creation is already in progress
    if (this.sessionCreationPromise) {
      console.log(`[SESSION] Session creation in progress, waiting...`);
      return this.sessionCreationPromise;
    }

    // 3. Create new session (with promise tracking to prevent duplicates)
    console.log(`[SESSION] Creating new session for user ${userId}`);
    this.sessionCreationPromise = this.createSessionAsync(userId, zone);

    try {
      const session = await this.sessionCreationPromise;
      return session;
    } finally {
      this.sessionCreationPromise = null;
    }
  }

  /**
   * CREATE NEW SESSION (on user login)
   * Called when user successfully logs in
   * NOTE: Use getOrCreateSession() instead to prevent duplicates
   */
  public createSession(userId: string, zone?: string): UserSession {
    const sessionId = uuidv4();
    const mainThreadId = this.generateMainThreadId(userId, zone);

    const session: UserSession = {
      sessionId,
      userId,
      mainThreadId,
      zone: zone || 'DEFAULT',
      loginTime: new Date().toISOString(),
      lastActivityTime: new Date().toISOString(),
      subThreads: new Map(),
      windowCount: 1,
      currentWorkflow: null,
    };

    this.sessions.set(sessionId, session);
    this.currentSessionId = sessionId;
    this.persistSessionsToStorage();

    // Sync with backend orchestrator (async, non-blocking)
    this.orchestrationService.startSession({
      user_id: userId,
      main_thread_id: mainThreadId,
      zone: zone || 'DEFAULT',
      metadata: { frontend_session_id: sessionId },
    }).catch((error) => {
      console.error('[SESSION] Failed to sync with backend:', error);
    });

    console.log(`[SESSION] Created session for user ${userId}`);
    console.log(`[SESSION] Main Thread ID: ${mainThreadId}`);
    console.log(`[SESSION] Session ID: ${sessionId}`);

    return session;
  }

  /**
   * CREATE SESSION ASYNC (internal helper)
   * Async version of createSession for use with getOrCreateSession
   */
  private async createSessionAsync(userId: string, zone?: string): Promise<UserSession> {
    const sessionId = uuidv4();
    const mainThreadId = this.generateMainThreadId(userId, zone);

    const session: UserSession = {
      sessionId,
      userId,
      mainThreadId,
      zone: zone || 'DEFAULT',
      loginTime: new Date().toISOString(),
      lastActivityTime: new Date().toISOString(),
      subThreads: new Map(),
      windowCount: 1,
      currentWorkflow: null,
    };

    this.sessions.set(sessionId, session);
    this.currentSessionId = sessionId;
    this.persistSessionsToStorage();

    // Sync with backend orchestrator (async, with error handling)
    try {
      await this.orchestrationService.startSession({
        user_id: userId,
        main_thread_id: mainThreadId,
        zone: zone || 'DEFAULT',
        metadata: { frontend_session_id: sessionId },
      });
      console.log(`[SESSION] Backend sync successful for ${mainThreadId}`);
    } catch (error) {
      console.error('[SESSION] Failed to sync with backend:', error);
      // Continue anyway - session created locally
    }

    console.log(`[SESSION] Created session for user ${userId}`);
    console.log(`[SESSION] Main Thread ID: ${mainThreadId}`);
    console.log(`[SESSION] Session ID: ${sessionId}`);

    return session;
  }

  /**
   * FIND SESSION BY USER ID
   * Helper to check if user already has an active session
   */
  private findSessionByUserId(userId: string): UserSession | undefined {
    for (const [sessionId, session] of this.sessions.entries()) {
      if (session.userId === userId) {
        return session;
      }
    }
    return undefined;
  }

  /**
   * GET CURRENT SESSION
   */
  public getCurrentSession(): UserSession | undefined {
    if (!this.currentSessionId) return undefined;
    return this.sessions.get(this.currentSessionId);
  }

  /**
   * GET MAIN THREAD ID for current session
   * Convenience method for getting main thread ID directly
   */
  public getMainThreadId(): string | null {
    const session = this.getCurrentSession();
    return session?.mainThreadId || null;
  }

  /**
   * RESTORE FROM PROJECT (NEW)
   * Restore session context from a saved project
   * This allows session continuity across page loads
   */
  public restoreFromProject(savedContext: {
    session_id?: string | null;
    instance_id?: string | null;
    main_thread_id?: string | null;
    user_id?: string | null;
  }): void {
    if (!savedContext) {
      console.warn('[SessionManager] restoreFromProject called with no context');
      return;
    }

    console.log('[SessionManager] ========== RESTORING SESSION ==========');
    console.log('[SessionManager] Saved context:', savedContext);

    // Check if we need to create a new session with restored IDs
    const currentSession = this.getCurrentSession();

    if (savedContext.main_thread_id) {
      if (!currentSession || currentSession.mainThreadId !== savedContext.main_thread_id) {
        console.log('[SessionManager] Creating session with restored IDs');

        // Create a restored session
        const restoredSession: UserSession = {
          sessionId: savedContext.session_id || uuidv4(),
          userId: savedContext.user_id || 'restored_user',
          mainThreadId: savedContext.main_thread_id,
          zone: 'RESTORED',
          loginTime: new Date().toISOString(),
          lastActivityTime: new Date().toISOString(),
          subThreads: new Map(),
          windowCount: 1,
          currentWorkflow: 'solution'
        };

        this.sessions.set(restoredSession.sessionId, restoredSession);
        this.currentSessionId = restoredSession.sessionId;
        this.persistSessionsToStorage();

        console.log('[SessionManager] ✅ Restored session:', restoredSession.sessionId);
        console.log('[SessionManager] ✅ Main thread ID:', restoredSession.mainThreadId);
      } else {
        console.log('[SessionManager] ✅ Current session matches restored thread ID');
      }
    }

    console.log('[SessionManager] ==========================================');
  }

  /**
   * GET THREAD CONTEXT (for saving to project)
   */
  public getThreadContext() {
    const session = this.getCurrentSession();
    return {
      session_id: session?.sessionId || null,
      instance_id: this.tabInstanceId,
      main_thread_id: session?.mainThreadId || null,
      user_id: session?.userId || null
    };
  }

  // ========================================================================
  // WORKFLOW STATE MANAGEMENT (NEW)
  // ========================================================================

  /**
   * SET WORKFLOW - Store current workflow with TTL
   */
  public setWorkflow(workflow: WorkflowType): void {
    const session = this.getCurrentSession();
    if (!session) {
      console.warn('[SESSION] Cannot set workflow - no active session');
      return;
    }
    session.currentWorkflow = workflow;
    session.workflowLockedAt = workflow ? new Date().toISOString() : undefined;
    this.persistSessionsToStorage();
    console.log(`[SESSION] Workflow set: ${workflow}`);
  }

  /**
   * GET WORKFLOW - Returns current workflow (null if expired)
   */
  public getWorkflow(): WorkflowType {
    const session = this.getCurrentSession();
    if (!session?.currentWorkflow) return null;

    // TTL expiration check
    if (session.workflowLockedAt) {
      const elapsed = Date.now() - new Date(session.workflowLockedAt).getTime();
      if (elapsed > this.WORKFLOW_TTL_MS) {
        console.log('[SESSION] Workflow TTL expired, clearing...');
        this.clearWorkflow();
        return null;
      }
    }
    return session.currentWorkflow;
  }

  /**
   * CLEAR WORKFLOW - Remove workflow lock
   */
  public clearWorkflow(): void {
    const session = this.getCurrentSession();
    if (session) {
      session.currentWorkflow = null;
      session.workflowLockedAt = undefined;
      this.persistSessionsToStorage();
      console.log('[SESSION] Workflow cleared');
    }
  }

  /**
   * REFRESH WORKFLOW TTL - Extend expiration on activity
   */
  public refreshWorkflowTTL(): void {
    const session = this.getCurrentSession();
    if (session?.currentWorkflow) {
      session.workflowLockedAt = new Date().toISOString();
      this.persistSessionsToStorage();
    }
  }

  /**
   * CREATE SUB-THREAD (for each workflow/search)
   * Called when user starts a new workflow or search
   */
  public createSubThread(
    workflowType: 'instrument_identifier' | 'solution' | 'product_search' | 'grounded_chat',
    sessionId?: string
  ): SubThread | null {
    const session = sessionId ? this.sessions.get(sessionId) : this.getCurrentSession();

    if (!session) {
      console.error('[SESSION] No active session found');
      return null;
    }

    const subThreadId = this.generateSubThreadId(session.mainThreadId, workflowType);

    const subThread: SubThread = {
      subThreadId,
      workflowType,
      createdAt: new Date().toISOString(),
      status: 'active',
      itemThreads: new Map(),
    };

    session.subThreads.set(subThreadId, subThread);
    session.activeSubThreadId = subThreadId;
    session.lastActivityTime = new Date().toISOString();

    this.persistSessionsToStorage();

    console.log(`[SUB-THREAD] Created ${workflowType} sub-thread: ${subThreadId}`);
    console.log(`[SUB-THREAD] Parent main thread: ${session.mainThreadId}`);

    return subThread;
  }

  /**
   * CREATE ITEM THREAD (for each identified instrument/accessory)
   * Called when backend identifies items and returns item list
   */
  public addItemThreadToSubThread(
    subThreadId: string,
    itemNumber: number,
    itemName: string,
    itemType: 'instrument' | 'accessory',
    sessionId?: string
  ): string | null {
    const session = sessionId ? this.sessions.get(sessionId) : this.getCurrentSession();

    if (!session) {
      console.error('[SESSION] No active session found');
      return null;
    }

    const subThread = session.subThreads.get(subThreadId);
    if (!subThread) {
      console.error(`[SESSION] Sub-thread ${subThreadId} not found`);
      return null;
    }

    // Generate item thread ID locally
    const itemThreadId = this.generateItemThreadId(
      subThreadId,
      itemType,
      itemName,
      itemNumber
    );

    // Store in sub-thread
    subThread.itemThreads.set(itemNumber, itemThreadId);
    session.lastActivityTime = new Date().toISOString();

    this.persistSessionsToStorage();

    console.log(`[ITEM-THREAD] Created ${itemType} item thread for item ${itemNumber}`);
    console.log(`[ITEM-THREAD] Item: ${itemName}`);
    console.log(`[ITEM-THREAD] Thread ID: ${itemThreadId}`);

    return itemThreadId;
  }

  /**
   * CREATE PRODUCT SEARCH SUB-THREAD (for each product search)
   * Called when user selects an item and starts product search
   */
  public createProductSearchSubThread(
    parentSubThreadId: string,
    itemNumber: number,
    itemThreadId: string,
    sessionId?: string
  ): SubThread | null {
    const session = sessionId ? this.sessions.get(sessionId) : this.getCurrentSession();

    if (!session) {
      console.error('[SESSION] No active session found');
      return null;
    }

    // Create new product_search sub-thread
    const productSearchSubThread = this.createSubThread('product_search', sessionId);

    if (!productSearchSubThread) {
      return null;
    }

    // Link it to the parent item
    productSearchSubThread.itemThreads.set(itemNumber, itemThreadId);
    session.lastActivityTime = new Date().toISOString();

    this.persistSessionsToStorage();

    console.log(`[PRODUCT-SEARCH] Created product search thread: ${productSearchSubThread.subThreadId}`);
    console.log(`[PRODUCT-SEARCH] For item ${itemNumber}: ${itemThreadId}`);
    console.log(`[PRODUCT-SEARCH] Parent workflow: ${parentSubThreadId}`);

    return productSearchSubThread;
  }

  /**
   * GET ALL ITEM THREADS IN SUB-THREAD
   */
  public getItemThreadsInSubThread(subThreadId: string, sessionId?: string): Map<number, string> | null {
    const session = sessionId ? this.sessions.get(sessionId) : this.getCurrentSession();

    if (!session) return null;

    const subThread = session.subThreads.get(subThreadId);
    return subThread?.itemThreads || null;
  }

  /**
   * CLOSE SUB-THREAD (when workflow/search completes)
   */
  public closeSubThread(subThreadId: string, sessionId?: string): boolean {
    const session = sessionId ? this.sessions.get(sessionId) : this.getCurrentSession();

    if (!session) return false;

    const subThread = session.subThreads.get(subThreadId);
    if (!subThread) return false;

    subThread.status = 'completed';
    session.lastActivityTime = new Date().toISOString();

    this.persistSessionsToStorage();

    console.log(`[SUB-THREAD] Closed sub-thread: ${subThreadId}`);
    return true;
  }

  /**
   * SET ACTIVE SUB-THREAD
   * Used when switching between open windows
   */
  public setActiveSubThread(subThreadId: string, sessionId?: string): boolean {
    const session = sessionId ? this.sessions.get(sessionId) : this.getCurrentSession();

    if (!session) return false;

    if (!session.subThreads.has(subThreadId)) return false;

    session.activeSubThreadId = subThreadId;
    session.lastActivityTime = new Date().toISOString();

    this.persistSessionsToStorage();

    console.log(`[SESSION] Set active sub-thread: ${subThreadId}`);
    return true;
  }

  /**
   * UPDATE WINDOW COUNT
   * Track how many windows are open for a session
   */
  public updateWindowCount(count: number, sessionId?: string): void {
    const session = sessionId ? this.sessions.get(sessionId) : this.getCurrentSession();

    if (!session) return;

    session.windowCount = count;
    session.lastActivityTime = new Date().toISOString();
    this.persistSessionsToStorage();

    console.log(`[SESSION] Updated window count: ${count}`);
  }

  /**
   * END SESSION (on user logout)
   * Called when user logs out
   */
  public endSession(sessionId?: string): boolean {
    const targetSessionId = sessionId || this.currentSessionId;

    if (!targetSessionId) return false;

    // Get main_thread_id before deleting session
    const session = this.sessions.get(targetSessionId);
    const mainThreadId = session?.mainThreadId;

    this.sessions.delete(targetSessionId);

    if (this.currentSessionId === targetSessionId) {
      this.currentSessionId = undefined;
    }

    this.persistSessionsToStorage();

    // Sync with backend orchestrator (async, non-blocking)
    if (mainThreadId) {
      this.orchestrationService.endSession(mainThreadId).catch((error) => {
        console.error('[SESSION] Failed to sync end session with backend:', error);
      });
    }

    console.log(`[SESSION] Ended session: ${targetSessionId}`);
    return true;
  }

  /**
   * GET SESSION METADATA
   */
  public getSessionMetadata(sessionId?: string): SessionMetadata | null {
    const session = sessionId ? this.sessions.get(sessionId) : this.getCurrentSession();

    if (!session) return null;

    return {
      sessionId: session.sessionId,
      userId: session.userId,
      mainThreadId: session.mainThreadId,
      subThreadCount: session.subThreads.size,
      activeWindowCount: session.windowCount,
      createdAt: session.loginTime,
      lastUpdated: session.lastActivityTime,
    };
  }

  /**
   * GET ALL ACTIVE SESSIONS (for monitoring/debugging)
   */
  public getAllActiveSessions(): SessionMetadata[] {
    return Array.from(this.sessions.values()).map(session => ({
      sessionId: session.sessionId,
      userId: session.userId,
      mainThreadId: session.mainThreadId,
      subThreadCount: session.subThreads.size,
      activeWindowCount: session.windowCount,
      createdAt: session.loginTime,
      lastUpdated: session.lastActivityTime,
    }));
  }

  /**
   * RESTORE SESSION (for page reload/browser recovery)
   */
  public restoreSession(sessionId: string): UserSession | null {
    const session = this.sessions.get(sessionId);

    if (!session) {
      console.log(`[SESSION] Could not restore session: ${sessionId}`);
      return null;
    }

    this.currentSessionId = sessionId;
    session.lastActivityTime = new Date().toISOString();
    this.persistSessionsToStorage();

    console.log(`[SESSION] Restored session: ${sessionId}`);
    return session;
  }

  // ========================================================================
  // THREAD ID GENERATION (UI-side)
  // ========================================================================

  /**
   * Generate Main Thread ID
   * Format: main_{user_id}_{zone}_{uuid}_{timestamp}
   *
   * UUID segment ensures:
   * - Zero collision even if same user logs in at same millisecond
   * - Unpredictable IDs (security)
   * - Enterprise-grade uniqueness for distributed systems
   */
  private generateMainThreadId(userId: string, zone?: string): string {
    const sanitizedUserId = this.sanitizeForThreadId(userId);
    const zoneStr = (zone || 'DEFAULT').replace('-', '_');
    const uuidSegment = uuidv4().split('-')[0]; // 8 char UUID segment
    const timestamp = this.generateTimestamp();

    return `main_${sanitizedUserId}_${zoneStr}_${uuidSegment}_${timestamp}`;
  }

  /**
   * Generate Sub-Thread ID
   * Format: {workflow_type}_{main_ref}_{uuid}_{timestamp}
   *
   * UUID segment provides extra collision protection for:
   * - Same-millisecond requests
   * - Distributed systems
   * - Enterprise-grade isolation
   */
  private generateSubThreadId(mainThreadId: string, workflowType: string): string {
    const mainRef = mainThreadId.substring(mainThreadId.length - 12).replace(/_/g, '').substring(0, 8);
    const uuidSegment = uuidv4().split('-')[0]; // 8 char UUID segment
    const timestamp = this.generateTimestamp();

    return `${workflowType}_${mainRef}_${uuidSegment}_${timestamp}`;
  }

  /**
   * Generate Item Thread ID
   * Format: item_{wf_ref}_{item_type}_{hash}_{uuid}_{timestamp}
   *
   * UUID segment ensures uniqueness even for items with same name/number
   */
  private generateItemThreadId(
    subThreadId: string,
    itemType: 'instrument' | 'accessory',
    itemName: string,
    itemNumber: number
  ): string {
    const wfParts = subThreadId.split('_');
    const wfRef = wfParts[1] || 'wf';
    const itemTypeCode = itemType === 'instrument' ? 'inst' : 'acc';
    const itemHash = this.generateHash(`${itemName}_${itemNumber}`);
    const uuidSegment = uuidv4().split('-')[0]; // 8 char UUID segment
    const timestamp = this.generateTimestamp();

    return `item_${wfRef}_${itemTypeCode}_${itemHash}_${uuidSegment}_${timestamp}`;
  }

  /**
   * Generate Timestamp
   * Format: YYYYMMdd_HHMMSS_mmm
   */
  private generateTimestamp(): string {
    const now = new Date();
    const year = now.getFullYear();
    const month = String(now.getMonth() + 1).padStart(2, '0');
    const day = String(now.getDate()).padStart(2, '0');
    const hours = String(now.getHours()).padStart(2, '0');
    const minutes = String(now.getMinutes()).padStart(2, '0');
    const seconds = String(now.getSeconds()).padStart(2, '0');
    const ms = String(now.getMilliseconds()).padStart(3, '0');

    return `${year}${month}${day}_${hours}${minutes}${seconds}_${ms}`;
  }

  /**
   * Generate Short Hash
   * Used for item thread ID uniqueness
   */
  private generateHash(value: string, length: number = 8): string {
    let hash = 0;
    for (let i = 0; i < value.length; i++) {
      const char = value.charCodeAt(i);
      hash = ((hash << 5) - hash) + char;
      hash = hash & hash; // Convert to 32-bit integer
    }
    return Math.abs(hash).toString(16).substring(0, length);
  }

  /**
   * Sanitize String for Thread ID
   * Only alphanumeric + underscore
   */
  private sanitizeForThreadId(value: string): string {
    return value
      .replace(/[^a-zA-Z0-9_]/g, '')
      .replace(/\s+/g, '_')
      .substring(0, 32);
  }

  // ========================================================================
  // PERSISTENCE
  // ========================================================================

  /**
   * Persist Sessions to Browser Storage
   */
  private persistSessionsToStorage(): void {
    try {
      // Convert Map to serializable format
      const sessionsData = Array.from(this.sessions.entries()).map(([key, session]) => ({
        key,
        sessionId: session.sessionId,
        userId: session.userId,
        mainThreadId: session.mainThreadId,
        zone: session.zone,
        loginTime: session.loginTime,
        lastActivityTime: session.lastActivityTime,
        windowCount: session.windowCount,
        activeSubThreadId: session.activeSubThreadId,
        currentWorkflow: session.currentWorkflow,
        workflowLockedAt: session.workflowLockedAt,
        subThreads: Array.from(session.subThreads.entries()).map(([subKey, subThread]) => ({
          key: subKey,
          subThreadId: subThread.subThreadId,
          workflowType: subThread.workflowType,
          createdAt: subThread.createdAt,
          status: subThread.status,
          itemThreads: Array.from(subThread.itemThreads.entries()),
        })),
      }));

      const storage = this.storageType === 'localStorage' ? localStorage : sessionStorage;
      storage.setItem(this.sessionStorageKey, JSON.stringify(sessionsData));

      console.log(`[STORAGE] Persisted ${this.sessions.size} sessions`);
    } catch (error) {
      console.error('[STORAGE] Failed to persist sessions:', error);
    }
  }

  /**
   * Load Sessions from Browser Storage
   */
  private loadSessionsFromStorage(): void {
    try {
      const storage = this.storageType === 'localStorage' ? localStorage : sessionStorage;
      const data = storage.getItem(this.sessionStorageKey);

      if (!data) return;

      const sessionsData = JSON.parse(data);

      sessionsData.forEach((sessionData: any) => {
        const session: UserSession = {
          sessionId: sessionData.sessionId,
          userId: sessionData.userId,
          mainThreadId: sessionData.mainThreadId,
          zone: sessionData.zone,
          loginTime: sessionData.loginTime,
          lastActivityTime: sessionData.lastActivityTime,
          windowCount: sessionData.windowCount,
          activeSubThreadId: sessionData.activeSubThreadId,
          currentWorkflow: sessionData.currentWorkflow || null,
          workflowLockedAt: sessionData.workflowLockedAt,
          subThreads: new Map(
            sessionData.subThreads.map((subData: any) => [
              subData.key,
              {
                subThreadId: subData.subThreadId,
                workflowType: subData.workflowType,
                createdAt: subData.createdAt,
                status: subData.status,
                itemThreads: new Map(subData.itemThreads),
              },
            ])
          ),
        };

        this.sessions.set(sessionData.key, session);
      });

      console.log(`[STORAGE] Loaded ${this.sessions.size} sessions from storage`);
    } catch (error) {
      console.error('[STORAGE] Failed to load sessions:', error);
    }
  }

  /**
   * Clear All Sessions (for testing/debugging)
   */
  public clearAllSessions(): void {
    this.sessions.clear();
    this.currentSessionId = undefined;
    const storage = this.storageType === 'localStorage' ? localStorage : sessionStorage;
    storage.removeItem(this.sessionStorageKey);
    console.log('[STORAGE] Cleared all sessions');
  }
}

// Re-export orchestration service for convenience
export { sessionOrchestrationService };

// Export singleton getter for backward compatibility
export const getSessionManager = (): SessionManager => {
  return SessionManager.getInstance();
};

export default SessionManager;
