// api.ts - No changes are necessary here
import axios from "axios";
import { getSessionManager, WorkflowType as SessionWorkflowType } from "../../services/SessionManager";
import {
  ValidationResult,
  AnalysisResult,
  RequirementSchema,
  UserCredentials,
  ChatMessage,
  IntentClassificationResult,
  WorkflowRoutingResult,
  AgentResponse,
  AdvancedParametersResult,
  AdvancedParametersSelection,
  InstrumentIdentificationResult,
  AnalysisImageResult,
  ModifyInstrumentsRequest,
} from "./types";





// Use environment variable for production API URL, fallback to relative path (proxy) for dev
export const BASE_URL = import.meta.env.VITE_API_URL || "";
axios.defaults.baseURL = BASE_URL;
axios.defaults.withCredentials = true;

interface User {
  username: string;
  name: string;
  email: string;
  // Add other user properties if needed
}

interface Vendor {
  name: string;
  logoUrl: string | null;
}

interface PendingUser {
  id: number;
  username: string;
  email: string;
}

// --- NEW INTERFACES FOR PDF SEARCH ---
interface PdfSearchResult {
  title: string;
  url: string;
  snippet: string;
}
interface PriceReviewResponse {
  productName: string;
  results: Array<{
    price: string | null;
    reviews: number | null;
    source: string | null;
  }>;
}

/**
 * Converts snake_case or kebab-case keys to camelCase recursively.
 */
function convertKeysToCamelCase(obj: any): any {
  if (Array.isArray(obj)) {
    return obj.map((v) => convertKeysToCamelCase(v));
  } else if (obj !== null && typeof obj === "object") {
    return Object.keys(obj).reduce((acc: Record<string, any>, key: string) => {
      const camelKey = key.replace(/([-_][a-z])/g, (group) =>
        group.toUpperCase().replace("-", "").replace("_", "")
      );
      acc[camelKey] = convertKeysToCamelCase(obj[key]);
      return acc;
    }, {});
  }
  return obj;
}

/**
 * Normalizes user input by removing backslashes, underscores, and hyphens and converting to lowercase.
 */
function normalizeUserInput(input: string): string {
  return input.replace(/[\\_-]/g, "").toLowerCase();
}

/**
 * Registers a new user (status is set to 'pending' on backend).
 */
export const signup = async (
  credentials: UserCredentials
): Promise<{ message: string }> => {
  try {
    const response = await axios.post(`/register`, credentials);
    return response.data;
  } catch (error: any) {
    console.error("Signup error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Signup failed");
  }
};

/**
 * Logs a user in; will fail if user status is not 'active'.
 */
export const login = async (
  credentials: Omit<UserCredentials, "email">
): Promise<{ message: string; user: User }> => {
  try {
    const response = await axios.post(`/login`, credentials);
    return convertKeysToCamelCase(response.data) as { message: string; user: User };
  } catch (error: any) {
    console.error("Login error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Login failed");
  }
};

/**
 * Logs out the current user.
 */
export const logout = async (): Promise<{ message: string }> => {
  try {
    const response = await axios.post(`/logout`);
    return response.data;
  } catch (error: any) {
    console.error("Logout error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Logout failed");
  }
};

/**
 * Updates the user's profile (first name, last name, username).
 */
export const updateProfile = async (
  data: { first_name?: string; last_name?: string; username?: string; company?: string; location?: string }
): Promise<any> => {
  try {
    const response = await axios.post(`/api/update_profile`, data, {
      headers: {
        'Content-Type': 'application/json'
      }
    });
    return convertKeysToCamelCase(response.data);
  } catch (error: any) {
    console.error("Profile update error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Profile update failed");
  }
};



export const uploadStandardsDocument = async (file: File): Promise<any> => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    const response = await axios.post('/api/upload-standards-file', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return convertKeysToCamelCase(response.data);
  } catch (error: any) {
    console.error("Standards upload error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Standards upload failed");
  }
};





// =============================================================================
// STRATEGY DOCUMENT APIs
// =============================================================================







/**
 * Checks if the user is authenticated, returning user info or null if not authenticated.
 */
export const checkAuth = async (): Promise<{ user: User } | null> => {
  try {
    const response = await axios.get(`/user`);
    return convertKeysToCamelCase(response.data) as { user: User };
  } catch (error: any) {
    if (error.response && error.response.status === 401) {
      console.log("Authentication check failed: User is not logged in.");
      return null;
    }
    console.error("Unexpected error during authentication check:", error);
    return null;
  }
};

/**
 * Fetches the list of vendors with their logo URLs.
 * @param vendorNames - Array of vendor names from analysis results (optional)
 */
export const getVendors = async (vendorNames?: string[]): Promise<Vendor[]> => {
  try {
    // Build query string with vendor names if provided
    const params = vendorNames && vendorNames.length > 0
      ? { vendors: vendorNames.join(',') }
      : {};

    const response = await axios.get(`/vendors`, { params });
    const vendors = convertKeysToCamelCase(response.data.vendors) as Vendor[];
    return vendors;
  } catch (error: any) {
    console.error("Failed to fetch vendors:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Failed to fetch vendors");
  }
};

/**
 * Fetches the submodel to model series mapping.
 * This is used to map analysis results (submodel names) to images (model series names).
 */
export const getSubmodelMapping = async (): Promise<Record<string, string>> => {
  try {
    const response = await axios.get(`/submodel-mapping`);
    return convertKeysToCamelCase(response.data.mapping) as Record<string, string>;
  } catch (error: any) {
    console.error("Failed to fetch submodel mapping:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Failed to fetch submodel mapping");
  }
};

// Track if validation has been called at least once per session to help backend
const validationTracker = new Map<string, boolean>();

/**
 * Initialize a new search session
 */
export const initializeNewSearch = async (searchSessionId: string): Promise<void> => {
  try {
    // Clear validation tracker for fresh start
    validationTracker.set(searchSessionId, false);

    await axios.post('/new-search', {
      search_session_id: searchSessionId,
      reset: true
    });
    console.log(`[NEW_SEARCH] Initialized search session: ${searchSessionId}`);
  } catch (error: any) {
    console.error("Failed to initialize new search session:", error.response?.data || error.message);
    // Don't throw error - continue with search even if initialization fails
  }
};

/**
 * Clean up validation tracker for a session (call when tab closes)
 */
export const clearSessionValidationState = (searchSessionId: string): void => {
  validationTracker.delete(searchSessionId);
};

/**
 * Validates user input requirements.
 */
export const validateRequirements = async (
  userInput: string,
  productType?: string,
  searchSessionId?: string,
  currentStep?: string // ✅ Add current step to determine if this is a repeat validation
): Promise<ValidationResult> => {
  try {
    const normalizedInput = normalizeUserInput(userInput);

    // Check if this session has validated before
    const sessionId = searchSessionId || 'default';
    const hasSessionValidated = validationTracker.get(sessionId) || false;

    // ✅ If user is in awaitMissingInfo step, it means they've already seen the validation alert once
    // So this is a repeat validation and is_repeat should be true
    const isRepeat = hasSessionValidated || currentStep === "awaitMissingInfo";

    const payload: any = {
      user_input: normalizedInput,
      requirements: normalizedInput, // ✅ Include requirements for modern endpoint
      is_repeat: isRepeat, // ✅ tell backend if this is a repeat validation
      reset: false, // By default do not reset session state on validation
    };

    if (productType) {
      payload.product_type = productType; // 🚀 Only include if detected
      payload.productType = productType; // Handle both cases
    }

    if (searchSessionId) {
      payload.search_session_id = searchSessionId; // 🚀 Include search session ID for independent searches
      payload.sessionId = searchSessionId; // Handle both cases
    }

    const response = await axios.post(`/validate`, payload);

    validationTracker.set(sessionId, true); // ✅ mark that this session has validated at least once

    const data = convertKeysToCamelCase(response.data);

    // ✅ Handle new endpoint structure which nests result in 'validation'
    if (data.validation) {
      return data.validation as ValidationResult;
    }

    return data as ValidationResult;
  } catch (error: any) {
    console.error("Validation error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Validation failed");
  }
};





/**
 * Run final product analysis (Steps 4-5: Vendor Analysis + Ranking)
 *
 * This calls the workflow.run_analysis_only() method after requirements are collected.
 * It performs vendor matching and product ranking.
 *
 * @param structuredRequirements Complete requirements object with mandatory, optional, and advanced params
 * @param productType Product type detected from validation
 * @param schema Product schema (optional)
 * @param sessionId Session tracking ID (optional)
 * @returns Analysis result with vendor matches and ranked products
 */
export const runFinalProductAnalysis = async (
  structuredRequirements: {
    productType?: string;
    mandatoryRequirements?: Record<string, any>;
    optionalRequirements?: Record<string, any>;
    selectedAdvancedParams?: Record<string, any>;
  },
  productType: string,
  schema?: any,
  sessionId?: string
): Promise<AnalysisResult> => {
  try {
    console.log('[RUN_ANALYSIS] Calling /api/agentic/run-analysis', {
      productType,
      hasStructuredReqs: !!structuredRequirements,
      sessionId
    });

    const response = await axios.post(`/api/agentic/run-analysis`, {
      structured_requirements: structuredRequirements,
      product_type: productType,
      schema: schema,
      session_id: sessionId
    });

    if (!response.data.success) {
      throw new Error(response.data.error || 'Analysis failed');
    }

    const analysisData = response.data.data;

    // CRITICAL: Validate analysisData before proceeding
    if (!analysisData) {
      console.error('[RUN_ANALYSIS] Backend returned empty data object');
      throw new Error('Backend returned empty analysis data');
    }

    console.log('[RUN_ANALYSIS] Analysis complete', {
      totalRanked: analysisData.totalRanked,
      exactMatches: analysisData.exactMatchCount,
      approximateMatches: analysisData.approximateMatchCount
    });

    // DEBUG: Log full structure to identify field naming issues
    console.log('[RUN_ANALYSIS] DEBUG: Raw analysisData keys:', Object.keys(analysisData));
    console.log('[RUN_ANALYSIS] DEBUG: overallRanking keys:', Object.keys(analysisData.overallRanking || {}));
    if (analysisData.overallRanking?.rankedProducts?.length > 0) {
      console.log('[RUN_ANALYSIS] DEBUG: First rankedProduct:', analysisData.overallRanking.rankedProducts[0]);
      console.log('[RUN_ANALYSIS] DEBUG: First product fields:', Object.keys(analysisData.overallRanking.rankedProducts[0]));
    } else {
      console.warn('[RUN_ANALYSIS] WARNING: No ranked products in response!');
    }

    // Transform to expected AnalysisResult format
    const result: AnalysisResult = {
      productType: productType,
      vendorAnalysis: analysisData.vendorAnalysis || {
        vendorMatches: [],
        vendorRunDetails: [],
        totalMatches: 0,
        vendorsAnalyzed: 0
      },
      overallRanking: analysisData.overallRanking || {
        rankedProducts: []
      },
      topRecommendation: analysisData.topRecommendation,
      totalMatches: analysisData.totalRanked || 0,
      exactMatchCount: analysisData.exactMatchCount || 0,
      approximateMatchCount: analysisData.approximateMatchCount || 0
    };

    return result;
  } catch (error: any) {
    console.error("[RUN_ANALYSIS] Error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Final analysis failed");
  }
};

/**
 * Gets requirement schema for the given product type.
 */
export const getRequirementSchema = async (
  productType: string
): Promise<RequirementSchema> => {
  try {
    if (!productType || productType.trim() === "") {
      return {
        default: { mandatory: {}, optional: {} },
        mandatoryRequirements: {},
        optionalRequirements: {},
      } as RequirementSchema;
    }
    const response = await axios.get(`/schema`, {
      params: { product_type: productType },
    });
    return convertKeysToCamelCase(response.data) as RequirementSchema;
  } catch (error: any) {
    console.error("Schema fetch error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Schema fetch failed");
  }
};

/**
 * Processes additional requirements and returns explanations.
 */
export const additionalRequirements = async (
  productType: string,
  userInput: string
): Promise<{ explanation: string; providedRequirements: any }> => {
  try {
    const response = await axios.post(`/additional_requirements`, {
      product_type: productType,
      user_input: userInput,
    });
    return convertKeysToCamelCase(response.data);
  } catch (error: any) {
    console.error("Additional requirements error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Failed to process additional requirements.");
  }
};

/**
 * Structures the requirements using backend logic.
 */
export const structureRequirements = async (fullInput: string): Promise<any> => {
  try {
    const normalizedInput = normalizeUserInput(fullInput);
    const response = await axios.post(`/structure_requirements`, { full_input: normalizedInput });
    return convertKeysToCamelCase(response.data);
  } catch (error: any) {
    console.error("Requirement structuring error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Requirement structuring failed");
  }
};

/**
 * Discovers advanced parameters from top vendors for a product type (Flask endpoint)
 */
export const discoverAdvancedParameters = async (productType: string, searchSessionId?: string): Promise<any> => {
  try {
    const payload: any = {
      product_type: productType
    };

    if (searchSessionId) {
      payload.search_session_id = searchSessionId;
    }

    const response = await axios.post(`/api/advanced_parameters`, payload);
    return convertKeysToCamelCase(response.data);
  } catch (error: any) {
    console.error("Advanced parameters discovery error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Failed to discover advanced parameters");
  }
};

// ==================== AGENTIC TOOL WRAPPER APIS ====================

/**
 * Call Agentic Validation Tool Wrapper
 *
 * Standalone validation endpoint that detects product type, loads/generates schema
 * (with PPI workflow if needed), and validates user requirements.
 *
 * @param userInput User's requirements description
 * @param productType Optional: Expected product type hint
 * @param sessionId Optional: Session tracking ID
 * @param enablePpi Optional: Enable PPI workflow (default: true)
 * @returns Validation result with product type, schema, requirements, etc.
 */
export const callAgenticValidate = async (
  userInput: string,
  productType?: string,
  sessionId?: string,
  enablePpi: boolean = true
): Promise<any> => {
  try {
    const payload: any = {
      user_input: userInput,
      enable_ppi: enablePpi
    };

    if (productType) {
      payload.product_type = productType;
    }

    if (sessionId) {
      payload.session_id = sessionId;
    }

    console.log('[AGENTIC_VALIDATE] Starting validation:', {
      inputPreview: userInput.substring(0, 50),
      productType,
      sessionId,
      enablePpi
    });

    const response = await axios.post('/api/agentic/validate', payload);

    if (!response.data.success) {
      throw new Error(response.data.error || "Validation failed");
    }

    const result = convertKeysToCamelCase(response.data.data);

    console.log('[AGENTIC_VALIDATE] Validation complete:', {
      productType: result.productType,
      isValid: result.isValid,
      ppiUsed: result.ppiWorkflowUsed,
      missingFields: result.missingFields
    });

    return result;
  } catch (error: any) {
    console.error("[AGENTIC_VALIDATE] Error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Validation failed");
  }
};

/**
 * Call Agentic Advanced Parameters Tool Wrapper
 *
 * Standalone advanced parameters discovery endpoint that discovers latest
 * advanced specifications with series numbers from top vendors.
 *
 * @param productType Product type to discover parameters for (required)
 * @param sessionId Optional: Session tracking ID
 * @returns Advanced parameters result with unique specifications
 */
export const callAgenticAdvancedParameters = async (
  productType: string,
  sessionId?: string
): Promise<any> => {
  try {
    if (!productType || !productType.trim()) {
      throw new Error("product_type is required");
    }

    const payload: any = {
      product_type: productType.trim()
    };

    if (sessionId) {
      payload.session_id = sessionId;
    }

    console.log('[AGENTIC_ADVANCED_PARAMS] Discovering for:', productType);

    const response = await axios.post('/api/agentic/advanced-parameters', payload);

    if (!response.data.success) {
      throw new Error(response.data.error || "Advanced parameters discovery failed");
    }

    const result = convertKeysToCamelCase(response.data.data);

    console.log('[AGENTIC_ADVANCED_PARAMS] Discovery complete:', {
      productType: result.productType,
      totalUnique: result.totalUniqueSpecifications,
      existingFiltered: result.existingSpecificationsFiltered,
      specificationsFound: result.uniqueSpecifications?.length || 0
    });

    return result;
  } catch (error: any) {
    console.error("[AGENTIC_ADVANCED_PARAMS] Error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Advanced parameters discovery failed");
  }
};

/**
 * Call Agentic Sales Agent Tool Wrapper
 *
 * Generates conversational AI responses for all workflow tools (validation, advanced parameters, etc.)
 * This is the universal response generator that creates user-facing messages.
 *
 * @param step Current workflow step (e.g., "initialInput", "awaitMissingInfo", "awaitAdditionalAndLatestSpecs")
 * @param userMessage User's message or input
 * @param dataContext Context data including product type, schema, requirements, etc.
 * @param sessionId Optional: Session tracking ID
 * @param intent Optional: Intent type ("workflow" or "knowledgeQuestion")
 * @param saveImmediately Optional: Whether to save state immediately
 * @returns Sales agent response with content, nextStep, and discoveredParameters
 */
export const callAgenticSalesAgent = async (
  step: string,
  userMessage: string,
  dataContext: any,
  sessionId?: string,
  intent: string = "workflow",
  saveImmediately: boolean = false
): Promise<any> => {
  try {
    const payload: any = {
      step: step,
      user_message: userMessage,
      data_context: dataContext,
      intent: intent,
      save_immediately: saveImmediately
    };

    if (sessionId) {
      payload.session_id = sessionId;
    }

    console.log('[AGENTIC_SALES_AGENT] Generating response:', {
      step,
      intent,
      sessionId,
      productType: dataContext?.product_type || dataContext?.productType,
      hasSchema: !!(dataContext?.schema)
    });

    const response = await axios.post('/api/agentic/sales-agent', payload);

    if (!response.data.success) {
      throw new Error(response.data.error || "Sales agent response generation failed");
    }

    const result = convertKeysToCamelCase(response.data.data);

    console.log('[AGENTIC_SALES_AGENT] Response generated:', {
      nextStep: result.nextStep,
      hasDiscoveredParams: !!(result.discoveredParameters && result.discoveredParameters.length > 0),
      contentPreview: result.content?.substring(0, 100)
    });

    return result;
  } catch (error: any) {
    console.error("[AGENTIC_SALES_AGENT] Error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Failed to generate sales agent response");
  }
};

/**
 * Processes user selection of advanced parameters
 */
export const addAdvancedParameters = async (
  productType: string,
  userInput: string,
  availableParameters: string[]
): Promise<any> => {
  try {
    const response = await axios.post(`/api/add_advanced_parameters`, {
      product_type: productType,
      user_input: userInput,
      available_parameters: availableParameters
    });
    return convertKeysToCamelCase(response.data);
  } catch (error: any) {
    console.error("Add advanced parameters error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Failed to process advanced parameters");
  }
};

/**
 * Fetches a human-readable description for a schema field.
 */
export const getFieldDescription = async (
  field: string,
  productType: string | null
): Promise<{ description: string }> => {
  try {
    const response = await axios.post(`/get_field_description`, { field, product_type: productType });
    return convertKeysToCamelCase(response.data);
  } catch (error: any) {
    console.error("Field description fetch error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Failed to fetch field description.");
  }
};

/**
 * Fetches ALL field descriptions and values in a single batch request.
 * This is much faster than calling getFieldDescription for each field individually.
 * 
 * @param fields Array of field paths (e.g., ["Performance.accuracy", "Electrical.outputSignal"])
 * @param productType Product type for context-specific values
 * @returns Map of field paths to their values
 */
export const getAllFieldDescriptions = async (
  fields: string[],
  productType: string | null
): Promise<Record<string, string>> => {
  try {
    if (!fields.length) return {};

    console.log(`[BATCH_FIELDS] Fetching ${fields.length} field values for ${productType}`);

    const response = await axios.post(`/api/get_all_field_descriptions`, {
      fields,
      product_type: productType
    });

    const result = convertKeysToCamelCase(response.data);

    // Convert the response to a simple field -> value map
    const fieldMap: Record<string, string> = {};
    const fieldValues = result.fieldValues || {};

    for (const [fieldPath, data] of Object.entries(fieldValues)) {
      const fieldData = data as any;
      fieldMap[fieldPath] = fieldData?.value || "";
    }

    console.log(`[BATCH_FIELDS] Received ${result.fieldsPopulated}/${result.totalFields} values`);

    return fieldMap;
  } catch (error: any) {
    console.error("Batch field description fetch error:", error.response?.data || error.message);
    // Return empty map on error - don't throw, allow graceful degradation
    return {};
  }
};

/**
 * Fetches the list of users pending approval (admin only).
 */
export const getPendingUsers = async (): Promise<PendingUser[]> => {
  try {
    const response = await axios.get(`/admin/pending_users`);
    const users = convertKeysToCamelCase(response.data.pendingUsers) as PendingUser[];
    return users;
  } catch (error: any) {
    console.error("Failed to fetch pending users:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Failed to fetch pending users");
  }
};

/**
 * Admin approves or rejects a user.
 * @param userId - ID of the user to approve/reject.
 * @param action - "approve" or "reject".
 */
export const approveOrRejectUser = async (
  userId: number,
  action: "approve" | "reject"
): Promise<{ message: string }> => {
  try {
    const response = await axios.post(`/admin/approve_user`, { user_id: userId, action });
    return response.data;
  } catch (error: any) {
    console.error("Failed to update user status:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Failed to update user status");
  }
};


// ====================================================================
// === NEW: LLM Sales Agent API call ===
// ====================================================================

/**
 * Classifies user intent and determines next workflow step
 * Uses SessionManager for workflow state (cross-tab persistence)
 */

// ============================================================================
// WORKFLOW STATE HELPERS (delegates to SessionManager)
// ============================================================================

export const getCurrentWorkflow = (): string | null => {
  return getSessionManager().getWorkflow();
};

export const setCurrentWorkflow = (workflow: string | null): void => {
  getSessionManager().setWorkflow(workflow as SessionWorkflowType);
};

export const clearWorkflow = (): void => {
  getSessionManager().clearWorkflow();
};




/**
 * Classifies user query and determines which workflow to route to.
 * 
 * Routes to:
 * - solution: Complex engineering systems requiring multiple instruments
 * - instrument_identifier: Single product requirements
 * - engenie_chat: Questions about products, standards, vendors (opens EnGenie Chat)
 * - out_of_domain: Unrelated queries (rejected with helpful message)
 * 
 * @param query User query from UI textarea
 * @param context Optional context (current_step, conversation history)
 * @returns WorkflowRoutingResult with target_workflow and routing details
 */
export const classifyRoute = async (
  query: string,
  context?: { current_step?: string; context?: string },
  searchSessionId?: string
): Promise<WorkflowRoutingResult> => {
  try {
    const sessionManager = getSessionManager();
    const currentWorkflow = sessionManager.getWorkflow();

    const payload: Record<string, any> = {
      query,
      // Always pass session ID from SessionManager if not provided
      search_session_id: searchSessionId || sessionManager.getMainThreadId() || 'default',
    };

    if (context) {
      payload.context = context;
    }

    // Pass workflow as hint for backend validation
    if (currentWorkflow) {
      payload.workflow_hint = currentWorkflow;
    }

    console.log('[CLASSIFY_ROUTE] Request:', {
      query: query.substring(0, 50),
      session: payload.search_session_id?.substring(0, 16),
      hint: currentWorkflow
    });

    const response = await axios.post('/api/agentic/classify-route', payload);

    if (!response.data.success) {
      throw new Error(response.data.error || 'Classification failed');
    }

    const result = response.data.data as WorkflowRoutingResult;

    // Update workflow from backend decision
    if (result.target_workflow && result.target_workflow !== 'out_of_domain') {
      sessionManager.setWorkflow(result.target_workflow as SessionWorkflowType);
    }

    // Refresh TTL on activity
    sessionManager.refreshWorkflowTTL();

    console.log('[CLASSIFY_ROUTE] Routing decision:', {
      target_workflow: result.target_workflow,
      intent: result.intent,
      confidence: result.confidence
    });

    return result;
  } catch (error: any) {
    console.error('[CLASSIFY_ROUTE] Error:', error.response?.data || error.message);

    // Error handling with retryable flag
    const isRetryable = !error.response || error.response.status >= 500;

    return {
      query: query,
      target_workflow: 'instrument_identifier',
      intent: 'error',
      confidence: 0,
      reasoning: `Fallback due to classification error: ${error.message}`,
      is_solution: false,
      solution_indicators: [],
      extracted_info: { error: true, retryable: isRetryable },
      classification_time_ms: 0,
      timestamp: new Date().toISOString(),
      reject_message: null
    };
  }
};

export const uploadPdfFile = async (file: File): Promise<any> => {
  try {
    const formData = new FormData();
    formData.append('file', file);
    const response = await axios.post('/upload', formData, {
      headers: { 'Content-Type': 'multipart/form-data' }
    });
    return convertKeysToCamelCase(response.data);
  } catch (error: any) {
    console.error("File upload error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "File upload failed");
  }
};


// ====================================================================
// === UPDATES FOR PDF SEARCH, VIEW, AND URL UPLOAD ===
// ====================================================================

/**
 * Searches for PDF files based on a user query.
 * @param query The search term.
 * @returns A promise that resolves with a list of search results.
 */
export const searchPdfs = async (query: string): Promise<PdfSearchResult[]> => {
  try {
    const response = await axios.get(`/api/search_pdfs`, { params: { query } });
    const results = convertKeysToCamelCase(response.data.results) as PdfSearchResult[];
    return results;
  } catch (error: any) {
    console.error("PDF search error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "PDF search failed");
  }
};

/**
 * Returns the URL to view a PDF. The backend handles the file serving.
 * The client-side code should open this URL in a new tab or iframe.
 * @param pdfUrl The URL of the PDF to view.
 * @returns The backend endpoint URL to view the PDF.
 */
export const viewPdf = (pdfUrl: string): string => {
  // Use encodeURIComponent to ensure the URL is safe for a query parameter
  return `${BASE_URL}/api/view_pdf?url=${encodeURIComponent(pdfUrl)}`;
};

/**
 * Uploads a PDF to the backend for analysis by providing its URL.
 * @param url The URL of the PDF file.
 * @returns A promise that resolves with the analysis result.
 */
export const uploadPdfFromUrl = async (url: string): Promise<any> => {
  try {
    const response = await axios.post(`/api/upload_pdf_from_url`, { url });
    return convertKeysToCamelCase(response.data);
  } catch (error: any) {
    console.error("URL-based PDF upload error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "URL-based PDF upload failed");
  }
};


/**
 * Fetches price and reviews dynamically from the backend for a given product.
 * @param productName - Name of the product.
 * @returns An object with an array of results, each containing price, reviews, and source.
 */
export const getProductPriceReview = async (
  productName: string
): Promise<PriceReviewResponse> => {
  try {
    if (!productName) {
      throw new Error("productName is required");
    }

    const params: Record<string, string> = { productName };

    const response = await axios.get("/api/get-price-review", { params });

    // The backend now returns a structured object with a 'results' array.
    // The response data matches the PriceReviewResponse interface.
    return convertKeysToCamelCase(response.data) as PriceReviewResponse;

  } catch (error: any) {
    console.error(
      `Failed to fetch price/review for product ${productName}:`,
      error.response?.data || error.message
    );
    // Return a default object with an empty results array on failure.
    return { productName: productName, results: [] };
  }
};

// --- NEW: Function to handle analysis feedback ---

/**
 * Submits user feedback (thumbs up/down and a comment) and gets an LLM-generated response.
 * @param feedbackType - 'positive' for thumbs up, 'negative' for thumbs down. Can be null if only a comment is provided.
 * @param comment - Optional text feedback from the user.
 * @returns A promise that resolves with the LLM's response string.
 */
export const submitFeedback = async (
  feedbackType: "positive" | "negative" | null,
  comment?: string,
  projectId?: string
): Promise<string> => {
  try {
    const body: any = { feedbackType, comment: comment || "" };
    if (projectId) body.projectId = projectId;

    const response = await axios.post("/api/feedback", body);
    // The backend returns a JSON object with a 'response' field.
    return response.data.response;
  } catch (error: any) {
    console.error(
      "Failed to submit feedback:",
      error.response?.data || error.message
    );
    throw new Error(
      error.response?.data?.error || "Submitting feedback failed"
    );
  }
};

// ====================================================================
// === PRIMARY CONVERSATION API CALL ===
// ====================================================================

/**
 * Identifies instruments from user requirements using the AGENTIC workflow.
 * 
 * This function calls the agentic instrument-identifier endpoint which:
 * 1. Classifies user intent
 * 2. Identifies instruments and accessories
 * 3. Aggregates RAG data (strategy, standards, inventory)
 * 4. Generates sample_input for each item (for product search)
 * 5. Returns items in awaiting_selection state for user to select
 * 
 * After user selects items, call callAgenticProductSearch() to trigger Product Search Workflow.
 * 
 * @param requirements User's requirements text
 * @param currentInstruments Optional - existing instruments list for modification detection
 * @param currentAccessories Optional - existing accessories list for modification detection
 * @param sessionId Optional - session ID for workflow tracking
 * @returns A promise that resolves with identified instruments and sample_inputs
 */

export const identifyInstruments = async (
  requirements: string,
  currentInstruments?: any[],
  currentAccessories?: any[],
  sessionId?: string
): Promise<InstrumentIdentificationResult> => {
  try {
    const payload: any = {
      message: requirements,  // Agentic API uses 'message' not 'requirements'
    };

    // Include session ID if provided
    if (sessionId) {
      payload.session_id = sessionId;
    }

    // If existing data is provided, include it for modification detection
    if (currentInstruments && currentInstruments.length > 0) {
      payload.current_instruments = currentInstruments;
    }
    if (currentAccessories && currentAccessories.length > 0) {
      payload.current_accessories = currentAccessories;
    }

    console.log('[AGENTIC] Calling /api/agentic/instrument-identifier');
    const response = await axios.post(`/api/agentic/instrument-identifier`, payload);

    // Agentic API returns { success, data, tags }
    const responseData = response.data;

    if (!responseData.success) {
      throw new Error(responseData.error || "Agentic workflow failed");
    }

    // Extract the actual result from data.result or data directly
    const result = responseData.data?.result || responseData.data || {};

    console.log('[AGENTIC] Instrument identifier response:', {
      success: responseData.success,
      hasItems: !!(result.response_data?.items || result.instruments),
      awaitingSelection: result.response_data?.awaiting_selection
    });

    // Convert the agentic response to match the expected format
    // Agentic returns: { response, response_data: { items, awaiting_selection, ... } }
    // Frontend expects: { instruments, accessories, responseType, message, projectName }

    const agenticResponse = result.response_data || result;
    const items = agenticResponse.items || [];

    // Separate items into instruments and accessories
    const instruments = items
      .filter((item: any) => item.type === 'instrument')
      .map((item: any) => ({
        category: item.category || '',
        productName: item.name || '',
        quantity: item.quantity || 1,
        specifications: item.specifications || {},
        sampleInput: item.sample_input || item.sampleInput || '',
        imageUrl: item.imageUrl || item.image_url || '',  // Include image URL from backend
      }));

    const accessories = items
      .filter((item: any) => item.type === 'accessory')
      .map((item: any) => ({
        category: item.category || '',
        accessoryName: item.name || '',
        quantity: item.quantity || 1,
        specifications: item.specifications || {},
        sampleInput: item.sample_input || item.sampleInput || '',
        imageUrl: item.imageUrl || item.image_url || '',  // Include image URL from backend
      }));

    // If no items from agentic format, try legacy format
    const finalInstruments = instruments.length > 0 ? instruments :
      (result.instruments || agenticResponse.instruments || []).map((inst: any) => convertKeysToCamelCase(inst));
    const finalAccessories = accessories.length > 0 ? accessories :
      (result.accessories || agenticResponse.accessories || []).map((acc: any) => convertKeysToCamelCase(acc));

    return {
      instruments: finalInstruments,
      accessories: finalAccessories,
      responseType: agenticResponse.response_type || 'requirements',
      message: result.response || agenticResponse.response || '',
      projectName: agenticResponse.project_name || 'Project',
      // Include agentic-specific fields
      awaitingSelection: agenticResponse.awaiting_selection || false,
      items: items,
      threadId: agenticResponse.thread_id,
    } as InstrumentIdentificationResult;

  } catch (error: any) {
    console.error("[AGENTIC] Instrument identification error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Failed to identify instruments");
  }
};

// ====================================================================
// === SOLUTION WORKFLOW API CALL ===
// ====================================================================

/**
 * Result interface for solution workflow
 */
export interface SolutionWorkflowResult {
  success: boolean;
  responseType: 'solution' | 'requirements' | 'greeting' | 'question' | 'error';
  message?: string;
  instruments: any[];
  accessories: any[];
  projectName?: string;
  awaitingSelection?: boolean;
  items?: any[];
  solutionContext?: {
    solutionName?: string;
    industry?: string;
    processType?: string;
    keyParameters?: Record<string, string>;
  };
  fieldDescriptions?: Record<string, string>;
  // Redirect fields - set when the solution deep agent determines input is out-of-context
  shouldRedirect?: boolean;
  routedTo?: string;
  intentType?: string;
}

/**
 * Calls the Solution Workflow for complex engineering challenges.
 * 
 * This function is used for inputs that describe complete measurement/control systems
 * requiring multiple instruments (e.g., reactor instrumentation, distillation column setup).
 * 
 * The solution workflow:
 * 1. Analyzes the solution context (industry, process type, parameters)
 * 2. Identifies all required instruments and accessories
 * 3. Generates sample_input for each item for subsequent product search
 * 
 * @param requirements User's solution/challenge description
 * @param sessionId Optional session ID for workflow tracking
 * @returns A promise that resolves with the solution analysis and identified items
 */
export const callSolutionWorkflow = async (
  requirements: string,
  sessionId?: string
): Promise<SolutionWorkflowResult> => {
  try {
    const payload: any = {
      message: requirements,
    };

    if (sessionId) {
      payload.session_id = sessionId;
    }

    console.log('[SOLUTION_WORKFLOW] Calling /api/agentic/solution');
    const response = await axios.post(`/api/agentic/solution`, payload);

    const responseData = response.data;

    if (!responseData.success) {
      throw new Error(responseData.error || "Solution workflow failed");
    }

    // Extract the result
    const result = responseData.data?.result || responseData.data || {};
    const agenticResponse = result.response_data || result;

    // Check if the solution deep agent is redirecting this input to another workflow
    if (agenticResponse.should_redirect) {
      console.log('[SOLUTION_WORKFLOW] Backend redirecting to:', agenticResponse.routed_to);
      return {
        success: true,
        responseType: 'solution',
        message: result.response || '',
        instruments: [],
        accessories: [],
        items: [],
        shouldRedirect: true,
        routedTo: agenticResponse.routed_to || 'chat',
        intentType: agenticResponse.intent_type || '',
      };
    }

    const items = agenticResponse.items || [];

    // Separate items into instruments and accessories
    const instruments = items
      .filter((item: any) => item.type === 'instrument')
      .map((item: any) => ({
        category: item.category || '',
        productName: item.name || '',
        quantity: item.quantity || 1,
        specifications: item.specifications || {},
        sampleInput: item.sample_input || item.sampleInput || '',
        purpose: item.purpose || '',
        imageUrl: item.imageUrl || item.image_url || '',  // Include image URL from backend
      }));

    const accessories = items
      .filter((item: any) => item.type === 'accessory')
      .map((item: any) => ({
        category: item.category || '',
        accessoryName: item.name || '',
        quantity: item.quantity || 1,
        specifications: item.specifications || {},
        sampleInput: item.sample_input || item.sampleInput || '',
        relatedInstrument: item.related_instrument || '',
        imageUrl: item.imageUrl || item.image_url || '',  // Include image URL from backend
      }));

    console.log('[SOLUTION_WORKFLOW] Success:', {
      itemCount: items.length,
      instruments: instruments.length,
      accessories: accessories.length
    });

    return {
      success: true,
      responseType: 'solution',
      message: result.response || agenticResponse.response || '',
      instruments,
      accessories,
      projectName: agenticResponse.solution_name || agenticResponse.project_name || 'Solution Project',
      awaitingSelection: agenticResponse.awaiting_selection || true,
      items,
      solutionContext: agenticResponse.solution_context,
      fieldDescriptions: agenticResponse.field_descriptions || agenticResponse.fieldDescriptions || {},
    };

  } catch (error: any) {
    console.error("[SOLUTION_WORKFLOW] Error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Failed to run solution workflow");
  }
};

// ====================================================================
// === ENGENIE CHAT API ===
// ====================================================================

/**
 * EnGenie Chat result interface
 */
export interface EnGenieChatResult {
  success: boolean;
  response_text: string;
  citations: Array<{ source: string; content: string }>;
  source_type: string;
  confidence: number;
  is_validated: boolean;
  error?: string;
}

/**
 * Calls the EnGenie Chat workflow for greetings, knowledge questions, and general chat.
 * This is the conversational AI component that handles non-product-search interactions.
 * 
 * @param message User's message (greeting, question, chitchat)
 * @param sessionId Optional session ID for conversation context
 * @returns A promise that resolves with the chat response
 */
export const callEnGenieChat = async (
  message: string,
  sessionId?: string
): Promise<EnGenieChatResult> => {
  try {
    console.log('[ENGENIE_CHAT] Calling Product Info API...');
    console.log('[ENGENIE_CHAT] Message:', message.substring(0, 100));

    const payload: any = {
      query: message, // /api/product-info/query expects 'query' field
    };

    if (sessionId) {
      payload.session_id = sessionId;
    }

    // Use the correct endpoint: /api/engenie-chat/query
    const response = await axios.post('/api/engenie-chat/query', payload);
    const result = response.data;

    console.log('[ENGENIE_CHAT] Response received:', {
      success: result.success,
      hasAnswer: !!result.answer,
      source: result.source,
      sourcesUsed: result.sources_used
    });

    if (result.success) {
      return {
        success: true,
        response_text: result.answer || '',
        citations: result.sources_used || [],
        source_type: result.source || 'unknown',
        confidence: result.found_in_database ? 0.9 : 0.7,
        is_validated: result.found_in_database || false,
      };
    } else {
      return {
        success: false,
        response_text: result.answer || result.error || 'Failed to get response',
        citations: [],
        source_type: 'error',
        confidence: 0,
        is_validated: false,
        error: result.error,
      };
    }

  } catch (error: any) {
    console.error("[ENGENIE_CHAT] Error:", error.response?.data || error.message);

    // Return a friendly fallback for greetings
    const lowerMessage = message.toLowerCase().trim();
    const isGreeting = ['hi', 'hello', 'hey', 'good morning', 'good afternoon', 'good evening'].some(
      g => lowerMessage === g || lowerMessage === g + '!'
    );

    if (isGreeting) {
      return {
        success: true,
        response_text: "Hello! I'm Engenie, your industrial procurement assistant. How can I help you find the right instruments for your project today?",
        citations: [],
        source_type: 'fallback',
        confidence: 1.0,
        is_validated: true,
      };
    }

    throw new Error(error.response?.data?.error || "Failed to get chat response");
  }
};

// ====================================================================
// === UNIFIED INTENT ROUTING ===
// ====================================================================

/**
 * Unified routing result that can come from any workflow
 */
export interface WorkflowSuggestion {
  name: string;
  workflow_id: string;
  description: string;
  action: string;
}

export interface UnifiedRoutingResult {
  intent: string;
  responseType: 'solution' | 'requirements' | 'greeting' | 'question' | 'modification' | 'error' | 'workflowSuggestion';
  message?: string;
  instruments: any[];
  accessories: any[];
  projectName?: string;
  awaitingSelection?: boolean;
  items?: any[];
  isSolution?: boolean;
  changesMade?: string[];
  suggestWorkflow?: WorkflowSuggestion;  // For suggesting workflows without auto-routing
  fieldDescriptions?: Record<string, string>;
  field_descriptions?: Record<string, string>;
}

/**
 * Routes user input by intent classification.
 * 
 * This is the PRIMARY entry point for user input on the Project page.
 * It replaces direct calls to identifyInstruments or other workflows.
 * 
 * Flow:
 * 1. Calls the Solution Deep Agent (/api/agentic/solution) FIRST for ALL inputs.
 * 2. If the deep agent determines the input is out-of-context, it internally calls
 *    the router and returns should_redirect=true with a routed_to target.
 * 3. Based on routed_to:
 *    - "out_of_domain"        → Block the query with a rejection message
 *    - "search" / "instrument_identifier" → identifyInstruments (product search)
 *    - "chat" / "engenie_chat" / others   → callEnGenieChat (conversational)
 * 4. If no redirect → returns the solution workflow result directly.
 * 
 * @param userInput User's input text
 * @param currentInstruments Optional - existing instruments for modification detection
 * @param currentAccessories Optional - existing accessories for modification detection
 * @param sessionId Optional - session ID for workflow tracking
 * @returns A promise that resolves with unified routing result
 */
export const routeUserInputByIntent = async (
  userInput: string,
  currentInstruments?: any[],
  currentAccessories?: any[],
  sessionId?: string
): Promise<UnifiedRoutingResult> => {
  try {
    // Step 1: Call the Solution Deep Agent FIRST for ALL inputs.
    // The deep agent handles solution context internally and calls the router
    // only when the input is out-of-context for the ongoing solution workflow.
    console.log('[INTENT_ROUTER] Calling Solution Deep Agent as primary entry point...');
    const solutionResult = await callSolutionWorkflow(userInput, sessionId);

    // Step 2: Check if the deep agent redirected this input to another workflow
    if (solutionResult.shouldRedirect) {
      const routedTo = solutionResult.routedTo || 'chat';
      console.log('[INTENT_ROUTER] Solution Deep Agent redirecting to:', routedTo);

      // ROUTE A: Out-of-domain - block the query immediately
      if (routedTo === 'out_of_domain') {
        console.log('[INTENT_ROUTER] OUT_OF_DOMAIN detected - blocking query');
        return {
          intent: 'out_of_domain',
          responseType: 'error',
          message:
            "I can only help with industrial instrumentation and process control equipment. " +
            "This includes transmitters, sensors, valves, actuators, flow meters, and similar devices.\n\n" +
            "I cannot help with:\n" +
            "- General knowledge questions (weather, sports, entertainment)\n" +
            "- PLC/SCADA programming or control logic\n" +
            "- Pricing, purchasing, or availability\n" +
            "- Installation, troubleshooting, or maintenance\n\n" +
            "Please describe the instrumentation or equipment you need.",
          instruments: currentInstruments || [],
          accessories: currentAccessories || [],
          isSolution: false,
        };
      }

      // ROUTE B: Product search / instrument identification
      if (routedTo === 'search' || routedTo === 'instrument_identifier') {
        console.log('[INTENT_ROUTER] Routing to INSTRUMENT IDENTIFIER workflow');
        const identifyResult = await identifyInstruments(
          userInput,
          currentInstruments,
          currentAccessories,
          sessionId
        );
        return {
          intent: 'productRequirements',
          responseType: identifyResult.responseType || 'requirements',
          message: identifyResult.message,
          instruments: identifyResult.instruments,
          accessories: identifyResult.accessories,
          projectName: identifyResult.projectName,
          awaitingSelection: identifyResult.awaitingSelection,
          items: identifyResult.items,
          isSolution: false,
          changesMade: identifyResult.changesMade,
        };
      }

      // ROUTE C: Chat / conversational (greeting, chitchat, knowledge question)
      console.log('[INTENT_ROUTER] Routing to ENGENIE CHAT for:', routedTo);
      try {
        const chatResult = await callEnGenieChat(userInput, sessionId);
        const intentType = solutionResult.intentType || routedTo;
        const isGreeting = ['greeting', 'chitchat', 'conversational'].includes(intentType);
        return {
          intent: intentType,
          responseType: isGreeting ? 'greeting' : 'question',
          message: chatResult.response_text ||
            "Hello! I'm Engenie, your industrial procurement assistant. How can I help you today?",
          instruments: currentInstruments || [],
          accessories: currentAccessories || [],
          isSolution: false,
        };
      } catch (e) {
        return {
          intent: routedTo,
          responseType: 'greeting',
          message: "Hello! I'm Engenie, your industrial procurement assistant. How can I help you find the right instruments for your project today?",
          instruments: currentInstruments || [],
          accessories: currentAccessories || [],
          isSolution: false,
        };
      }
    }

    // Step 3: No redirect — this was a valid solution workflow response
    console.log('[INTENT_ROUTER] Solution Deep Agent returned solution result');
    return {
      intent: 'solution',
      responseType: 'solution',
      message: solutionResult.message,
      instruments: solutionResult.instruments,
      accessories: solutionResult.accessories,
      projectName: solutionResult.projectName,
      awaitingSelection: solutionResult.awaitingSelection,
      items: solutionResult.items,
      isSolution: true,
      fieldDescriptions: solutionResult.fieldDescriptions,
      field_descriptions: solutionResult.fieldDescriptions,
    };

  } catch (error: any) {
    console.error("[INTENT_ROUTER] Error:", error.message);
    throw new Error(error.message || "Failed to route user input");
  }
};


/**
 * Modification result interface
 */
export interface InstrumentModificationResult {
  responseType: string;
  message: string;
  instruments: any[];
  accessories: any[];
  changesMade?: string[];
  summary?: string;
}

/**
 * Modifies/refines the identified instruments and accessories list based on user's natural language request.
 * Supports adding, removing, and updating items in the list.
 * 
 * @param modificationRequest User's modification request in natural language (e.g., "Add a control valve", "Remove the flow meter")
 * @param currentInstruments Current list of identified instruments
 * @param currentAccessories Current list of identified accessories
 * @param searchSessionId Optional session ID for tracking
 * @returns A promise that resolves with the modified instruments and accessories
 * 
 * @example
 * // Add new items
 * modifyInstruments("Add 2 control valves with 3-inch size", instruments, accessories);
 * 
 * // Remove items
 * modifyInstruments("Remove the temperature transmitter", instruments, accessories);
 * 
 * // Update specifications
 * modifyInstruments("Change the pressure range to 0-200 psi for all transmitters", instruments, accessories);
 */
export const modifyInstruments = async (
  modificationRequest: string,
  currentInstruments: any[],
  currentAccessories: any[],
  searchSessionId?: string
): Promise<InstrumentModificationResult> => {
  try {
    const payload: any = {
      modification_request: modificationRequest,
      current_instruments: currentInstruments,
      current_accessories: currentAccessories,
    };

    if (searchSessionId) {
      payload.search_session_id = searchSessionId;
    }

    const response = await axios.post(`/api/agentic/modify-instruments`, payload);
    return convertKeysToCamelCase(response.data) as InstrumentModificationResult;
  } catch (error: any) {
    console.error("Instrument modification error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Failed to modify instruments");
  }
};

/**
 * Gets images for specific analysis products with vendor logo
 * @param vendor Vendor name (e.g. "Emerson")
 * @param productType Product type (e.g. "Flow Transmitter") 
 * @param productName Product name (e.g. "Rosemount 3051")
 * @param modelFamilies Array of model families (e.g. ["3051C", "3051S"])
 * @returns A promise that resolves with analysis image result containing top image, vendor logo, and all images
 */
export const getAnalysisProductImages = async (
  vendor: string,
  productType: string,
  productName: string,
  modelFamilies: string[]
): Promise<AnalysisImageResult> => {
  try {
    const response = await axios.post(`/api/get_analysis_product_images`, {
      vendor,
      product_type: productType,
      product_name: productName,
      model_families: modelFamilies
    });
    return convertKeysToCamelCase(response.data) as AnalysisImageResult;
  } catch (error: any) {
    console.error("Analysis image fetch error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Failed to fetch analysis images");
  }
};

// searchVendors function removed as it is deprecated.
// Use callAgenticProductSearch instead.

// ==================== STRATEGY DOCUMENT API FUNCTIONS ====================

/**
 * Strategy document interface
 */
export interface StrategyDocument {
  _id?: string;
  userId?: string;
  vendorId: string;
  vendorName: string;
  category: string;
  subcategory: string;
  strategy: string;
  refinery: string;
  additionalComments: string;
  ownerName: string;
  createdAt?: string;
  updatedAt?: string;
  isActive?: boolean;
}

/**
 * Get all strategy documents for the current user
 * @param category Optional category filter
 * @returns A promise that resolves with the list of strategy documents
 */
export const getStrategyDocuments = async (category?: string): Promise<{
  success: boolean;
  documents: StrategyDocument[];
  totalCount: number;
}> => {
  try {
    const params = category ? { category } : {};
    const response = await axios.get(`/api/strategy-documents`, { params });
    return convertKeysToCamelCase(response.data);
  } catch (error: any) {
    console.error("Get strategy documents error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Failed to get strategy documents");
  }
};

/**
 * Upload a single strategy document
 * @param document Strategy document data
 * @returns A promise that resolves with the created document ID
 */
export const uploadStrategyDocument = async (document: Partial<StrategyDocument>): Promise<{
  success: boolean;
  documentId: string;
  message: string;
}> => {
  try {
    const response = await axios.post(`/api/strategy-documents`, {
      vendor_id: document.vendorId,
      vendor_name: document.vendorName,
      category: document.category,
      subcategory: document.subcategory,
      strategy: document.strategy,
      refinery: document.refinery,
      additional_comments: document.additionalComments,
      owner_name: document.ownerName
    });
    return convertKeysToCamelCase(response.data);
  } catch (error: any) {
    console.error("Upload strategy document error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Failed to upload strategy document");
  }
};

/**
 * Bulk upload strategy documents from a CSV file
 * @param file CSV file containing strategy documents
 * @returns A promise that resolves with upload results
 */
export const bulkUploadStrategyDocuments = async (file: File): Promise<{
  success: boolean;
  uploadedCount: number;
  documentIds: string[];
  message: string;
}> => {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await axios.post(`/api/strategy-documents/bulk`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return convertKeysToCamelCase(response.data);
  } catch (error: any) {
    console.error("Bulk upload strategy documents error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Failed to bulk upload strategy documents");
  }
};

/**
 * Delete all strategy documents for the current user
 * @returns A promise that resolves with delete results
 */
export const deleteAllStrategyDocuments = async (): Promise<{
  success: boolean;
  deletedCount: number;
  message: string;
}> => {
  try {
    const response = await axios.delete(`/api/strategy-documents`);
    return convertKeysToCamelCase(response.data);
  } catch (error: any) {
    console.error("Delete strategy documents error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Failed to delete strategy documents");
  }
};

/**
 * Import strategy documents from the default CSV file on the server
 * @returns A promise that resolves with import results
 */
export const importDefaultStrategyDocuments = async (): Promise<{
  success: boolean;
  uploadedCount: number;
  message: string;
}> => {
  try {
    const response = await axios.post(`/api/strategy-documents/import-default`);
    return convertKeysToCamelCase(response.data);
  } catch (error: any) {
    console.error("Import default strategy documents error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Failed to import default strategy documents");
  }
};

/**
 * Strategy file extraction result interface
 */
export interface StrategyFileUploadResult {
  success: boolean;
  message: string;
  documentId?: string;
  filename?: string;
  fileType?: string;
  entryCount?: number;
  extractedData?: Array<{
    vendorName: string;
    category: string;
    subcategory: string;
    stratergy: string;
  }>;
  error?: string;
  extractedTextPreview?: string;
}

/**
 * Upload a strategy file (PDF, DOCX, TXT, Images) and extract structured strategy data using Gemini 2.5 Flash
 * 
 * This function accepts any supported file format and sends it to the backend where:
 * 1. Text is extracted from the file (PDF, DOCX, images via OCR, etc.)
 * 2. Gemini 2.5 Flash LLM analyzes the text to extract structured strategy data
 * 3. The extracted data is stored in MongoDB
 * 
 * @param file The file to upload (PDF, DOCX, TXT, JPG, PNG, etc.)
 * @returns A promise that resolves with the extraction results including extracted strategy data
 */
export const uploadStrategyFile = async (file: File): Promise<StrategyFileUploadResult> => {
  try {
    const formData = new FormData();
    formData.append('file', file);

    const response = await axios.post(`/api/upload-strategy-file`, formData, {
      headers: {
        'Content-Type': 'multipart/form-data'
      }
    });
    return convertKeysToCamelCase(response.data);
  } catch (error: any) {
    console.error("Strategy file upload error:", error.response?.data || error.message);

    // Return a structured error response
    return {
      success: false,
      message: error.response?.data?.error || "Failed to upload and process strategy file",
      error: error.response?.data?.error || error.message,
      extractedTextPreview: error.response?.data?.extracted_text_preview
    };
  }
};

// ==================== AGENTIC PRODUCT SEARCH WORKFLOW ====================

/**
 * Response from agentic product search workflow with checkpoint support
 */
export interface AgenticProductSearchResponse {
  success: boolean;
  awaiting_user_input: boolean;        // Is workflow paused for user input?
  sales_agent_response: string;        // Message to show user
  current_sales_step: string;          // Current checkpoint name (legacy)
  current_phase: string;               // Current workflow phase
  thread_id: string;                   // Thread ID for resuming conversation
  product_type?: string;               // Detected product type
  schema?: any;                        // Schema for left sidebar display
  missing_fields?: string[];           // Missing mandatory fields
  validation_result?: any;             // Full validation result
  available_advanced_params?: any[];   // Discovered advanced parameters
  advanced_parameters_result?: {       // Advanced params discovery result
    discovered_specifications?: any[];
    selected_specifications?: any[];
    total_discovered?: number;
    total_selected?: number;
  };
  ranked_products?: any[];             // Final results (if completed)
  completed?: boolean;                 // Is workflow finished?
  ready_for_vendor_search?: boolean;   // Ready to proceed with vendor search
  final_requirements?: {               // Final collected requirements
    productType?: string;
    mandatoryRequirements?: Record<string, any>;
    optionalRequirements?: Record<string, any>;
    advancedParameters?: any[];
  };
  error?: string;                      // Error message if failed
  provided_requirements?: Record<string, any>; // Requirements already extracted from input
}

/**
 * Calls the agentic product search workflow with checkpoint support
 *
 * This function handles:
 * - Thread-based conversation continuity (thread_id)
 * - Interrupt state detection (awaiting_user_input)
 * - Multi-checkpoint workflow navigation
 * - Knowledge question handling
 * - Error recovery with retry
 *
 * @param message User's message or response to checkpoint
 * @param threadId Optional thread ID for resuming conversation
 * @param searchSessionId Optional session ID for tracking
 * @returns Agentic workflow response with checkpoint state
 *
 * @example
 * // Start new conversation
 * const response = await callAgenticProductSearch("I need a pressure transmitter");
 *
 * // Resume conversation
 * const response = await callAgenticProductSearch("100 PSI range", response.thread_id);
 */
export const callAgenticProductSearch = async (
  message: string,
  threadId?: string,
  searchSessionId?: string,
  productType?: string,              // Product type for schema lookup
  sourceWorkflow?: string,           // Source workflow identifier
  itemThreadId?: string,             // CRITICAL: Item thread ID for resuming branch
  workflowThreadIdOverride?: string, // CRITICAL: Override workflow thread ID
  parentSessionId?: string,          // Solution session that spawned this search
  parentInstanceId?: string          // Solution workflow_thread_id for observability
): Promise<AgenticProductSearchResponse> => {
  try {
    // Get thread context from SessionManager
    const sessionManager = getSessionManager();
    const mainThreadId = sessionManager.getMainThreadId();

    // CRITICAL: Use provided thread IDs for workflow resumption
    // If workflowThreadIdOverride provided, use it (resume existing workflow)
    // Otherwise create new sub-thread
    let workflowThreadId = workflowThreadIdOverride;
    if (!workflowThreadId) {
      const productSearchThread = sessionManager.createSubThread('product_search');
      workflowThreadId = productSearchThread?.subThreadId || threadId;
    }

    const payload: any = {
      message: message,
      // UI-MANAGED THREAD IDs (required by backend)
      main_thread_id: mainThreadId,
      workflow_thread_id: workflowThreadId,
    };

    // CRITICAL: Include item_thread_id if provided (for resuming specific item branch)
    if (itemThreadId) {
      payload.item_thread_id = itemThreadId;
    }

    // Include legacy thread_id for backward compatibility
    if (threadId) {
      payload.thread_id = threadId;
    }

    // Include session ID if provided
    if (searchSessionId) {
      payload.session_id = searchSessionId;
    }

    // Include product_type if provided (critical for schema lookup)
    if (productType) {
      payload.product_type = productType;
    }

    // Include source_workflow to identify caller
    if (sourceWorkflow) {
      payload.source_workflow = sourceWorkflow;
    }

    // Parent linkage — forward when item was spawned by Solution Deep Agent
    if (parentSessionId) {
      payload.parent_session_id = parentSessionId;
    }
    if (parentInstanceId) {
      payload.parent_instance_id = parentInstanceId;
    }

    console.log('[AGENTIC_PS] Calling product search:', {
      messagePreview: message.substring(0, 50),
      mainThreadId,
      workflowThreadId,
      itemThreadId,
      productType,
      sourceWorkflow,
      parentSessionId: parentSessionId?.substring(0, 16),
      parentInstanceId: parentInstanceId?.substring(0, 16),
      isResume: !!threadId,
      isResumeExistingBranch: !!itemThreadId
    });

    const response = await axios.post(`/api/agentic/product-search`, payload);

    if (!response.data.success) {
      throw new Error(response.data.error || "Product search failed");
    }

    // Extract and convert response data
    const responseData = convertKeysToCamelCase(response.data.data) || {};

    // Check if workflow is interrupted (awaiting user input)
    const isInterrupted = responseData.awaitingUserInput || false;

    console.log('[AGENTIC_PS] Response received:', {
      currentPhase: responseData.currentPhase,
      currentStep: responseData.currentSalesStep,
      isInterrupted,
      hasSchema: !!responseData.schema,
      missingFields: responseData.missingFields,
      hasProducts: !!responseData.rankedProducts,
      productCount: responseData.rankedProducts?.length || 0
    });

    return {
      success: true,
      awaiting_user_input: isInterrupted,
      sales_agent_response: responseData.salesAgentResponse || responseData.response || "",
      current_sales_step: responseData.currentSalesStep || responseData.currentPhase || "initialInput",
      current_phase: responseData.currentPhase || "initialValidation",
      thread_id: responseData.threadId || threadId || "",
      product_type: responseData.productType,
      schema: responseData.schema,                              // Schema for left sidebar
      missing_fields: responseData.missingFields || [],        // Missing fields
      validation_result: responseData.validationResult,        // Full validation result
      available_advanced_params: responseData.availableAdvancedParams || [],
      ranked_products: responseData.rankedProducts || [],
      ready_for_vendor_search: responseData.readyForVendorSearch || false,
      completed: responseData.completed || !isInterrupted,
      // CRITICAL: Include final_requirements for triggering analysis after workflow completion
      final_requirements: responseData.finalRequirements,
      provided_requirements: responseData.providedRequirements || {}
    };

  } catch (error: any) {
    console.error("[AGENTIC_PS] Product search error:", error.response?.data || error.message);

    return {
      success: false,
      awaiting_user_input: false,
      sales_agent_response: "",
      current_sales_step: "analysisError",
      current_phase: "error",
      thread_id: threadId || "",
      completed: false,
      error: error.response?.data?.error || error.message || "Product search failed"
    };
  }
};

/**
 * Calls the Product Search Agentic Workflow (New Unified Endpoint)
 * 
 * This integrates the complete product search workflow:
 * - Step 1: Validation (product type detection & schema generation with PPI)
 * - Step 2: Sales Agent (requirements collection)
 * - Ready for Step 3: Vendor Search
 * 
 * @param userInput User's requirements description
 * @param source Source of the request: "direct", "instrument_identifier", or "solution"
 * @param sourceData Optional data from source workflow
 * @param searchSessionId Optional session ID
 * @returns Product search workflow result
 */
export const callProductSearchWorkflow = async (
  userInput: string,
  source: "direct" | "instrument_identifier" | "solution" = "direct",
  sourceData?: any,
  searchSessionId?: string
): Promise<any> => {
  try {
    // Get thread context from SessionManager
    const sessionManager = getSessionManager();
    const mainThreadId = sessionManager.getMainThreadId();

    // Create a new product_search sub-thread
    const productSearchThread = sessionManager.createSubThread('product_search');
    const workflowThreadId = productSearchThread?.subThreadId;

    const payload: any = {
      user_input: userInput,
      source: source,
      // UI-MANAGED THREAD IDs (required by backend)
      main_thread_id: mainThreadId,
      workflow_thread_id: workflowThreadId,
    };

    if (sourceData) {
      payload.source_data = sourceData;
    }

    if (searchSessionId) {
      payload.session_id = searchSessionId;
    }

    console.log(`[PRODUCT_SEARCH] Calling workflow from source: ${source}`);
    console.log(`[PRODUCT_SEARCH] Thread IDs: main=${mainThreadId}, workflow=${workflowThreadId}`);
    console.log(`[PRODUCT_SEARCH] User input: ${userInput.substring(0, 100)}...`);

    const response = await axios.post(`/api/agentic/product-search`, payload);

    if (!response.data.success) {
      throw new Error(response.data.error || "Product search workflow failed");
    }

    console.log(`[PRODUCT_SEARCH] Workflow completed successfully`);
    console.log(`[PRODUCT_SEARCH] Product type: ${response.data.data.product_type}`);

    return convertKeysToCamelCase(response.data.data);
  } catch (error: any) {
    console.error("[PRODUCT_SEARCH] Workflow error:", error.response?.data || error.message);
    throw new Error(error.response?.data?.error || "Product search workflow failed");
  }
};

/**
 * Resume the Product Search Workflow with user decision
 * 
 * Call this when the user responds to an awaiting_user_input checkpoint.
 * 
 * @param threadId Thread ID from previous response (required)
 * @param userDecision User's decision: "add_fields", "continue", "yes", "no"
 * @param userProvidedFields Optional: Field values provided by user for missing fields
 * @param userInput Optional: Free-text input from user
 * @param sessionId Optional: Session tracking ID
 * @returns Updated workflow response
 * 
 * @example
 * // User chooses to add missing fields
 * await resumeProductSearch(threadId, "add_fields");
 * 
 * // User provides the missing field values
 * await resumeProductSearch(threadId, undefined, { measurementRange: "0-100 PSI" });
 * 
 * // User chooses to skip advanced params
 * await resumeProductSearch(threadId, "no");
 */
export const resumeProductSearch = async (
  threadId: string,
  userDecision?: string,
  userProvidedFields?: Record<string, any>,
  userInput?: string,
  sessionId?: string,
  currentPhase?: string
): Promise<AgenticProductSearchResponse> => {
  try {
    if (!threadId) {
      throw new Error("thread_id is required to resume workflow");
    }

    // Get main thread ID from SessionManager
    const sessionManager = getSessionManager();
    const mainThreadId = sessionManager.getMainThreadId();

    const payload: any = {
      thread_id: threadId,
      // UI-MANAGED THREAD IDs (required by backend)
      main_thread_id: mainThreadId,
      workflow_thread_id: threadId,  // Use existing thread ID for resume
      current_phase: currentPhase,
    };

    if (userDecision) {
      payload.user_decision = userDecision;
    }

    if (userProvidedFields && Object.keys(userProvidedFields).length > 0) {
      payload.user_provided_fields = userProvidedFields;
    }

    if (userInput) {
      payload.user_input = userInput;
    }

    if (sessionId) {
      payload.session_id = sessionId;
    }

    console.log('[RESUME_PS] Resuming workflow:', {
      mainThreadId,
      workflowThreadId: threadId,
      userDecision,
      hasProvidedFields: !!userProvidedFields,
      hasUserInput: !!userInput
    });

    const response = await axios.post('/api/agentic/product-search', payload);

    if (!response.data.success) {
      throw new Error(response.data.error || "Failed to resume workflow");
    }

    // Extract and convert response data
    const responseData = convertKeysToCamelCase(response.data.data) || {};
    const isInterrupted = responseData.awaitingUserInput || false;

    console.log('[RESUME_PS] Response received:', {
      currentPhase: responseData.currentPhase,
      isInterrupted,
      hasSchema: !!responseData.schema,
      completed: responseData.completed
    });

    return {
      success: true,
      awaiting_user_input: isInterrupted,
      sales_agent_response: responseData.salesAgentResponse || "",
      current_sales_step: responseData.currentSalesStep || responseData.currentPhase || "initialInput",
      current_phase: responseData.currentPhase || "initialValidation",
      thread_id: responseData.threadId || threadId,
      product_type: responseData.productType,
      schema: responseData.schema,
      missing_fields: responseData.missingFields || [],
      validation_result: responseData.validationResult,
      available_advanced_params: responseData.availableAdvancedParams || [],
      ranked_products: responseData.rankedProducts || [],
      ready_for_vendor_search: responseData.readyForVendorSearch || false,
      completed: responseData.completed || !isInterrupted,
      // CRITICAL: Include final_requirements for triggering analysis after workflow completion
      final_requirements: responseData.finalRequirements
    };

  } catch (error: any) {
    console.error("[RESUME_PS] Error:", error.response?.data || error.message);

    return {
      success: false,
      awaiting_user_input: false,
      sales_agent_response: "",
      current_sales_step: "analysisError",
      current_phase: "error",
      thread_id: threadId,
      completed: false,
      error: error.response?.data?.error || error.message || "Failed to resume workflow"
    };
  }
};

