// Action button interface for routing between screens
export interface ChatActionButton {
  label: string;
  action: 'openNewWindow' | 'navigate' | 'custom';
  url?: string;
  icon?: string;
  contextData?: any;
}

export interface ChatMessage {
  role: any;
  id: string;
  type: "user" | "assistant" | "feedback";
  content: string;
  timestamp: Date;
  actionButtons?: ChatActionButton[];  // NEW: Routing buttons for screen navigation
  metadata?: {
    productType?: string;
    validationResult?: ValidationResult;
    analysisResult?: AnalysisResult;
    examplePrompt?: string;
    vendorAnalysisComplete?: boolean;
    requirementSchema?: RequirementSchema | null;
  };
}

export interface ValidationResult {
  validationAlert: any;
  isComplete: boolean;
  detectedSchema?: Record<string, any>;
  providedRequirements: Record<string, any>;
  productType: string;
  missingFields?: string[];            // Fields that are missing from user input
  originalInput?: string;              // The original user input text
}
export interface StructuredRequirements {
  [requirement: string]: string | number | boolean | null;
}

export interface StandardsCompliance {
  compliant_standards?: string[];
  non_compliant_standards?: string[];
  compliance_notes?: string;
}

export interface MatchedRequirement {
  parameter_name: string;
  user_requirement: string;
  product_specification: string;
  source_reference?: string;
  status: "MATCHED" | "PARTIAL" | "UNMATCHED";
  is_critical?: boolean;
  standards_compliance?: string;
  match_explanation?: string;
}

export interface ProductMatch {
  productName: string;
  vendor: string;
  matchScore: number;
  requirementsMatch: boolean;
  reasoning: string;
  limitations: string;
  imageUrl?: string; // ✅ Product image URL for vendorMatches
  // New fields for enhanced image support
  topImage?: ProductImage;
  vendorLogo?: VendorLogo;
  allImages?: ProductImage[];
  // Additional data fields
  modelFamily?: string;
  recommendation?: string;
  standardsCompliance?: StandardsCompliance;
  standards_compliance?: StandardsCompliance; // snake_case from backend
  matchedRequirements?: Record<string, MatchedRequirement>;
  matched_requirements?: Record<string, MatchedRequirement>; // snake_case from backend
  unmatchedRequirements?: string[];
  unmatched_requirements?: string[]; // snake_case from backend
  keyStrengths?: string[];
  key_strengths?: string[]; // snake_case from backend
}

export interface VendorAnalysis {
  vendorMatches: ProductMatch[];
  vendorsAnalyzed?: number;
  totalMatches?: number;
  vendorRunDetails?: any[];
}

// ADD imageUrl HERE:
export interface RankedProduct {
  modelFamily?: string;
  productType: string;
  rank: number;
  productName: string;
  vendor: string;
  overallScore: number;
  keyStrengths: string | string[];
  concerns: string;
  requirementsMatch: boolean;
  imageUrl?: string; // ✅ Product image URL for rankedProducts
  // New fields for enhanced image support
  topImage?: ProductImage;
  vendorLogo?: VendorLogo;
  allImages?: ProductImage[];
  // Additional data fields
  recommendation?: string;
  standardsCompliance?: StandardsCompliance;
  standards_compliance?: StandardsCompliance; // snake_case from backend
  matchedRequirements?: Record<string, MatchedRequirement>;
  matched_requirements?: Record<string, MatchedRequirement>; // snake_case from backend
  unmatchedRequirements?: string[];
  unmatched_requirements?: string[]; // snake_case from backend
}

export interface AnalysisResult {
  productType: string;
  vendorAnalysis: VendorAnalysis;
  overallRanking: {
    markdownAnalysis: any;
    rankedProducts: RankedProduct[];
  };
  topRecommendation?: RankedProduct;   // Top recommended product
  totalMatches?: number;               // Total number of matching products
  exactMatchCount?: number;            // Count of exact matches
  approximateMatchCount?: number;      // Count of approximate matches
}

export interface RequirementSchema {
  [productType: string]: {
    mandatory?: Record<string, string>;
    optional?: Record<string, string>;
  } | Record<string, string>;
  mandatoryRequirements?: Record<string, any>; // camelCase keys as per backend
  optionalRequirements?: Record<string, any>; // camelCase keys as per backend
}


export interface IdentifiedItem {
  number: number;
  type: "instrument" | "accessory";
  name: string;
  category: string;
  quantity?: number;
  keySpecs?: string;
  imageUrl?: string;
  sampleInput?: string;
}

export interface AppState {
  messages: ChatMessage[];
  currentProductType: string | null;
  validationResult: ValidationResult | null;
  analysisResult: AnalysisResult | null;
  identifiedItems: IdentifiedItem[] | null; // Added for Instrument Identifier results
  requirementSchema: RequirementSchema | null;
  isLoading: boolean;
  inputValue: string;
  productType?: string;
}

export interface UserCredentials {
  username: string;
  email: string;
  password: string;
  first_name?: string;
  last_name?: string;
}

// New types for step-based workflow
export interface WorkflowSuggestion {
  name: string;
  workflow_id: string;
  description: string;
  action: string;
}

export interface IntentClassificationResult {
  intent: "greeting" | "knowledgeQuestion" | "productRequirements" | "solution" | "workflow" | "chitchat" | "chat" | "other";
  nextStep: "greeting" | "initialInput" | "solutionWorkflow" | "awaitAdditionalAndLatestSpecs" | "awaitAdvancedSpecs" | "showSummary" | "finalAnalysis" | null;
  resumeWorkflow?: boolean;
  isSolution?: boolean;  // True if the input is a complex engineering challenge
  suggestWorkflow?: WorkflowSuggestion;  // Suggestion for UI to display as clickable option
  workflowLocked?: boolean;  // True if user is locked in a workflow
  currentWorkflow?: string;  // Current workflow the user is in
  retryable?: boolean;  // True if error is retryable (network/server error)
}

/**
 * Workflow Routing Result from IntentClassificationRoutingAgent
 * Used to determine which workflow to route user input to
 */
export interface WorkflowRoutingResult {
  query: string;
  target_workflow: "solution" | "instrument_identifier" | "engenie_chat" | "out_of_domain" | "greeting" | "conversational";
  intent: string;
  confidence: number;
  reasoning: string;
  is_solution: boolean;
  solution_indicators: string[];
  extracted_info: Record<string, any>;
  classification_time_ms: number;
  timestamp: string;
  reject_message: string | null;
  direct_response?: string | null;  // Direct response from classifier (e.g. for greetings)
}


export interface AgentResponse {
  content: string;
  nextStep?: string | null;
  maintainWorkflow?: boolean;
}

export type WorkflowStep =
  | "greeting"
  | "initialInput"
  | "awaitMissingInfo"
  | "awaitAdditionalAndLatestSpecs"
  | "awaitAdvancedSpecs"
  | "confirmAfterMissingInfo"
  | "showSummary"
  | "finalAnalysis"
  | "analysisError"
  | "default";

// Advanced Parameters types
export interface AdvancedParameter {
  name: string;
  value?: string;
  selected?: boolean;
}

export interface VendorParametersResult {
  vendor: string;
  parameters: string[];
  sourceUrl: string;
}

export interface AdvancedParametersResult {
  productType: string;
  vendorParameters: VendorParametersResult[];
  uniqueParameters: string[];
  uniqueSpecifications?: Array<{ key: string; name: string; }>;  // Newer format
  totalVendorsSearched: number;
  totalUniqueParameters: number;
  totalUniqueSpecifications?: number;
  fallback?: boolean;
}

export interface AdvancedParametersSelection {
  selectedParameters: Record<string, string>;
  explanation: string;
  friendlyResponse: string;
  totalSelected: number;
}

// Instrument Identification types
export interface IdentifiedInstrument {
  category: string;
  quantity?: number;
  productName: string;
  specifications: Record<string, string>;
  sampleInput: string;
  // Thread context for resuming workflow (backend-provided)
  item_thread_id: string;
  workflow_thread_id: string;
  main_thread_id: string;
}

export interface IdentifiedAccessory {
  category: string;
  quantity?: number;
  accessoryName: string;
  specifications: Record<string, string>;
  sampleInput: string;
  // Thread context for resuming workflow (backend-provided)
  item_thread_id: string;
  workflow_thread_id: string;
  main_thread_id: string;
}

export interface InstrumentIdentificationResult {
  // Response type indicator
  response_type?: 'greeting' | 'question' | 'requirements' | 'modification' | 'error';
  responseType?: 'greeting' | 'question' | 'requirements' | 'modification' | 'error';

  // For greeting and question responses
  message?: string;
  isIndustrial?: boolean;

  // For requirements response
  projectName?: string;
  instruments: IdentifiedInstrument[];
  accessories?: IdentifiedAccessory[];
  summary?: string;  // Made optional for agentic workflow

  // For error responses
  error?: string;

  // Agentic workflow specific fields
  awaitingSelection?: boolean;  // True when waiting for user to select items
  items?: any[];  // Raw items array from agentic workflow
  threadId?: string;  // Workflow thread ID for tracking
  changesMade?: string[];  // List of changes made during modification
}


// New interfaces for image API integration
export interface ProductImage {
  url: string;
  title: string;
  source: "google_cse" | "serpapi" | "serper";
  thumbnail: string;
  domain: string;
  searchType?: string;
  searchPriority?: number;
  relevanceScore?: number;
}

export interface VendorLogo {
  url: string;
  thumbnail: string;
  source: string;
  title?: string;
  domain?: string;
}

export interface AnalysisImageResult {
  vendor: string;
  productType: string;
  productName: string;
  modelFamilies: string[];
  topImage: ProductImage | null;
  vendorLogo: VendorLogo | null;
  allImages: ProductImage[];
  totalFound: number;
  uniqueCount: number;
  bestCount: number;
  searchSummary: {
    searchesPerformed: number;
    searchTypes: string[];
    sourcesUsed: string[];
  };
}

// ==================== AGENTIC WORKFLOW TYPES ====================

/**
 * Agentic checkpoint state
 * Tracks the current state of the agentic product search workflow
 */
export interface AgenticCheckpointState {
  currentStep: string;                  // Current sales-agent checkpoint
  awaitingUserInput: boolean;           // Is workflow paused for user input?
  threadId: string | null;              // Thread ID for resuming conversation
  productType: string | null;           // Detected product type
  availableAdvancedParams: any[];       // Discovered advanced parameters
  isKnowledgeQuestion: boolean;         // Is user asking a question?
  hasError: boolean;                    // Is there an error?
  errorMessage: string | null;          // Error message if any
  retryCount: number;                   // Number of retry attempts
}

/**
 * Agentic conversation message
 * Extends ChatMessage with agentic-specific metadata
 */
export interface AgenticChatMessage extends ChatMessage {
  checkpoint?: string;                  // Which checkpoint generated this message
  isInterrupt?: boolean;                // Is this an interrupt response?
  requiresAction?: boolean;             // Does this message need user action?
  agenticMetadata?: {
    threadId?: string;
    productType?: string;
    advancedParams?: any[];
    currentStep?: string;
  };
}

/**
 * Workflow type discriminator
 */
export type WorkflowType = "flask" | "agentic";

/**
 * Agentic checkpoint names
 */
export type AgenticCheckpoint =
  | "greeting"
  | "initialInput"
  | "awaitMissingInfo"
  | "awaitAdditionalAndLatestSpecs"
  | "awaitAdvancedSpecs"
  | "showSummary"
  | "finalAnalysis"
  | "analysisError"
  | "knowledgeQuestion"
  | "default";

export interface ModifyInstrumentsRequest {
  modification_request: string;
  current_instruments: any[];
  current_accessories: any[];
  session_id?: string;
}
