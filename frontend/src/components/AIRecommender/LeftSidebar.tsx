import React, { useState, useEffect, useRef } from "react";
import { Bot, LogOut, ChevronLeft, ChevronRight, User, Upload } from "lucide-react";
import { RequirementSchema, ValidationResult } from "./types";
// Note: getAllFieldDescriptions is imported dynamically in useEffect for batch fetching
import { Button } from "@/components/ui/button";
import {
  DropdownMenu,
  DropdownMenuTrigger,
  DropdownMenuContent,
  DropdownMenuLabel,
  DropdownMenuItem,
  DropdownMenuSeparator,
} from "@/components/ui/dropdown-menu";
import {
  Tooltip,
  TooltipContent,
  TooltipProvider,
  TooltipTrigger,
} from "@/components/ui/tooltip";
import { useAuth } from "../../contexts/AuthContext";
import { useNavigate } from "react-router-dom";

// Helper functions
function capitalizeFirstLetter(str?: string): string {
  if (!str) return "";
  return str.charAt(0).toUpperCase() + str.slice(1);
}

function prettify(raw: string): string {
  if (!raw) return "";
  return raw
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[-_]/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .split(" ")
    .map((word) => word.charAt(0).toUpperCase() + word.slice(1))
    .join(" ");
}

/**
 * Deduplicate concatenated values like "5 mΩ 50 mΩ 5 mΩ 10 mΩ 5 mΩ"
 * This handles cases where multiple values are joined together with spaces
 */
function deduplicateValue(value: string): string {
  if (!value || typeof value !== 'string') return value;

  // Pattern to match value+unit pairs (e.g., "5 mΩ", "24-16 AWG", "50g")
  const unitPattern = /(\d+[-\d]*\s*(?:mΩ|Ω|AWG|V|VAC|VDC|A|mA|kV|Hz|g|N|mm|°C|%|MΩ-km|MΩ|µm))/gi;
  const matches = value.match(unitPattern);

  if (matches && matches.length > 1) {
    // Deduplicate the matches
    const seen = new Set<string>();
    const unique: string[] = [];
    for (const match of matches) {
      const normalized = match.trim().toLowerCase();
      if (!seen.has(normalized)) {
        seen.add(normalized);
        unique.push(match.trim());
      }
    }
    // Only replace if we reduced duplicates
    if (unique.length < matches.length) {
      return unique.join(', ');
    }
  }

  return value;
}

function getNestedValue(obj: any, path: string): any {
  if (!obj) return undefined;
  return path.split(".").reduce((acc, k) => (acc ? acc[k] : undefined), obj);
}

function getAllLeafKeys(obj: any, parentKey = ""): string[] {
  if (!obj) return [];
  return Object.entries(obj).flatMap(([key, value]) =>
    value && typeof value === "object" && !Array.isArray(value)
      ? getAllLeafKeys(value, parentKey ? `${parentKey}.${key}` : key)
      : [parentKey ? `${parentKey}.${key}` : key]
  );
}

// Helper function to render all fields in a flat structure within one container
function renderFlatFieldsList(
  obj: { [key: string]: any },
  collectedData: { [key: string]: any },
  fieldDescriptions: Record<string, string>,
  fetchOnHover?: (fieldName: string, contextValue?: string) => void,
  parentKey = ""
): JSX.Element[] {
  const fieldsByCategory: { [category: string]: any[] } = {};
  // Track seen field names to prevent duplicates from mandatory + optional
  const seenFieldNames = new Set<string>();

  function traverseAndCollect(currentObj: any, currentParentKey = "", hierarchyPath: string[] = []) {
    Object.entries(currentObj).forEach(([key, value]) => {
      const fullKey = currentParentKey ? `${currentParentKey}.${key}` : key;
      const newHierarchyPath = [...hierarchyPath, prettify(key)];

      if (value !== null && typeof value === "object" && !Array.isArray(value)) {
        // Check if this is a Deep Agent structured object with 'value' property or suggested_values
        // Deep Agent returns: { value: "...", source: "...", confidence: 0.9, standards_referenced: [...] }
        const isMetadataObject = "value" in value || "suggested_values" in value || "source" in value;

        if (isMetadataObject) {
          // This is a structured field metadata object - extract the value and treat as leaf
          const extractedValue = (value as any).value;

          // Check if filled - handle string or non-string values
          const isFilled = extractedValue !== undefined &&
            extractedValue !== null &&
            extractedValue !== "" &&
            (typeof extractedValue !== "string" || extractedValue.toLowerCase() !== "not specified");

          const fieldName = prettify(key);
          const hierarchicalLabel = newHierarchyPath.length > 1
            ? `${newHierarchyPath.slice(0, -1).join(" > ")} > ${fieldName}`
            : fieldName;

          const displayValue = deduplicateValue(String(extractedValue ?? ""));
          const categoryPath = newHierarchyPath.slice(0, -1).join(" ");

          const category = categoryPath || "General";
          if (!fieldsByCategory[category]) {
            fieldsByCategory[category] = [];
          }

          // Skip if field name already seen (prevents duplicates from mandatory + optional)
          if (!seenFieldNames.has(fieldName)) {
            seenFieldNames.add(fieldName);
            fieldsByCategory[category].push({
              fullKey,
              fieldName,
              hierarchicalLabel,
              displayValue,
              isFilled,
              fieldDescriptions
            });
          }
        } else {
          // Recursively traverse nested objects, building the hierarchy path
          traverseAndCollect(value, fullKey, newHierarchyPath);
        }
      } else {
        // This is a leaf field, group it by category
        let valueRaw = getNestedValue(collectedData, fullKey);
        // Handle structured Deep Agent data in collectedData
        if (valueRaw && typeof valueRaw === "object" && !Array.isArray(valueRaw) && "value" in valueRaw) {
          valueRaw = valueRaw.value;
        }
        const isFilled = valueRaw !== undefined && valueRaw !== "" && valueRaw !== null;
        const fieldName = prettify(key);
        const hierarchicalLabel = newHierarchyPath.length > 1
          ? `${newHierarchyPath.slice(0, -1).join(" > ")} > ${fieldName}`
          : fieldName;
        const displayValue = deduplicateValue(Array.isArray(valueRaw) ? valueRaw.join(", ") : String(valueRaw ?? ""));
        const categoryPath = newHierarchyPath.slice(0, -1).join(" ");

        // Group fields by category
        const category = categoryPath || "General";
        if (!fieldsByCategory[category]) {
          fieldsByCategory[category] = [];
        }

        // Skip if field name already seen (prevents duplicates from mandatory + optional)
        if (!seenFieldNames.has(fieldName)) {
          seenFieldNames.add(fieldName);
          fieldsByCategory[category].push({
            fullKey,
            fieldName,
            hierarchicalLabel,
            displayValue,
            isFilled,
            fieldDescriptions
          });
        }
      }
    });
  }

  traverseAndCollect(obj, parentKey);

  // Now render grouped fields
  const fields: JSX.Element[] = [];
  Object.entries(fieldsByCategory).forEach(([category, categoryFields]) => {
    // Add category header
    fields.push(
      <div key={`category-${category}`} className="font-bold text-foreground mb-1 text-xs">
        {category}
      </div>
    );

    // Add all fields for this category
    categoryFields.forEach((field) => {
      fields.push(
        <div key={field.fullKey} className="flex items-start gap-x-2 ml-6 text-xs mb-1">
          <TooltipProvider delayDuration={200}>
            <Tooltip>
              <TooltipTrigger asChild>
                <span
                  className="cursor-pointer font-medium text-muted-foreground hover:underline"
                  onMouseEnter={() => fetchOnHover && fetchOnHover(field.fullKey, field.displayValue)}
                  onTouchStart={() => fetchOnHover && fetchOnHover(field.fullKey, field.displayValue)}
                >
                  {field.fieldName}:
                </span>
              </TooltipTrigger>
              <TooltipContent
                side="top"
                align="start"
                className="w-64 bg-popover p-2 rounded-md shadow-md border"
              >
                <p className="text-sm whitespace-normal mt-1">
                  {(() => {
                    // Try multiple key variations to find description
                    const descriptions = field.fieldDescriptions;
                    const fullKey = field.fullKey;
                    const leafKey = fullKey.split('.').pop() || fullKey;

                    // Convert camelCase to snake_case for matching template keys
                    const snakeCaseKey = leafKey.replace(/([a-z])([A-Z])/g, '$1_$2').toLowerCase();

                    // Try different key formats
                    const desc = descriptions[fullKey]
                      || descriptions[leafKey]
                      || descriptions[snakeCaseKey]
                      || descriptions[leafKey.toLowerCase()]
                      || descriptions[snakeCaseKey.replace(/_/g, '')]
                      // Try finding by partial match
                      || Object.entries(descriptions).find(([k]) =>
                        k.toLowerCase().includes(leafKey.toLowerCase()) ||
                        leafKey.toLowerCase().includes(k.toLowerCase().replace(/_/g, ''))
                      )?.[1]
                      || null;

                    // If still no description, generate one from the field name
                    if (!desc) {
                      const words = leafKey
                        .replace(/([a-z])([A-Z])/g, '$1 $2')
                        .replace(/[_-]/g, ' ')
                        .split(' ')
                        .map(w => w.charAt(0).toUpperCase() + w.slice(1))
                        .join(' ');
                      return `Specification for ${words}`;
                    }
                    return desc;
                  })()}
                </p>
              </TooltipContent>
            </Tooltip>
          </TooltipProvider>

          {(() => {
            // Determine the actual display value
            // When field is filled, show the actual value
            // When field is empty, show "Not specified" (descriptions are in tooltips)
            const displayValue = field.isFilled ? field.displayValue : "Not specified";

            // Check if the value is meaningful (not empty or "Not specified")
            const isValueMeaningful = field.isFilled &&
              displayValue.trim() !== "" &&
              displayValue.toLowerCase() !== "not specified";

            return (
              <span className={`break-words ${isValueMeaningful ? "text-green-700 font-mono" : "text-red-500 font-mono"}`}>
                {displayValue}
              </span>
            );
          })()}
        </div>
      );
    });
  });

  return fields;
}


// Props interface
interface LeftSidebarProps {
  validationResult: ValidationResult | null;
  requirementSchema: RequirementSchema | null;
  currentProductType: string | null;
  collectedData: { [key: string]: any };
  logout: () => void;
  isDocked: boolean;
  setIsDocked: React.Dispatch<React.SetStateAction<boolean>>;
  hideProfile?: boolean;
  fieldDescriptions?: Record<string, string>;
  onFieldDescriptionsChange?: (descriptions: Record<string, string>) => void;
  fetchOnHover?: (fieldName: string, contextValue?: string) => void;
  onScroll?: (scrollTop: number) => void;
  initialScrollTop?: number;
}

// Main rendering function


const LeftSidebar = ({
  requirementSchema,
  currentProductType,
  collectedData = {},
  logout,
  isDocked,
  setIsDocked,
  hideProfile = false,
  fieldDescriptions: savedFieldDescriptions,
  onFieldDescriptionsChange,
  fetchOnHover,
  onScroll,
  initialScrollTop,
}: LeftSidebarProps) => {
  const [fieldDescriptions, setFieldDescriptions] = useState<Record<string, string>>(savedFieldDescriptions || {});
  const navigate = useNavigate();
  const { user } = useAuth();
  const scrollContainerRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    if (initialScrollTop && scrollContainerRef.current) {
      // Use timeout to ensure content is fully rendered before scrolling
      setTimeout(() => {
        if (scrollContainerRef.current) {
          scrollContainerRef.current.scrollTop = initialScrollTop;
        }
      }, 100);
    }
  }, []);

  // Note: Auto-undock is now controlled by parent component (index.tsx)
  // This ensures the sidebar undocks at the same time as the API response message appears

  const profileButtonLabel = capitalizeFirstLetter(user?.name || user?.username || "User");
  const profileFullName = user?.name || `${user?.firstName || ''} ${user?.lastName || ''}`.trim() || user?.username || "User";

  useEffect(() => {
    // Update fieldDescriptions when savedFieldDescriptions prop changes
    if (savedFieldDescriptions && Object.keys(savedFieldDescriptions).length > 0) {
      console.log('[FIELDS] Using saved field descriptions:', Object.keys(savedFieldDescriptions).length, 'fields');
      setFieldDescriptions(savedFieldDescriptions);
      return;
    }

    async function fetchAllDescriptionsBatch() {
      if (!requirementSchema || !currentProductType) return;

      const allKeys = [
        ...getAllLeafKeys(requirementSchema.mandatoryRequirements || {}),
        ...getAllLeafKeys(requirementSchema.optionalRequirements || {}),
      ];
      if (allKeys.length === 0) return;

      console.log('[FIELDS] Fetching all field values in BATCH for', allKeys.length, 'fields');

      try {
        // Import and use the batch API
        const { getAllFieldDescriptions } = await import("./api");

        // Single batch request for ALL fields
        const batchResults = await getAllFieldDescriptions(allKeys, currentProductType);

        // Build descriptions map from batch results
        const newDescriptions: Record<string, string> = {};
        allKeys.forEach((key) => {
          newDescriptions[key] = batchResults[key] || "Not specified";
        });

        setFieldDescriptions(newDescriptions);
        console.log('[FIELDS] ✓ Batch fetch complete:', Object.keys(newDescriptions).length, 'fields');

        // Notify parent component about the new field descriptions
        if (onFieldDescriptionsChange) {
          onFieldDescriptionsChange(newDescriptions);
        }
      } catch (err) {
        console.error("[FIELDS] Batch fetch error:", err);
        // Fallback: set empty descriptions
        const emptyDescriptions: Record<string, string> = {};
        allKeys.forEach((key) => {
          emptyDescriptions[key] = "Not specified";
        });
        setFieldDescriptions(emptyDescriptions);
      }
    }

    fetchAllDescriptionsBatch();
  }, [requirementSchema, currentProductType, savedFieldDescriptions]);


  return (
    <div className="flex flex-col h-full w-full glass-sidebar">
      {/* Header - Only show icon if hideProfile is false */}
      {!hideProfile && (
        <div className="flex items-center justify-between py-4 px-3 flex-shrink-0">
          <div className="w-14 h-14 rounded-full flex items-center justify-center shadow" style={{ background: 'var(--gradient-primary)' }}>
            <Bot className="h-8 w-8 text-white" />
          </div>
          {/* Dock button moved to corner */}
        </div>
      )}

      {/* Minimal header for dashboard when hideProfile is true */}
      {hideProfile && (
        <div className="flex items-center justify-end py-4 px-3 flex-shrink-0">
          {/* Dock button moved to corner, this space can be used for other content */}
        </div>
      )}

      {/* Main content */}
      <div
        ref={scrollContainerRef}
        onScroll={(e) => onScroll?.(e.currentTarget.scrollTop)}
        className="flex-1 overflow-y-auto custom-visible-scrollbar p-4 space-y-4"
      >
        {requirementSchema &&
          (isDocked ? (
            // Docked view
            <div >

            </div>
          ) : (
            // Expanded view
            <div>
              <div className="text-center mb-4">
                {/* <h2 className="text-sm font-semibold text-foreground">
                  looking for
                </h2> */}
                <h2 className="text-xl font-semibold">
                  <span className="text-gradient inline-block">
                    {prettify(currentProductType || "")}
                  </span>
                </h2>
              </div>

              {requirementSchema.mandatoryRequirements && (
                <div className="mb-6">
                  <div className="bg-white/40 backdrop-blur-sm rounded-lg p-4 shadow-sm border border-white/20">
                    <div className="space-y-3">
                      {renderFlatFieldsList(
                        requirementSchema.mandatoryRequirements,
                        collectedData,
                        fieldDescriptions,
                        fetchOnHover
                      )}
                    </div>
                  </div>
                </div>
              )}
              {requirementSchema.optionalRequirements && (
                <div>
                  <div className="bg-white/40 backdrop-blur-sm rounded-lg p-4 shadow-sm border border-white/20">
                    <div className="space-y-3">
                      {renderFlatFieldsList(
                        requirementSchema.optionalRequirements,
                        collectedData,
                        fieldDescriptions,
                        fetchOnHover
                      )}
                    </div>
                  </div>
                </div>
              )}
            </div>
          ))}
      </div>

      {/* Footer - Only show if hideProfile is false */}
      {!hideProfile && (
        <div className="p-3 border-t border-white/20 flex-shrink-0">
          <DropdownMenu>
            <DropdownMenuTrigger asChild>
              <Button
                variant="outline"
                className="w-full text-sm font-semibold text-muted-foreground hover:bg-secondary/50"
              >
                <div className="w-7 h-7 rounded-full bg-ai-primary flex items-center justify-center text-white font-bold">
                  {profileButtonLabel.charAt(0)}
                </div>
                {!isDocked && <span className="ml-2">{profileButtonLabel}</span>}
              </Button>
            </DropdownMenuTrigger>
            <DropdownMenuContent
              className="w-56 glass-card mt-1"
              align="end"
              side="top"
            >
              <DropdownMenuLabel className="flex items-center gap-2">
                <User className="w-4 h-4" />
                {profileFullName}
              </DropdownMenuLabel>
              <DropdownMenuSeparator />

              {user?.role?.toLowerCase() === "admin" && (
                <>
                  <DropdownMenuItem className="flex gap-2" onClick={() => navigate("/admin")}>
                    <Bot className="h-4 w-4" />
                    Approve Sign Ups
                  </DropdownMenuItem>
                  <DropdownMenuItem className="flex gap-2" onClick={() => navigate("/upload")}>
                    <Upload className="h-4 w-4" />
                    Upload
                  </DropdownMenuItem>
                  <DropdownMenuSeparator />
                </>
              )}

              <DropdownMenuItem className="flex gap-2" onClick={logout}>
                <LogOut className="h-4 w-4" />
                Logout
              </DropdownMenuItem>
            </DropdownMenuContent>
          </DropdownMenu>
        </div>
      )}
    </div>
  );
};

export default LeftSidebar;