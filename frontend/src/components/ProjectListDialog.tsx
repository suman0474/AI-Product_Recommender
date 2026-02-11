import React, { useState, useEffect } from 'react';
import { Dialog, DialogContent, DialogHeader, DialogTitle, DialogDescription, DialogTrigger } from '@/components/ui/dialog';
import { Button } from '@/components/ui/button';
import { Card, CardContent, CardTitle } from '@/components/ui/card';
import { Trash2, FileText, Loader2, Download } from 'lucide-react';
import { BASE_URL } from './AIRecommender/api';
import { useToast } from '@/hooks/use-toast';
import {
  AlertDialog,
  AlertDialogAction,
  AlertDialogCancel,
  AlertDialogContent,
  AlertDialogDescription,
  AlertDialogFooter,
  AlertDialogHeader,
  AlertDialogTitle,
  AlertDialogTrigger
} from '@/components/ui/alert-dialog';

interface Project {
  id: string;
  projectName?: string;
  project_name?: string;
  projectDescription?: string;
  project_description?: string;
  productType?: string;
  product_type?: string;
  instrumentsCount?: number;
  instruments_count?: number;
  accessoriesCount?: number;
  accessories_count?: number;
  searchTabsCount?: number;
  search_tabs_count?: number;
  currentStep?: string;
  current_step?: string;
  activeTab?: string;
  active_tab?: string;
  projectPhase?: string;
  project_phase?: string;
  conversationsCount?: number;
  conversations_count?: number;
  hasAnalysis?: boolean;
  has_analysis?: boolean;
  schemaVersion?: string;
  schema_version?: string;
  fieldDescriptionsAvailable?: boolean;
  field_descriptions_available?: boolean;
  projectStatus?: string;
  project_status?: string;
  createdAt?: string;
  created_at?: string;
  updatedAt?: string;
  updated_at?: string;
  requirementsPreview?: string;
  requirements_preview?: string;
  initial_requirements?: string;
  initialRequirements?: string;
  conversation_history?: any;
  conversation_histories?: any;
  conversationHistory?: any;
  conversationHistories?: any;
  analysis_results?: any;
  analysisResults?: any;
  identified_instruments?: any[];
  identifiedInstruments?: any[];
  identified_accessories?: any[];
  identifiedAccessories?: any[];
  collected_data?: any;
  collectedData?: any;
  search_tabs?: any[];
  searchTabs?: any[];
  feedback_entries?: any[];
  feedbackEntries?: any[];
}

interface ProjectListDialogProps {
  open: boolean;
  onOpenChange: (open: boolean) => void;
  children: React.ReactNode;
  onProjectSelect: (projectId: string) => void;
  onProjectDelete?: (deletedProjectId: string) => void;
}

const ProjectListDialog: React.FC<ProjectListDialogProps> = ({
  open,
  onOpenChange,
  children,
  onProjectSelect,
  onProjectDelete
}) => {
  const [projects, setProjects] = useState<Project[]>([]);
  const [loading, setLoading] = useState(false);
  const [exportingId, setExportingId] = useState<string | null>(null);
  const { toast } = useToast();

  const fetchProjects = async () => {
    setLoading(true);
    try {
      const response = await fetch(`${BASE_URL}/api/projects`, {
        credentials: 'include'
      });

      if (!response.ok) {
        const errorData = await response.json();
        throw new Error(errorData.error || 'Failed to fetch projects');
      }

      const data = await response.json();
      console.log('Received projects data:', data);
      setProjects(data.projects || []);
    } catch (error: any) {
      toast({
        title: "Failed to load projects",
        description: error.message || "Could not retrieve your projects",
        variant: "destructive",
      });
    } finally {
      setLoading(false);
    }
  };

  const deleteProject = async (projectId: string, projectName: string) => {
    try {
      console.log(`Deleting project ${projectId} (${projectName}) from MongoDB...`);

      const response = await fetch(`${BASE_URL}/api/projects/${projectId}`, {
        method: 'DELETE',
        credentials: 'include'
      });

      console.log(`Delete response status: ${response.status}`);

      if (!response.ok) {
        const errorData = await response.json();
        console.error('Delete failed:', errorData);
        throw new Error(errorData.error || 'Failed to delete project');
      }

      const result = await response.json();
      console.log('Delete successful:', result);

      // Remove from local state
      setProjects(prevProjects =>
        prevProjects.filter(project => project.id !== projectId)
      );

      // Notify parent component about the deletion
      if (onProjectDelete) {
        onProjectDelete(projectId);
      }

      toast({
        title: "Project Deleted",
        description: `"${projectName}" has been permanently deleted from MongoDB`,
      });

    } catch (error: any) {
      console.error('Delete error:', error);
      toast({
        title: "Delete Failed",
        description: error.message || "Failed to delete project",
        variant: "destructive",
      });
    }
  };

  const handleExportProject = async (projectId: string, projectName: string) => {
    setExportingId(projectId);
    try {
      // 1. Fetch full project details
      const response = await fetch(`${BASE_URL}/api/projects/${projectId}`, {
        credentials: 'include'
      });

      if (!response.ok) {
        throw new Error('Failed to fetch project details');
      }

      const data = await response.json();
      const project = data.project;

      if (!project) {
        throw new Error('Project data is empty');
      }
      const { default: jsPDF } = await import("jspdf");

      // 2. Generate PDF
      const doc = new jsPDF();
      const pageWidth = doc.internal.pageSize.getWidth();
      const pageHeight = doc.internal.pageSize.getHeight();
      const margin = 20;
      let yPos = 20;

      // Helper to add text and advance yPos
      const addText = (text: string, fontSize: number = 12, isBold: boolean = false, indent: number = 0, color: string = '#000000') => {
        // sanitize basic markdown and HTML tokens so clear text doesn't include markers like **
        let s = (text === null || text === undefined) ? '' : String(text);
        s = s.replace(/<[^>]*>/g, '');
        s = s.replace(/\*\*/g, '');
        s = s.replace(/\*/g, '');
        s = s.replace(/\*/g, '');
        s = s.replace(/_/g, '');
        s = s.replace(/`/g, '');
        s = s.replace(/#/g, '');
        s = s.replace(/~~/g, '');
        s = s.replace(/\r/g, '');

        doc.setFontSize(fontSize);
        doc.setFont("helvetica", isBold ? "bold" : "normal");
        doc.setTextColor(color);

        const maxWidth = pageWidth - (margin * 2) - indent;
        const splitText = doc.splitTextToSize(s, maxWidth);

        // Check for page break
        if (yPos + (splitText.length * fontSize * 0.5) > pageHeight - margin) {
          doc.addPage();
          yPos = margin;
        }

        doc.text(splitText, margin + indent, yPos);
        yPos += (splitText.length * fontSize * 0.5) + 2;

        // Reset color to black
        doc.setTextColor('#000000');
      };

      // Helper to print a field label and its value(s) handling nested objects/arrays
      const printFieldValue = (label: string, value: any, indent: number = 10) => {
        // Base font sizes
        const keyFontSize = 11; // match top-level key size
        const valFontSize = 11; // same size for nested values as requested

        // If value is missing, show explicit 'Not specified'
        if (value === undefined || value === null || value === "") {
          addText(`  • ${label}: Not specified`, valFontSize, false, indent);
          return;
        }

        // Arrays: join on comma
        if (Array.isArray(value)) {
          const valStr = value.length > 0 ? value.join(', ') : 'Not specified';
          addText(`  • ${label}: ${valStr}`, valFontSize, false, indent);
          return;
        }

        // Objects: print nested keys on their own indented lines with same key size
        if (typeof value === 'object') {
          addText(label, keyFontSize, true, indent - 2);
          Object.entries(value).forEach(([k, v]) => {
            const display = (v === undefined || v === null || v === "") ? 'Not specified' : (typeof v === 'object' ? JSON.stringify(v) : String(v));
            addText(`• ${k}: ${display}`, valFontSize, false, indent + 8);
          });
          return;
        }

        // Primitives
        addText(`  • ${label}: ${String(value)}`, valFontSize, false, indent);
      };

      // Clean text helper: remove markdown formatting characters and clean up text
      const cleanText = (input: string | null | undefined) => {
        if (input === null || input === undefined) return '';
        let s = String(input);

        // Remove HTML tags
        s = s.replace(/<[^>]*>/g, '');

        // Remove markdown formatting characters
        s = s.replace(/\*\*/g, '');  // Remove **
        s = s.replace(/_/g, '');     // Remove _
        s = s.replace(/`/g, '');     // Remove `
        s = s.replace(/#/g, '');     // Remove #
        s = s.replace(/~~/g, '');
        s = s.replace(/\/\*\*/g, '');    // Remove ~~

        // Clean up whitespace and newlines
        s = s.replace(/[\t\r]+/g, ' ')  // Tabs and carriage returns to space
          .replace(/\n{3,}/g, '\n\n') // Max 2 consecutive newlines
          .replace(/ {2,}/g, ' ')      // Multiple spaces to single
          .trim();

        return s;
      };

      // Strip leading bullet/numbering characters from a line (to avoid double-bullets)
      const stripLeadingBullets = (input: string | null | undefined) => {
        if (input === null || input === undefined) return '';
        let s = String(input).trim();
        // Remove common bullet characters, asterisk, hyphen, middle dot, numbering like '1.' or '1)'
        s = s.replace(/^[\s]*(?:[\u2022\u00B7\u25E6\-*•·◦oO]+|\d+[\.)]|\(|\)|>)+[\s]*/g, '');
        // Also remove any leading dash or asterisk sequences
        s = s.replace(/^[\s]*[-*]+[\s]*/g, '');
        return s.trim();
      };

      // Normalize keys for fuzzy matching pricing keys
      const normalizeKey = (k: string) => {
        if (!k) return '';
        return String(k).toLowerCase().replace(/[^a-z0-9]/g, '');
      };

      const addSectionHeader = (title: string) => {
        yPos += 5;
        if (yPos > pageHeight - margin) {
          doc.addPage();
          yPos = margin;
        }
        addText(title, 14, true);
        yPos += 2;
      };

      // Helper to add a bold label and an inline clickable link on the same line
      const addInlineLink = (label: string, link: string, fontSize: number = 10, indent: number = 20) => {
        const cleanUrl = cleanText(link);

        // Ensure there's room on the page
        if (yPos > pageHeight - margin) {
          doc.addPage();
          yPos = margin;
        }

        // Draw label on its own line in bold (matching other section headings)
        addText(label, fontSize, true, indent - 2);

        // Draw the clickable url on next line, indented further
        doc.setFontSize(fontSize);
        doc.setFont('helvetica', 'normal');
        doc.setTextColor(0, 0, 255);

        const startX = margin + indent + 4;
        // Use textWithLink to embed link
        doc.textWithLink(cleanUrl, startX, yPos, { url: cleanUrl });

        // Underline link
        const textWidth = doc.getTextWidth(cleanUrl);
        doc.setDrawColor(0, 0, 255);
        doc.setLineWidth(0.2);
        doc.line(startX, yPos + 0.5, startX + textWidth, yPos + 0.5);

        // Reset colors
        doc.setDrawColor(0, 0, 0);
        doc.setTextColor('#000000');

        // Advance yPos
        yPos += (fontSize * 0.5) + 4;
      };

      // Helper to draw inline bold label followed by normal text on same line
      const drawInlineBoldThenText = (boldText: string, normalText: string, indent: number = 25, fontSize: number = 10) => {
        if (yPos > pageHeight - margin) {
          doc.addPage();
          yPos = margin;
        }

        doc.setFontSize(fontSize);

        // sanitize inputs (remove markdown tokens including single asterisks)
        const b = (boldText === null || boldText === undefined) ? '' : String(boldText).replace(/<[^>]*>/g, '').replace(/\*\*/g, '').replace(/\*/g, '').replace(/_/g, '').replace(/`/g, '').replace(/#/g, '').replace(/~~/g, '');
        const n = (normalText === null || normalText === undefined) ? '' : String(normalText).replace(/<[^>]*>/g, '').replace(/\*\*/g, '').replace(/\*/g, '').replace(/_/g, '').replace(/`/g, '').replace(/#/g, '').replace(/~~/g, '');

        // Bold part
        doc.setFont('helvetica', 'bold');
        const startX = margin + indent;
        doc.text(b, startX, yPos);
        const labelWidth = doc.getTextWidth(b + ' ');

        // Normal part
        doc.setFont('helvetica', 'normal');
        doc.text(n, startX + labelWidth, yPos);

        yPos += (fontSize * 0.5) + 2;
      };

      // Simple markdown-like renderer for AI messages (headings, bold labels, lists)
      const renderRichText = (input: string | null | undefined, indent: number = 15, fontSize: number = 10) => {
        if (!input) return;
        const lines = String(input).split('\n');
        lines.forEach((raw) => {
          const line = raw.replace(/\r/g, '').trim();
          if (!line) {
            yPos += (fontSize * 0.5) + 2;
            return;
          }

          // ATX-style heading (#) -> bold header
          if (/^#{1,6}\s+/.test(line)) {
            addText(line.replace(/^#{1,6}\s+/, ''), fontSize + 1, true, indent - 2);
            return;
          }

          // Bold-wrapped heading or section like **Title:**
          const boldHeadingMatch = line.match(/^\*\*(.+?)\*\*:?\s*(.*)$/);
          if (boldHeadingMatch && boldHeadingMatch[2] === '') {
            addText(boldHeadingMatch[1], fontSize + 0, true, indent - 2);
            return;
          }

          // List item (numbered or star) e.g. '* **Pressure Range:** 0-75 psi' or '1. Text'
          const listMatch = line.match(/^([*\-]|\d+\.)\s+(.*)$/);
          if (listMatch) {
            const content = listMatch[2];

            // If content contains bold label like **Label:** rest
            const labelMatch = content.match(/^\*\*(.+?)\*\*[:\-]?\s*(.*)$/);
            if (labelMatch) {
              const lbl = labelMatch[1];
              const rest = labelMatch[2] || '';
              drawInlineBoldThenText(`• ${lbl}:`, rest, indent, fontSize);
            } else {
              addText(`• ${content}`, fontSize, false, indent);
            }
            return;
          }

          // Fallback: treat as normal paragraph, but if it starts with bold label, split
          const inlineLabel = line.match(/^\*\*(.+?)\*\*[:\-]?\s*(.*)$/);
          if (inlineLabel) {
            const lbl = inlineLabel[1];
            const rest = inlineLabel[2] || '';
            drawInlineBoldThenText(`${lbl}:`, rest, indent, fontSize);
            return;
          }

          // Default plain text
          addText(line, fontSize, false, indent);
        });
      };

      // --- Title ---
      addText(`Project : ${projectName}`, 20, true);
      yPos += 10;

      // --- Description ---
      const description = project.projectDescription || project.project_description;
      if (description) {
        addSectionHeader("Description");
        addText(description, 11);
      }

      // --- Initial Requirements ---
      const initialRequirements = project.initialRequirements || project.initial_requirements;
      if (initialRequirements) {
        addSectionHeader("Initial Requirements");
        addText(initialRequirements, 11);
      }

      // --- Identified Instruments ---
      const identifiedInstruments = project.identifiedInstruments || project.identified_instruments;
      if (identifiedInstruments && identifiedInstruments.length > 0) {
        addSectionHeader("Identified Instruments");
        identifiedInstruments.forEach((inst: any, index: number) => {
          const category = inst.category || 'Unknown Category';
          const name = inst.productName || inst.product_name || 'Unknown Name';
          const quantity = inst.quantity ? ` (Qty: ${inst.quantity})` : '';

          addText(`${index + 1}. ${category}${quantity} - ${name}`, 11, true, 5);

          if (inst.specifications && Object.keys(inst.specifications).length > 0) {
            addText('Specifications:', 10, true, 10);
            Object.entries(inst.specifications).forEach(([key, val]) => {
              addText(`• ${key}: ${val}`, 10, false, 15);
            });
          }
          yPos += 2;
        });
      }

      // --- Identified Accessories ---
      const identifiedAccessories = project.identifiedAccessories || project.identified_accessories;
      if (identifiedAccessories && identifiedAccessories.length > 0) {
        addSectionHeader("Identified Accessories");
        identifiedAccessories.forEach((acc: any, index: number) => {
          const category = acc.category || 'Unknown Category';
          const name = acc.accessoryName || acc.accessory_name || 'Unknown Name';
          const quantity = acc.quantity ? ` (Qty: ${acc.quantity})` : '';

          addText(`${index + 1}. ${category}${quantity} - ${name}`, 11, true, 5);

          if (acc.specifications && Object.keys(acc.specifications).length > 0) {
            addText('Specifications:', 10, true, 10);
            Object.entries(acc.specifications).forEach(([key, val]) => {
              addText(`• ${key}: ${val}`, 10, false, 15);
            });
          }
          yPos += 2;
        });
      }

      // --- Tabs Data (Collected Data, History, Analysis) ---
      const searchTabs = project.searchTabs || project.search_tabs || [];
      const conversationHistories = project.conversationHistories || project.conversation_histories || project.conversationHistory || project.conversation_history || {};
      const collectedDataMap = project.collectedData || project.collected_data || {};
      const analysisResultsMap = project.analysisResults || project.analysis_results || {};

      // Normalize history to map
      let historyMap: { [key: string]: any } = {};
      if (conversationHistories.messages && Array.isArray(conversationHistories.messages)) {
        historyMap['default'] = conversationHistories;
      } else {
        historyMap = conversationHistories;
      }

      const tabsToExport = [...searchTabs];
      // If we have a 'default' history but it's not in searchTabs, add it
      if (historyMap['default'] && !tabsToExport.find(t => t.id === 'default')) {
        tabsToExport.push({ id: 'default', title: 'Main Chat' });
      }

      if (tabsToExport.length > 0) {
        tabsToExport.forEach((tab: any) => {
          const tabId = tab.id;
          const tabTitle = tab.title || tabId;

          // Check if this tab has ANY data to show
          const hasCollected = collectedDataMap[tabId] && Object.keys(collectedDataMap[tabId]).length > 0;
          const hasHistory = historyMap[tabId] && historyMap[tabId].messages && historyMap[tabId].messages.length > 0;
          const hasAnalysis = analysisResultsMap[tabId];

          if (!hasCollected && !hasHistory && !hasAnalysis) return;

          addSectionHeader(`Tab: ${tabTitle}`);

          // 1. Collected Data (Categorized)
          if (hasCollected) {
            addText("Collected Data:", 12, true, 5);
            const data = collectedDataMap[tabId];

            // Try to get schema from history
            const schema = historyMap[tabId]?.requirementSchema;

            if (schema && (schema.mandatory_requirements || schema.optional_requirements)) {
              // Helper to process a requirement group
              const processGroup = (groupName: string, requirements: any) => {
                if (!requirements) return;

                Object.entries(requirements).forEach(([category, fields]: [string, any]) => {
                  // Check if any field in this category has data
                  const fieldKeys = Object.keys(fields);
                  const fieldsWithData = fieldKeys.filter(key => {
                    const value = data[key];
                    return value !== undefined && value !== null && value !== "";
                  });

                  // For mandatory, also check if we need to show missing fields
                  const hasMissingMandatory = groupName === 'Mandatory' && fieldKeys.some(key => {
                    const value = data[key];
                    return value === undefined || value === null || value === "";
                  });

                  // Only show category if it has data OR has missing mandatory fields
                  if (fieldsWithData.length > 0 || hasMissingMandatory) {
                    addText(category, 11, true, 8);

                    fieldKeys.forEach(key => {
                      const value = data[key];
                      // Skip empty optional fields entirely
                      if (groupName !== 'Mandatory' && (value === undefined || value === null || value === "")) {
                        return;
                      }

                      // Format label: convert camelCase to Title Case with proper spacing
                      const label = key
                        .replace(/([A-Z])/g, ' $1')
                        .replace(/^./, str => str.toUpperCase())
                        .trim();

                      // Use helper to print value (handles nested objects/arrays and missing values)
                      printFieldValue(label, value, 10);
                    });

                    yPos += 2;
                  }
                });
              };

              if (schema.mandatory_requirements) {
                processGroup('Mandatory', schema.mandatory_requirements);
              }

              if (schema.optional_requirements) {
                processGroup('Optional', schema.optional_requirements);
              }

              // Handle "Other" data not in schema
              const schemaFields = new Set<string>();
              const collectFields = (reqs: any) => {
                if (!reqs) return;
                Object.values(reqs).forEach((cat: any) => {
                  Object.keys(cat).forEach(k => schemaFields.add(k));
                });
              };
              collectFields(schema.mandatory_requirements);
              collectFields(schema.optional_requirements);

              const otherKeys = Object.keys(data).filter(k => !schemaFields.has(k));
              if (otherKeys.length > 0) {
                addText("Other Data", 11, true, 8);
                otherKeys.forEach(key => {
                  const label = key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase());
                  const value = data[key];
                  printFieldValue(label, value, 10);
                });
              }

            } else {
              // Fallback for no schema: simple list
              Object.entries(data).forEach(([key, value]) => {
                // Skip empty values
                if (value === undefined || value === null || value === "") return;

                const label = key.replace(/([A-Z])/g, ' $1').replace(/^./, str => str.toUpperCase()).trim();
                printFieldValue(label, value, 10);
              });
            }
            yPos += 3;
          }

          // 2. Conversation History
          if (hasHistory) {
            addText("Conversation History:", 12, true, 5);
            const messages = historyMap[tabId].messages;
            messages.forEach((msg: any) => {
              const role = (msg.type === 'human' || msg.type === 'user') ? 'User' : 'AI';
              const content = msg.content || '';
              addText(`${role}:`, 11, true, 10);
              addText(content, 10, false, 15);
              yPos += 2;
            });
            yPos += 3;
          }

          // 3. Analysis Results
          if (hasAnalysis) {
            addText("Analysis Results:", 12, true, 5);
            const result = analysisResultsMap[tabId];

            if (result.summary) {
              addText("Summary:", 11, true, 10);
              addText(result.summary, 10, false, 15);
              yPos += 2;
            }

            if (result.overallRanking?.rankedProducts?.length > 0) {
              addText("Ranked Products:", 11, true, 10);
              result.overallRanking.rankedProducts.forEach((product: any, idx: number) => {
                const pName = product.productName || product.product_name || product.name || 'Unknown Product';
                const vendor = product.vendor || product.vendorName || product.vendor_name || 'Unknown Vendor';

                yPos += 2;
                addText(`${idx + 1}. ${pName} (${vendor})`, 11, true, 15);

                // Description
                if (product.description) {
                  addText(`Description: ${product.description}`, 10, false, 20);
                }

                // Key Strengths / Pros (check many possible keys)
                const strengthKeys = ['keyStrengths', 'key_strengths', 'strengths', 'pros', 'pros_list', 'prosList', 'advantages', 'benefits', 'benefits_list', 'key_benefits', 'keyBenefits', 'positives', 'key_points', 'keyPoints', 'highlights'];
                let keyStrengths = null;
                for (const k of strengthKeys) {
                  if (product[k]) { keyStrengths = product[k]; break; }
                }
                if (keyStrengths) {
                  addText('Key Strengths:', 10, true, 20);
                  yPos += 2; // Add some spacing

                  const addBulletPoints = (content: any, indent: number = 25) => {
                    if (Array.isArray(content)) {
                      content.forEach((item: any) => {
                        if (item && typeof item === 'object') {
                          Object.entries(item).forEach(([k, v]) => {
                            const key = stripLeadingBullets(cleanText(k));
                            const val = stripLeadingBullets(cleanText(String(v)));
                            addText(`• ${key}: ${val}`, 10, false, indent);
                          });
                        } else if (item) {
                          const line = stripLeadingBullets(cleanText(String(item)));
                          addText(`• ${line}`, 10, false, indent);
                        }
                      });
                    } else if (content && typeof content === 'object') {
                      Object.entries(content).forEach(([k, v]) => {
                        const key = stripLeadingBullets(cleanText(k));
                        const val = stripLeadingBullets(cleanText(String(v)));
                        addText(`• ${key}: ${val}`, 10, false, indent);
                      });
                    } else if (content) {
                      const text = String(content);
                      // Split by newlines and add as separate bullet points
                      text.split('\n').filter(line => line.trim()).forEach(line => {
                        const cleanLine = stripLeadingBullets(cleanText(line));
                        addText(`• ${cleanLine}`, 10, false, indent);
                      });
                    }
                  };

                  addBulletPoints(keyStrengths);
                }

                // Limitations / Cons (check many possible keys)
                const limitationKeys = ['limitations', 'limitation', 'cons', 'cons_list', 'consList', 'weaknesses', 'drawbacks', 'tradeoffs', 'constraints', 'limitations_list', 'negativePoints', 'negative_points', 'limits', 'concerns', 'concern', 'concerns_list', 'issues', 'issue_list', 'key_limitations', 'keyLimitations', 'key-limitations'];
                let limitations: any = null;
                for (const k of limitationKeys) {
                  if (product[k]) { limitations = product[k]; break; }
                }
                if (limitations) {
                  addText('Limitations:', 10, true, 20);
                  yPos += 2; // Add some spacing

                  const addBulletPoints = (content: any, indent: number = 25) => {
                    if (Array.isArray(content)) {
                      content.forEach((item: any) => {
                        if (item && typeof item === 'object') {
                          Object.entries(item).forEach(([k, v]) => {
                            const key = stripLeadingBullets(cleanText(k));
                            const val = stripLeadingBullets(cleanText(String(v)));
                            addText(`• ${key}: ${val}`, 10, false, indent);
                          });
                        } else if (item) {
                          const line = stripLeadingBullets(cleanText(String(item)));
                          addText(`• ${line}`, 10, false, indent);
                        }
                      });
                    } else if (content && typeof content === 'object') {
                      Object.entries(content).forEach(([k, v]) => {
                        const key = stripLeadingBullets(cleanText(k));
                        const val = stripLeadingBullets(cleanText(String(v)));
                        addText(`• ${key}: ${val}`, 10, false, indent);
                      });
                    } else if (content) {
                      const text = String(content);
                      // Split by newlines and add as separate bullet points
                      text.split('\n').filter(line => line.trim()).forEach(line => {
                        const cleanLine = stripLeadingBullets(cleanText(line));
                        addText(`• ${cleanLine}`, 10, false, indent);
                      });
                    }
                  };

                  addBulletPoints(limitations);
                }

                // Price and Price URL extraction helper
                // Start with any pricing embedded on the product, but prefer pricing saved on the project (tab-specific or global)
                let price = product.price || product.pricing || product.prices || product.offer || product.offers || product.priceReview;
                // Try to read pricing saved with the project: project.pricing may be an object keyed by tabId -> { "Vendor-Model": pricing }
                const projectPricing = project.pricing || project.pricingData || project.pricing_data || null;
                try {
                  const keyForPricing = `${vendor}-${pName}`.trim();
                  if (!price && projectPricing) {
                    // tab-specific
                    if (projectPricing[tabId] && projectPricing[tabId][keyForPricing]) {
                      price = projectPricing[tabId][keyForPricing];
                    } else if (projectPricing[keyForPricing]) {
                      // global pricing map
                      price = projectPricing[keyForPricing];
                    }
                  }
                } catch (e) {
                  // ignore
                }
                let priceStr = '';
                let priceUrl = '';
                const findUrl = (obj: any): string | null => {
                  if (!obj) return null;
                  if (typeof obj === 'string') {
                    const s = obj.trim();
                    if (s.startsWith('http') || s.startsWith('//') || s.includes('http') || s.startsWith('www.')) return s;
                    return null;
                  }
                  if (Array.isArray(obj)) {
                    for (const item of obj) {
                      const u = findUrl(item);
                      if (u) return u;
                    }
                  }
                  if (typeof obj === 'object') {
                    const urlKeys = ['url', 'link', 'productUrl', 'product_url', 'priceUrl', 'price_url', 'href', 'buyLink', 'purchase_link', 'offerUrl', 'offer_url', 'checkout_url'];
                    for (const k of urlKeys) {
                      const v = obj[k];
                      if (typeof v === 'string' && v.startsWith('http')) return v;
                    }
                    // check nested common shapes
                    if (obj.offers) return findUrl(obj.offers);
                    if (obj[0]) return findUrl(obj[0]);
                    // look for any string value that looks like a url
                    for (const v of Object.values(obj)) {
                      if (typeof v === 'string' && v.startsWith('http')) return v;
                    }
                  }
                  return null;
                };

                if (price) {
                  if (typeof price === 'object') {
                    priceStr = price.amount ? `${price.amount} ${price.currency || ''}` : (price.price || price.amount || '');
                    priceUrl = findUrl(price) || '';
                  } else {
                    priceStr = String(price);
                  }
                }

                // Also look for price URL or product purchase links on the product object
                if (!priceUrl) {
                  priceUrl = findUrl(product) || '';
                }

                // If we still don't have a priceUrl, try to find a vendor-specific link
                const findVendorLink = (obj: any, vendorName?: string): string | null => {
                  if (!obj || !vendorName) return null;
                  const vn = String(vendorName).toLowerCase();

                  // If the object has a direct link and a source/vendor that matches, prefer it
                  if (typeof obj.link === 'string' && (String(obj.source || '').toLowerCase().includes(vn) || String(obj.vendor || '').toLowerCase().includes(vn))) {
                    return obj.link;
                  }

                  // If the object itself contains a url-like field, and source/vendor matches, return it
                  const urlKeys = ['url', 'link', 'productUrl', 'product_url', 'priceUrl', 'price_url', 'href', 'buyLink', 'purchase_link', 'offerUrl', 'offer_url', 'checkout_url'];
                  for (const k of urlKeys) {
                    if (typeof obj[k] === 'string' && String(obj.source || '').toLowerCase().includes(vn)) return obj[k];
                    if (typeof obj[k] === 'string' && String(obj.vendor || '').toLowerCase().includes(vn)) return obj[k];
                  }

                  // Search arrays like results/pricing for an entry that matches the vendor
                  const candidates = Array.isArray(obj.results) ? obj.results : (Array.isArray(obj.pricing?.results) ? obj.pricing.results : (Array.isArray(obj.priceReview?.results) ? obj.priceReview.results : null));
                  if (Array.isArray(candidates)) {
                    for (const entry of candidates) {
                      const src = String(entry.source || entry.vendor || '').toLowerCase();
                      if (src.includes(vn)) {
                        // prefer explicit link fields
                        if (typeof entry.link === 'string') return entry.link;
                        const maybe = findUrl(entry);
                        if (maybe) return maybe;
                      }
                    }
                  }

                  // As a last resort, if the object has nested entries, try to find any url where the entry's source/vendor matches
                  const searchNested = (o: any): string | null => {
                    if (!o || typeof o !== 'object') return null;
                    if (Array.isArray(o)) {
                      for (const it of o) {
                        const r = searchNested(it);
                        if (r) return r;
                      }
                    } else {
                      if (typeof o.source === 'string' && String(o.source).toLowerCase().includes(vn) && typeof o.link === 'string') return o.link;
                      if (typeof o.vendor === 'string' && String(o.vendor).toLowerCase().includes(vn) && typeof o.link === 'string') return o.link;
                      for (const val of Object.values(o)) {
                        if (typeof val === 'string' && (val.startsWith('http') || val.startsWith('www.'))) return val as string;
                        if (typeof val === 'object') {
                          const r = searchNested(val);
                          if (r) return r;
                        }
                      }
                    }
                    return null;
                  };

                  return searchNested(obj) || null;
                };

                if (!priceUrl && vendor) {
                  const vendorFound = findVendorLink(product, vendor);
                  if (vendorFound) priceUrl = vendorFound;
                }

                // Try to extract price URL from priceReview/pricing results if present
                const extractUrlFromResults = (r: any) => {
                  if (!r) return null;
                  if (Array.isArray(r)) {
                    for (const item of r) {
                      if (!item) continue;
                      if (typeof item === 'string' && (item.startsWith('http') || item.includes('http') || item.startsWith('www.'))) return item;
                      if (item.link && (String(item.link).startsWith('http') || String(item.link).includes('http'))) return String(item.link);
                      if (item.url && (String(item.url).startsWith('http') || String(item.url).includes('http'))) return String(item.url);
                    }
                  } else if (typeof r === 'object') {
                    if (r.link && (String(r.link).startsWith('http') || String(r.link).includes('http'))) return String(r.link);
                    if (r.url && (String(r.url).startsWith('http') || String(r.url).includes('http'))) return String(r.url);
                    // check nested
                    for (const val of Object.values(r)) {
                      if (typeof val === 'string' && (val.startsWith('http') || val.includes('http') || val.startsWith('www.'))) return val;
                      if (Array.isArray(val)) {
                        const u = extractUrlFromResults(val);
                        if (u) return u;
                      }
                    }
                  }
                  return null;
                };

                if (!priceUrl) {
                  // try product.priceReview or product.pricing structures
                  const pr = (product as any).priceReview || (product as any).pricing || (product as any).price || (product as any).prices;
                  const found = extractUrlFromResults(pr);
                  if (found) priceUrl = found;
                }

                if (!priceUrl && product.priceReview && Array.isArray(product.priceReview.results)) {
                  const found = extractUrlFromResults(product.priceReview.results);
                  if (found) priceUrl = found;
                }

                if (!priceUrl && (product as any).pricing && Array.isArray((product as any).pricing.results)) {
                  const found = extractUrlFromResults((product as any).pricing.results);
                  if (found) priceUrl = found;
                }

                // Pricing export removed (per user request)

                // Image (print clickable URL)
                const topImage = product.topImage || product.top_image || product.topImageUrl || product.top_image_url || product.image || product.images;
                if (topImage) {
                  const url = typeof topImage === 'string' ? topImage : (topImage.url || topImage.src || (Array.isArray(topImage) && topImage[0]) || '');
                  if (url) addInlineLink('Image URL:', url, 10, 20);
                }

                // Specs
                if (product.specifications && Object.keys(product.specifications).length > 0) {
                  addText("Specifications:", 10, true, 20);
                  Object.entries(product.specifications).forEach(([k, v]) => {
                    // If value is object/array, stringify or join
                    let valDisplay = v;
                    if (typeof v === 'object') {
                      valDisplay = Array.isArray(v) ? v.join(', ') : JSON.stringify(v);
                    }
                    addText(`- ${k}: ${valDisplay}`, 10, false, 25);
                  });
                }
              });
            }
            yPos += 3;
          }
        });
      }

      // Feedback section intentionally omitted from PDF export

      // Save PDF

      const safeName = (projectName || 'project').replace(/[^a-z0-9]/gi, '_').toLowerCase();
      doc.save(`${safeName}.pdf`);

      toast({
        title: "Export Successful",
        description: "Project exported to PDF successfully.",
      });

    } catch (error: any) {
      console.error('Export error:', error);
      toast({
        title: "Export Failed",
        description: error.message || "Failed to export project",
        variant: "destructive",
      });
    } finally {
      setExportingId(null);
    }
  };

  const handleProjectOpen = async (projectId: string) => {
    onOpenChange(false);
    onProjectSelect(projectId);
  };

  // Fetch projects when dialog opens
  useEffect(() => {
    if (open) {
      fetchProjects();
    }
  }, [open]);

  return (
    <Dialog open={open} onOpenChange={onOpenChange}>
      <DialogTrigger asChild>
        {children}
      </DialogTrigger>
      <DialogContent className="max-w-4xl max-h-[80vh] flex flex-col bg-gradient-to-br from-[#F5FAFC]/90 to-[#EAF6FB]/90 dark:from-slate-900/90 dark:to-slate-900/50 backdrop-blur-2xl border-white/20 dark:border-slate-700/30 shadow-2xl [&>button]:top-8 [&>button]:right-8 [&>button]:text-slate-700 [&>button]:hover:bg-transparent [&>button]:hover:text-slate-900 [&>button>svg]:w-5 [&>button>svg]:h-5 [&>button>svg]:transition-transform [&>button:hover>svg]:scale-125 dark:[&>button]:text-slate-400 dark:[&>button]:hover:bg-transparent dark:[&>button]:hover:text-slate-100">
        <DialogHeader className="sm:text-center items-center justify-center">
          <DialogTitle className="text-2xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-[#0F6CBD] to-[#004E8C]">Projects</DialogTitle>
          <DialogDescription className="sr-only">Select a project from the list to open it.</DialogDescription>
        </DialogHeader>

        <div className="flex-1 overflow-y-auto pr-2 custom-no-scrollbar">
          {loading ? (
            <div className="flex items-center justify-center py-8">
              <Loader2 className="h-8 w-8 animate-spin" />
              <span className="ml-2">Loading projects...</span>
            </div>
          ) : projects.length === 0 ? (
            <div className="text-center py-8 text-muted-foreground">
              <FileText className="h-12 w-12 mx-auto mb-4 opacity-50" />
              <p>No saved projects found</p>
              <p className="text-sm">Create your first project by working on requirements and clicking Save</p>
            </div>
          ) : (
            <div className="space-y-4 px-3 pt-4">
              {projects.map((project) => (
                <Card key={project.id} className="hover:shadow-lg transition-all duration-300 bg-gradient-to-br from-[#F5FAFC]/50 to-[#EAF6FB]/50 dark:from-slate-800/60 dark:to-slate-800/30 backdrop-blur-sm border-white/30 dark:border-slate-700/30 hover:bg-white/60 dark:hover:bg-slate-800/60 hover:scale-[1.02]">
                  <CardContent className="py-4">
                    <div className="flex items-center justify-between">
                      <div className="flex-1">
                        <CardTitle className="text-lg mb-1">{project.projectName || project.project_name}</CardTitle>
                      </div>
                      <div className="flex items-center gap-2">
                        <Button
                          onClick={() => handleProjectOpen(project.id)}
                          className="btn-glass-primary transition-transform hover:scale-110 hover:shadow-none bg-[#2B95D6] hover:bg-[#0F6CBD]"
                        >
                          Open
                        </Button>

                        <AlertDialog>
                          <AlertDialogTrigger asChild>
                            <Button
                              variant="ghost"
                              size="sm"
                              className="text-destructive hover:text-destructive hover:bg-transparent transition-transform hover:scale-110"
                            >
                              <Trash2 className="h-4 w-4" />
                            </Button>
                          </AlertDialogTrigger>
                          <AlertDialogContent>
                            <AlertDialogHeader>
                              <AlertDialogTitle>Delete Project</AlertDialogTitle>
                              <AlertDialogDescription>
                                Are you sure you want to delete "{project.projectName || project.project_name}"?
                                This action cannot be undone.
                              </AlertDialogDescription>
                            </AlertDialogHeader>
                            <AlertDialogFooter>
                              <AlertDialogCancel>Cancel</AlertDialogCancel>
                              <AlertDialogAction
                                onClick={() => deleteProject(project.id, project.projectName || project.project_name || 'Project')}
                                className="bg-destructive hover:bg-destructive/90"
                              >
                                Delete
                              </AlertDialogAction>
                            </AlertDialogFooter>
                          </AlertDialogContent>
                        </AlertDialog>

                        <AlertDialog>
                          <AlertDialogTrigger asChild>
                            <Button
                              variant="ghost"
                              size="sm"
                              disabled={exportingId === project.id}
                              className="hover:bg-transparent transition-transform hover:scale-110 hover:text-current"
                            >
                              {exportingId === project.id ? (
                                <Loader2 className="h-4 w-4 animate-spin" />
                              ) : (
                                <Download className="h-4 w-4" />
                              )}
                            </Button>
                          </AlertDialogTrigger>
                          <AlertDialogContent>
                            <AlertDialogHeader>
                              <AlertDialogTitle>Export Project</AlertDialogTitle>
                              <AlertDialogDescription>
                                Are you sure you want to export "{project.projectName || project.project_name}" to PDF?
                                This will download a document containing the full project history and details.
                              </AlertDialogDescription>
                            </AlertDialogHeader>
                            <AlertDialogFooter>
                              <AlertDialogCancel>Cancel</AlertDialogCancel>
                              <AlertDialogAction
                                onClick={() => handleExportProject(project.id, project.projectName || project.project_name || 'Project')}
                              >
                                Yes, Export
                              </AlertDialogAction>
                            </AlertDialogFooter>
                          </AlertDialogContent>
                        </AlertDialog>

                      </div>
                    </div>
                  </CardContent>
                </Card>
              ))}
            </div>
          )}
        </div>
      </DialogContent>
    </Dialog>
  );
};

export default ProjectListDialog;