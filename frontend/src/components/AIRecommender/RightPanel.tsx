import React, { useEffect, useState, useMemo, useRef } from "react";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Separator } from "@/components/ui/separator";
import { ScrollArea, ScrollBar } from "@/components/ui/scroll-area";
import { Tabs, TabsContent, TabsList, TabsTrigger } from "@/components/ui/tabs";
import { AnalysisResult, RequirementSchema, ValidationResult, AnalysisImageResult, ProductImage, VendorLogo, IdentifiedItem, StandardsCompliance, MatchedRequirement } from "./types";
import { getProductPriceReview, submitFeedback as submitFeedbackApi, BASE_URL } from "./api";
import { Trophy, Loader2, ChevronLeft, ChevronRight, Eye, Send, Play } from "lucide-react";
import { Button } from "@/components/ui/button";
import { Textarea } from "@/components/ui/textarea";
import { useToast } from "@/hooks/use-toast";
import ReactMarkdown from "react-markdown";
import "./ImageComponents.css";

// Helper functions
const cleanImageUrl = (url: string): string => {
  if (!url) return "";
  // Check and remove the non-standard x-raw-image:/// scheme
  const prefix = "x-raw-image:///";
  if (url.startsWith(prefix)) {
    return url.substring(prefix.length);
  }
  return url;
};

const getAbsoluteImageUrl = (url: string | undefined | null): string | undefined => {
  if (!url) return undefined;
  const clean = cleanImageUrl(url);
  if (clean.startsWith("http") || clean.startsWith("data:")) return clean;

  const baseUrl = BASE_URL.endsWith("/") ? BASE_URL.slice(0, -1) : BASE_URL;
  const path = clean.startsWith("/") ? clean : `/${clean}`;
  return `${baseUrl}${path}`;
};
// Type that allows images to be either objects or strings
type VendorImage = { fileName: string; url: string; productKey?: string } | string;
type VendorInfo = { name: string; logoUrl: string | null; images: VendorImage[] };

// Image Gallery Component
interface ImageGalleryProps {
  images: ProductImage[];
  isOpen: boolean;
  onClose: () => void;
  productName: string;
}

const ImageGallery: React.FC<ImageGalleryProps> = ({ images, isOpen, onClose, productName }) => {
  if (!isOpen) return null;

  // Helper function for quality badges
  const getQualityBadge = (source: string) => {
    switch (source) {
      case 'google_cse': return '🟢 Official';
      case 'serpapi': return '🟡 Verified';
      case 'serper': return '🟠 General';
      default: return '⚫ Unknown';
    }
  };

  return (
    <div className="fixed inset-0 bg-black bg-opacity-80 z-50 flex items-center justify-center p-4">
      <div className="bg-white dark:bg-slate-800 max-w-6xl max-h-[90vh] rounded-lg overflow-hidden">
        <div className="flex items-center justify-between p-4 border-b border-gray-200 dark:border-gray-700">
          <h3 className="text-xl font-bold text-gray-900 dark:text-gray-100">{productName} - All Images</h3>
          <button
            onClick={onClose}
            className="text-2xl font-bold text-gray-500 hover:text-gray-700 dark:text-gray-400 dark:hover:text-gray-200"
          >
            ×
          </button>
        </div>

        <div className="grid grid-cols-1 md:grid-cols-2 lg:grid-cols-3 xl:grid-cols-4 gap-4 p-4 max-h-[70vh] overflow-y-auto">
          {images.map((image, index) => {
            const finalSrc = getAbsoluteImageUrl(image.url) || "";

            return (
              <div key={index} className="bg-gray-50 dark:bg-gray-700 rounded-lg overflow-hidden">
                <img
                  src={finalSrc}
                  alt={image.title}
                  className="w-full h-48 object-cover"
                />
                <div className="p-3">
                  <p className="text-sm font-medium text-gray-900 dark:text-gray-100 truncate">{image.title}</p>
                  <div className="flex items-center justify-between mt-2">
                    <span className="text-xs px-2 py-1 bg-gray-200 dark:bg-gray-600 rounded">{getQualityBadge(image.source)}</span>
                    <span className="text-xs text-gray-500 dark:text-gray-400">{image.domain}</span>
                  </div>
                </div>
              </div>
            );
          })}
        </div>
      </div>
    </div>
  );
};

export type RightPanelProps = {
  productType?: string;
  analysisResult: AnalysisResult;
  validationResult?: ValidationResult;
  requirementSchema?: RequirementSchema;
  identifiedItems?: IdentifiedItem[] | null; // Added
  onRunSearch?: (sampleInput: string) => void; // Deprecated - use onPopulateInput
  onPopulateInput?: (sampleInput: string) => void; // NEW: Populate input without auto-submit
  isDocked: boolean;
  setIsDocked: React.Dispatch<React.SetStateAction<boolean>>;
  onPricingDataUpdate?: (priceReviewMap: Record<string, PriceReview>) => void;
  onScroll?: (scrollTop: number) => void;
  initialScrollTop?: number;
};

// PriceReviewResult now includes the 'link' name as nullable
interface PriceReviewResult {
  link: string | null;
  price: string | null;
  reviews: number | null;
  source: string | null;
}

interface PriceReview {
  results: PriceReviewResult[];
}

const RightPanel: React.FC<RightPanelProps> = ({
  productType = "",
  analysisResult,
  validationResult,
  requirementSchema,
  identifiedItems,
  onRunSearch, // Deprecated
  onPopulateInput, // NEW
  isDocked,
  setIsDocked,
  onPricingDataUpdate,
  onScroll,
  initialScrollTop
}) => {
  const [vendors, setVendors] = useState<VendorInfo[]>([]);
  const [hoveredImage, setHoveredImage] = useState<string | null>(null);
  const [priceReviewMap, setPriceReviewMap] = useState<Record<string, PriceReview>>({});
  // Use only embedded analysis images (no external fetching)
  const [analysisImages, setAnalysisImages] = useState<Record<string, AnalysisImageResult>>({});
  const [imageGalleryOpen, setImageGalleryOpen] = useState<{ images: ProductImage[]; productName: string } | null>(null);
  const hasAutoUndocked = useRef(false);
  const scrollContainerRef = useRef<HTMLDivElement>(null);
  const { toast } = useToast();

  useEffect(() => {
    if (initialScrollTop && scrollContainerRef.current) {
      // Use timeout to ensure content is fully rendered before scrolling
      setTimeout(() => {
        if (scrollContainerRef.current) {
          // For ScrollArea, we need to scroll the viewport div
          scrollContainerRef.current.scrollTop = initialScrollTop;
        }
      }, 100);
    }
  }, []);

  // Notify parent component when pricing data updates
  useEffect(() => {
    if (onPricingDataUpdate && Object.keys(priceReviewMap).length > 0) {
      onPricingDataUpdate(priceReviewMap);
    }
  }, [priceReviewMap, onPricingDataUpdate]);

  // Auto-undock when identified items are available
  useEffect(() => {
    if (identifiedItems && identifiedItems.length > 0 && isDocked && !hasAutoUndocked.current) {
      setIsDocked(false);
      hasAutoUndocked.current = true;
    }
  }, [identifiedItems, isDocked, setIsDocked]);

  // State for identified item images (fetched as fallback if backend doesn't provide)
  const [identifiedItemImages, setIdentifiedItemImages] = useState<Record<number, string>>({});

  // Fetch generic images for identified items that don't have imageUrl
  useEffect(() => {
    if (!identifiedItems || identifiedItems.length === 0) return;

    const fetchMissingImages = async () => {
      const itemsNeedingImages = identifiedItems.filter(
        (item) => !item.imageUrl && (item.name || item.category)
      );

      if (itemsNeedingImages.length === 0) return;

      console.log(`[RightPanel] Fetching images for ${itemsNeedingImages.length} items without imageUrl`);

      const newImages: Record<number, string> = {};

      await Promise.all(
        itemsNeedingImages.map(async (item) => {
          const productName = item.name || item.category;
          if (!productName) return;

          try {
            const encodedType = encodeURIComponent(productName);
            const response = await fetch(`${BASE_URL}/api/generic_image/${encodedType}`, {
              credentials: "include",
            });

            if (response.ok) {
              const data = await response.json();
              if (data.success && data.image?.url) {
                newImages[item.number] = data.image.url;
              }
            }
          } catch (error) {
            console.warn(`[RightPanel] Failed to fetch image for item #${item.number} (${productName})`);
          }
        })
      );

      if (Object.keys(newImages).length > 0) {
        console.log(`[RightPanel] Fetched ${Object.keys(newImages).length} images for identified items`);
        setIdentifiedItemImages((prev) => ({ ...prev, ...newImages }));
      }
    };

    fetchMissingImages();
  }, [identifiedItems]);

  // Helper to get image URL for an identified item (checks both item.imageUrl and fetched fallback)
  const getIdentifiedItemImageUrl = (item: IdentifiedItem): string | undefined => {
    // First check if item has imageUrl from backend
    if (item.imageUrl) return item.imageUrl;
    // Then check if we fetched it as fallback
    if (identifiedItemImages[item.number]) return identifiedItemImages[item.number];
    return undefined;
  };

  type FeedbackType = "positive" | "negative" | null;
  interface FeedbackEntry {
    type: FeedbackType;
    comment: string;
    loading: boolean;
    submitted: boolean;
    response?: string;
  }
  const [feedbackState, setFeedbackState] = useState<Record<string, FeedbackEntry>>({});

  const getProductKey = (vendor: string, productName: string) => `${vendor}-${productName}`;

  // Function to fetch images for analysis results
  // No external fetching: analysisImages will be prefilled from embedded project data when present

  const setFeedbackType = (key: string, type: FeedbackType) => {
    setFeedbackState((prev) => ({
      ...prev,
      [key]: { type, comment: prev[key]?.comment ?? "", loading: false, submitted: false, response: undefined },
    }));
  };

  const setFeedbackComment = (key: string, comment: string) => {
    setFeedbackState((prev) => ({
      ...prev,
      [key]: { type: prev[key]?.type ?? null, comment, loading: false, submitted: false, response: undefined },
    }));
  };

  const submitFeedback = async (key: string, vendor: string, productName: string) => {
    const entry = feedbackState[key] ?? { type: null, comment: "", loading: false, submitted: false };
    if (!entry.type && !entry.comment.trim()) {
      toast({ title: "Please provide thumbs up/down or a comment." });
      return;
    }
    setFeedbackState((prev) => ({ ...prev, [key]: { ...entry, loading: true } }));
    try {
      // Try to include projectId from analysisResult if available so backend can attach feedback to project
      const projectId = (analysisResult as any)?.projectId || (analysisResult as any)?.project_id || undefined;
      const response = await submitFeedbackApi(entry.type ?? null, `[${vendor} - ${productName}] ${entry.comment ?? ""}`, projectId);
      setFeedbackState((prev) => ({ ...prev, [key]: { ...entry, loading: false, submitted: true, response } }));
      toast({ title: "Thanks for your feedback!" });
    } catch (err: any) {
      setFeedbackState((prev) => ({ ...prev, [key]: { ...entry, loading: false } }));
      toast({ title: "Failed to send feedback", description: err?.message ?? "Please try again later." });
    }
  };

  const handleFeedbackKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>, key: string, vendor: string, productName: string) => {
    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault();
      submitFeedback(key, vendor, productName);
    }
  };

  useEffect(() => {
    // Prefer vendor logos/images embedded in the analysisResult (for saved projects)
    if (!analysisResult?.overallRanking?.rankedProducts || vendors.length > 0) return;

    // Use productsToDisplay instead of hardcoded exact match filter
    // This ensures we show images/logos for whichever products are actually displayed (exact or approximate)
    const matchedProducts = productsToDisplay;

    // Try to build vendors list from analysisResult if logos/images were saved with results
    const embeddedVendorsMap: Record<string, VendorInfo> = {};
    matchedProducts.forEach((product) => {
      const name = product.vendor;
      if (!name) return;
      if (!embeddedVendorsMap[name]) embeddedVendorsMap[name] = { name, logoUrl: null, images: [] };

      // Use vendorLogo / vendor_logo if present
      const vlogo = (product as any).vendorLogo || (product as any).vendor_logo;
      if (vlogo && (vlogo.url || vlogo.logo)) {
        const url = vlogo.url || vlogo.logo;
        embeddedVendorsMap[name].logoUrl = cleanImageUrl(url);
      }

      // Use topImage / top_image and allImages / all_images if present
      const topImg = (product as any).topImage || (product as any).top_image;
      if (topImg && topImg.url) {
        embeddedVendorsMap[name].images.push(cleanImageUrl(topImg.url));
      }
      const allImgs = (product as any).allImages || (product as any).all_images;
      if (Array.isArray(allImgs) && allImgs.length > 0) {
        allImgs.forEach((img: any) => {
          if (img && img.url) embeddedVendorsMap[name].images.push(cleanImageUrl(img.url));
        });
      }
    });

    const embeddedVendors = Object.values(embeddedVendorsMap).filter(v => v.logoUrl || (v.images && v.images.length > 0));
    if (embeddedVendors.length > 0) {
      // Use embedded vendor/logo data from analysisResult (e.g., when loading saved project)
      setVendors(embeddedVendors.map(v => ({ name: v.name, logoUrl: v.logoUrl, images: v.images })));
      // Also prefill analysisImages map for products that have top_image in analysisResult
      const preloadedImages: Record<string, AnalysisImageResult> = {};
      matchedProducts.forEach((product) => {
        const key = `${product.vendor}-${product.productName}`;
        const top = (product as any).topImage || (product as any).top_image;
        const all = (product as any).allImages || (product as any).all_images || [];
        const vlogo = (product as any).vendorLogo || (product as any).vendor_logo || null;
        if (top || (all && all.length > 0) || vlogo) {
          preloadedImages[key] = ({
            topImage: top ? { ...top, url: cleanImageUrl(top.url) } : null,
            vendorLogo: vlogo ? { ...vlogo, url: cleanImageUrl(vlogo.url || vlogo.logo) } : null,
            allImages: Array.isArray(all) ? all.map((img: any) => ({ ...img, url: cleanImageUrl(img.url) })) : []
          } as unknown) as AnalysisImageResult;
        }
      });
      setAnalysisImages(prev => ({ ...prev, ...preloadedImages }));
      return;
    }

    // No embedded vendor media found — do not perform external vendor/logo fetching
    setVendors([]);
  }, [analysisResult, vendors.length]);

  // Submodel mapping and other external image heuristics removed: we only use embedded media now.

  // External image fetching removed — we rely on embedded analysis images only.

  // Auto-undock when analysis data becomes available (only once)
  useEffect(() => {
    if (analysisResult?.vendorAnalysis?.vendorMatches?.length > 0 && isDocked && !hasAutoUndocked.current) {
      setIsDocked(false);
      hasAutoUndocked.current = true;
    }
  }, [analysisResult, isDocked, setIsDocked]);

  useEffect(() => {
    if (!analysisResult?.overallRanking?.rankedProducts) return;

    // Use productsToDisplay instead of hardcoded exact match filter
    // This ensures we fetch pricing for whichever products are actually displayed (exact or approximate)
    const matchedProducts = productsToDisplay;

    // Build a priceReviewMap but prefer embedded pricing in saved project data
    const map: Record<string, PriceReview> = {};
    matchedProducts.forEach((product) => {
      const key = `${product.vendor}-${product.productName}`;
      // If product already contains pricing information (saved with project), use it
      if ((product as any).price || (product as any).pricing || (product as any).priceReview) {
        const embedded: PriceReviewResult[] = [];
        if ((product as any).price) {
          embedded.push({ price: (product as any).price, reviews: (product as any).reviews ?? null, source: 'embedded', link: null });
        }
        if ((product as any).pricing && Array.isArray((product as any).pricing.results)) {
          (product as any).pricing.results.forEach((r: any) => embedded.push({ price: r.price ?? null, reviews: r.reviews ?? null, source: r.source ?? null, link: r.link ?? null }));
        }
        if ((product as any).priceReview && Array.isArray((product as any).priceReview.results)) {
          (product as any).priceReview.results.forEach((r: any) => embedded.push({ price: r.price ?? null, reviews: r.reviews ?? null, source: r.source ?? null, link: r.link ?? null }));
        }

        map[key] = { results: embedded };
      } else {
        // Otherwise, attempt to fetch dynamically (fallback)
        map[key] = { results: [] };
        getProductPriceReview(product.productName)
          .then((data: any) => {
            const normalizedResults: PriceReviewResult[] = (data?.results ?? []).map((r: any) => ({
              price: r.price ?? null,
              reviews: r.reviews ?? null,
              source: r.source ?? null,
              link: r.link ?? null,
            }));
            // sort ascending price
            const sortedResults = normalizedResults.sort((a, b) => {
              const getNumericPrice = (price: string | null) => {
                if (!price) return Infinity;
                const match = price.match(/\d+([.,]\d+)?/);
                return match ? parseFloat(match[0].replace(",", "")) : Infinity;
              };
              return getNumericPrice(a.price) - getNumericPrice(b.price);
            });
            setPriceReviewMap(prev => ({ ...prev, [key]: { results: sortedResults } }));
          })
          .catch(() => {
            // leave empty if fetch fails
          });
      }
    });

    setPriceReviewMap(prev => ({ ...prev, ...map }));
  }, [analysisResult]);

  // Enhanced normalization function for better name matching
  const normalizeText = (name: string): string => {
    if (!name) return "";
    return name
      .toLowerCase()
      .replace(/[\s\-_\.\+\&\(\)\[\]\{\}]/g, "") // Remove spaces, dashes, underscores, dots, plus, ampersand, brackets
      .replace(/[^a-z0-9]/g, "") // Remove any non-alphanumeric characters
      .trim();
  };

  // Create multiple normalized variations of a name for fuzzy matching
  const createNameVariations = (name: string): string[] => {
    const variations = new Set<string>();
    const normalized = normalizeText(name);

    // Add the fully normalized version
    variations.add(normalized);

    // Add version without numbers
    variations.add(normalized.replace(/[0-9]/g, ""));

    // Add version with just letters and first number group
    const firstNumberMatch = name.match(/\d+/);
    if (firstNumberMatch) {
      variations.add(normalizeText(name.split(firstNumberMatch[0])[0] + firstNumberMatch[0]));
    }

    // Add first word only
    const firstWord = name.split(/[\s\-_\.]/)[0];
    if (firstWord) {
      variations.add(normalizeText(firstWord));
    }

    return Array.from(variations).filter(v => v.length > 0);
  };

  const normalizeVendorName = (name: string) => normalizeText(name);

  const vendorLogoMap = useMemo(() => {
    const out: { [name: string]: string | null } = {};
    vendors.forEach(({ name, logoUrl }) => {
      out[normalizeVendorName(name)] = logoUrl ?? null;
    });
    return out;
  }, [vendors]);

  // Thresholds for match quality
  // Exact matches: No score threshold - show ALL products where requirementsMatch === true
  // Approximate matches: Minimum score of 50% to ensure reasonable quality
  const APPROXIMATE_THRESHOLD = 0; // TEMP: Set to 0 for debugging (was 50)

  const requirementsMatchMap = useMemo(() => {
    const map = new Map<string, boolean>();
    analysisResult?.vendorAnalysis?.vendorMatches?.forEach((match) => {
      map.set(`${match.vendor}-${match.productName}`, !!match.requirementsMatch);
    });
    return map;
  }, [analysisResult]);

  // Split products into exact and approximate matches
  const { exactMatches, approximateMatches, displayMode } = useMemo(() => {
    if (!analysisResult?.overallRanking?.rankedProducts) {
      return { exactMatches: [], approximateMatches: [], displayMode: 'exact' as const };
    }

    // Exact matches: ALL products where requirementsMatch === true (no score threshold)
    const exact = analysisResult.overallRanking.rankedProducts.filter(
      (product) => {
        const matchStatus = requirementsMatchMap.get(`${product.vendor}-${product.productName}`) ?? product.requirementsMatch;
        return matchStatus === true;  // No score requirement for exact matches
      }
    );

    // Approximate matches: requirementsMatch === false AND score >= 50%
    const approximate = analysisResult.overallRanking.rankedProducts.filter(
      (product) => {
        const matchStatus = requirementsMatchMap.get(`${product.vendor}-${product.productName}`) ?? product.requirementsMatch;
        return matchStatus === false && (product.overallScore ?? 0) >= APPROXIMATE_THRESHOLD;
      }
    );

    // Fallback logic: show exact if available, otherwise approximate
    const mode = exact.length > 0 ? 'exact' : 'approximate';

    console.log('[RightPanel] Debug:', {
      totalProducts: analysisResult.overallRanking.rankedProducts.length,
      exactMatchesCount: exact.length,
      approximateMatchesCount: approximate.length,
      displayMode: mode,
      sampleProduct: analysisResult.overallRanking.rankedProducts[0],
      allProductScores: analysisResult.overallRanking.rankedProducts.map(p => ({
        name: p.productName,
        score: p.overallScore,
        requirementsMatch: p.requirementsMatch
      }))
    });

    return { exactMatches: exact, approximateMatches: approximate, displayMode: mode };
  }, [analysisResult, requirementsMatchMap]);

  // Determine which products to display based on fallback logic
  const productsToDisplay = displayMode === 'exact' ? exactMatches : approximateMatches;

  const filteredAnalysisResult = analysisResult
    ? {
      ...analysisResult,
      displayMode, // Add display mode to result
      vendorAnalysis: {
        ...analysisResult.vendorAnalysis,
        vendorMatches: (analysisResult.vendorAnalysis?.vendorMatches ?? []).filter((match) => {
          const isExactMatch = match.requirementsMatch === true;
          const isApproximateMatch = match.requirementsMatch === false && (match.matchScore ?? 0) >= APPROXIMATE_THRESHOLD;

          // Show exact matches if available, otherwise approximate
          if (displayMode === 'exact') {
            return isExactMatch;
          } else {
            return isApproximateMatch;
          }
        }),
      },
      overallRanking: {
        ...analysisResult.overallRanking,
        rankedProducts: productsToDisplay.map((product, index) => ({
          ...product,
          rank: index + 1,
          requirementsMatch: requirementsMatchMap.get(`${product.vendor}-${product.productName}`) ?? product.requirementsMatch,
        })),
      },
    }
    : null;

  const finalAnalysisResult = filteredAnalysisResult;

  // Determine what to render
  const hasAnalysisData = !!(
    finalAnalysisResult?.vendorAnalysis?.vendorMatches?.length ||
    finalAnalysisResult?.overallRanking?.rankedProducts?.length
  );
  // Show identified items if they exist, regardless of analysis data (will be handled via tabs if both exist)
  const showIdentifiedItems = identifiedItems && identifiedItems.length > 0;
  // Reuse hasAnalysisData for showAnalysis
  const showAnalysis = hasAnalysisData;

  const renderIdentifiedItemsList = () => (
    <div className="space-y-4 max-w-3xl mx-auto pt-2">
      {identifiedItems?.map((item) => {
        // Use helper to get image URL (checks backend-provided + frontend fallback)
        const itemImageUrl = getIdentifiedItemImageUrl(item);
        return (
          <Card key={item.number} className="overflow-hidden border-border/50 shadow-sm hover:shadow-md transition-all">
            <CardContent className="p-0 flex flex-col md:flex-row gap-4">
              {/* Image Section */}
              <div className="w-full md:w-48 h-48 bg-muted/30 flex-shrink-0 relative group">
                {itemImageUrl ? (
                  <img src={getAbsoluteImageUrl(itemImageUrl)} alt={item.name} className="w-full h-full object-contain p-2" />
                ) : (
                  <div className="w-full h-full flex items-center justify-center text-muted-foreground">
                    <Trophy className="w-12 h-12 opacity-20" />
                  </div>
                )}
                <div className="absolute top-2 left-2 bg-background/90 backdrop-blur px-2 py-1 rounded text-xs font-mono font-medium border shadow-sm">
                  #{item.number}
                </div>
                <div className="absolute top-2 right-2 bg-primary/10 text-primary-foreground px-2 py-1 rounded text-xs font-medium border border-primary/20 shadow-sm">
                  {item.type}
                </div>
              </div>

              {/* Content Section */}
              <div className="flex-1 p-4 flex flex-col">
                <div className="flex-1">
                  <h3 className="text-xl font-bold mb-1 text-primary">{item.name}</h3>
                  <div className="text-sm text-muted-foreground mb-3 font-medium bg-muted/30 inline-block px-2 py-1 rounded">
                    {item.category}
                  </div>

                  {item.keySpecs && (
                    <div className="mt-2 text-sm text-foreground/80 leading-relaxed">
                      <strong>Specs:</strong> {item.keySpecs}
                    </div>
                  )}
                </div>

                <div className="mt-4 flex items-center justify-end gap-3">
                  <Button
                    onClick={() => onPopulateInput && item.sampleInput && onPopulateInput(item.sampleInput)}
                    className="gap-2 shadow-lg hover:shadow-xl transition-all hover:scale-105 active:scale-95 font-semibold"
                  >
                    <Play className="w-4 h-4 fill-current" />
                    Populate Input
                  </Button>
                </div>
              </div>
            </CardContent>
          </Card>
        );
      })}
    </div>
  );

  // DEBUG: Log render decisions
  console.log('[RightPanel] Render decision:', {
    hasAnalysisData,
    showAnalysis,
    showIdentifiedItems,
    vendorMatchesCount: finalAnalysisResult?.vendorAnalysis?.vendorMatches?.length || 0,
    rankedProductsCount: finalAnalysisResult?.overallRanking?.rankedProducts?.length || 0,
    productsToDisplayCount: productsToDisplay.length
  });

  // If no matches and no identified items, render minimal docked panel
  if (!showAnalysis && !showIdentifiedItems) {
    return (
      <div className="w-full h-full flex flex-col glass-sidebar text-foreground border-l border-border overflow-hidden sticky top-0 right-0 z-20" style={{ minWidth: 0 }}>
        {/* Header space - dock button moved to corner */}
        <div className="flex items-center justify-end py-4 px-3 flex-shrink-0">
          {/* Dock button moved to corner */}
        </div>
        <div style={{ flex: 1 }} />
        <style>{`
          .custom-no-scrollbar::-webkit-scrollbar {
            display: none;
          }
          .custom-no-scrollbar {
            -ms-overflow-style: none;
            scrollbar-width: none;
          }
        `}</style>
      </div>
    );
  }

  // Helper for Identified Items rendering (Standalone Mode: ONLY when no analysis data)
  if (showIdentifiedItems && !showAnalysis && !isDocked) {
    return (
      <div className="w-full h-full flex flex-col glass-sidebar text-foreground border-l border-border sticky top-0 right-0 z-20" style={{ minWidth: 0 }}>
        {/* Header */}
        <div className="flex items-center justify-between py-4 px-6 border-b border-border/50 bg-muted/20">
          <div className="flex items-center gap-2">
            <Trophy className="w-5 h-5 text-primary" />
            <h2 className="font-semibold text-lg">Identified Items</h2>
          </div>
          {/* Dock button is external/managed by parent or top-right overlay */}
        </div>

        <ScrollArea
          className="flex-1 p-4"
          viewportRef={scrollContainerRef}
          onScroll={(e) => onScroll?.(e.currentTarget.scrollTop)}
        >
          {renderIdentifiedItemsList()}
        </ScrollArea>
      </div>
    )
  }

  const getRankIcon = (rank: number | undefined) => {
    switch (rank) {
      case 1:
        return "🥇";
      case 2:
        return "🥈";
      case 3:
        return "🥉";
      default:
        return rank ? `#${rank}` : "•";
    }
  };

  // Helper function for quality badges
  const getQualityBadge = (source: string) => {
    switch (source) {
      case 'google_cse': return '🟢 Official';
      case 'serpapi': return '🟡 Verified';
      case 'serper': return '🟠 General';
      default: return '⚫ Unknown';
    }
  };

  // Function to open image gallery
  const openImageGallery = (images: ProductImage[], productName: string) => {
    setImageGalleryOpen({ images, productName });
  };

  const renderMarkdownContent = (content: string | string[] | undefined) => {
    if (!content) return null;
    const markdownText = Array.isArray(content) ? content.join("\n\n") : content;
    return (
      <div
        className="prose prose-sm max-w-none prose-slate dark:prose-invert 
                      prose-headings:text-lg prose-headings:font-bold prose-headings:text-slate-800 dark:prose-headings:text-slate-100 prose-headings:mb-3 prose-headings:mt-4
                      prose-p:text-base prose-p:text-slate-700 dark:prose-p:text-slate-300 prose-p:leading-relaxed prose-p:mb-3
                      prose-strong:text-slate-900 dark:prose-strong:text-slate-100 prose-strong:font-semibold prose-strong:bg-yellow-100 dark:prose-strong:bg-yellow-900/30 prose-strong:px-1 prose-strong:rounded
                      prose-ul:text-base prose-ul:text-slate-700 dark:prose-ul:text-slate-300 prose-ul:mb-3 prose-ul:mt-2
                      prose-li:text-slate-700 dark:prose-li:text-slate-300 prose-li:leading-relaxed prose-li:mb-2 prose-li:pl-1
                      prose-h1:text-xl prose-h1:font-bold prose-h1:text-[#0F6CBD] dark:prose-h1:text-sky-300 prose-h1:mb-4 prose-h1:mt-4 prose-h1:border-b prose-h1:border-[#5FB3E6]/30 prose-h1:pb-2
                      prose-h2:text-lg prose-h2:font-bold prose-h2:text-[#0F6CBD] dark:prose-h2:text-sky-400 prose-h2:mb-3 prose-h2:mt-4
                      prose-h3:text-base prose-h3:font-semibold prose-h3:text-[#0F6CBD] dark:prose-h3:text-sky-500 prose-h3:mb-2 prose-h3:mt-3
                      prose-h4:text-sm prose-h4:font-semibold prose-h4:text-slate-600 dark:prose-h4:text-slate-400 prose-h4:mb-2 prose-h4:mt-2
                      prose-code:text-pink-600 dark:prose-code:text-pink-400 prose-code:bg-pink-50 dark:prose-code:bg-pink-900/30 prose-code:px-1 prose-code:rounded prose-code:font-mono prose-code:text-sm
                      prose-blockquote:border-l-4 prose-blockquote:border-[#5FB3E6] prose-blockquote:bg-[#F5FAFC] dark:prose-blockquote:bg-sky-900/20 prose-blockquote:pl-4 prose-blockquote:py-2 prose-blockquote:italic
                      [&_ul]:ml-2 [&_li]:ml-0 [&_p]:select-text [&_li]:select-text [&_strong]:select-text [&_code]:select-text
                      [&_ul>li]:marker:text-[#5FB3E6] [&_ol>li]:marker:text-[#5FB3E6] [&_ol>li]:marker:font-semibold
                      [&_em]:text-slate-600 dark:[&_em]:text-slate-400 [&_em]:italic [&_em]:font-medium"
      >
        <ReactMarkdown>{markdownText}</ReactMarkdown>
      </div>
    );
  };

  const CircularProgressBarSVG = ({ score }: { score: number }) => {
    const safeScore = Math.max(0, Math.min(100, Number.isFinite(score) ? score : 0));
    const radius = 20;
    const circumference = 2 * Math.PI * radius;
    const offset = circumference * (1 - safeScore / 100);
    const strokeColor = safeScore >= 80 ? "#16a34a" : safeScore >= 60 ? "#f59e0b" : "#ef4444";
    return (
      <svg className="w-full h-full text-foreground" viewBox="0 0 44 44">
        <circle cx="22" cy="22" r={radius} stroke="#e5e7eb" strokeWidth="4" fill="transparent" />
        <circle
          cx="22"
          cy="22"
          r={radius}
          stroke={strokeColor}
          strokeWidth="4"
          strokeLinecap="round"
          fill="transparent"
          strokeDasharray={circumference}
          strokeDashoffset={offset}
          transform="rotate(-90 22 22)"
        />
        <text x="22" y="22" dominantBaseline="middle" textAnchor="middle" fontSize="10" fontWeight="bold" fill="currentColor">
          {`${safeScore}%`}
        </text>
      </svg>
    );
  };

  // Helper to return border color classes for containers based on score
  const getBorderColor = (overallScore: number | undefined) => {
    return "border-[#45A4DE]";
  };

  const { vendorAnalysis, overallRanking } = finalAnalysisResult!;
  const vendorsGrouped = (vendorAnalysis.vendorMatches || []).reduce((acc: { [vendor: string]: typeof vendorAnalysis.vendorMatches }, current) => {
    if (!acc[current.vendor]) acc[current.vendor] = [];
    acc[current.vendor].push(current);
    return acc;
  }, {});
  const vendorNames = Object.keys(vendorsGrouped);

  const RenderVendorLogo: React.FC<{ vendorName: string; size?: number }> = ({ vendorName, size = 22 }) => {
    const normalizedVendorName = normalizeVendorName(vendorName);
    const logoUrl = vendorLogoMap[normalizedVendorName];
    if (!logoUrl) return null;
    const safeUrl = getAbsoluteImageUrl(logoUrl);
    return (
      <img
        src={safeUrl}
        alt={`${vendorName} logo`}
        style={{
          width: size,
          height: size,
          objectFit: "contain",
          borderRadius: 4,
          marginRight: 6,
        }}
        onError={(e) => {
          e.currentTarget.style.display = "none";
        }}
      />
    );
  };

  const getPriceReview = (vendor: string, productName: string): PriceReview => {
    const key = `${vendor}-${productName}`;
    return priceReviewMap[key] || { results: [] };
  };

  return (
    <div className="w-full h-full flex flex-col glass-sidebar text-foreground border-l border-border sticky top-0 right-0 z-20" style={{ minWidth: 0, position: "relative" }}>
      {/* Fixed dock toggle for right panel (above tabs/content) */}
      {/* <Button
        variant="ghost"
        size="icon"
        className="fixed top-20 right-8 z-60 bg-background border shadow-lg hover:bg-background/90"
        onClick={() => setIsDocked(!isDocked)}
        aria-label={isDocked ? "Expand right panel" : "Collapse right panel"}
      >
        {isDocked ? <ChevronLeft /> : <ChevronRight />}
      </Button> */}

      {/* Header space - dock button moved to corner */}
      <div className="flex items-center justify-end py-4 px-3 flex-shrink-0">
        {/* existing header space */}
      </div>


      {/* Docked view: show compact summary/icon */}
      {isDocked ? (
        <div className="flex flex-col items-center justify-center flex-1 py-8">
          <Trophy className="w-8 h-8 text-muted-foreground mb-2" />
          <span className="text-xs text-muted-foreground">Results</span>
        </div>
      ) : (
        <ScrollArea
          className="flex-1 overflow-auto custom-visible-scrollbar min-w-0"
          viewportRef={scrollContainerRef}
          onScroll={(e) => onScroll?.(e.currentTarget.scrollTop)}
        >
          <div className="p-4 space-y-4 min-w-0 flex flex-col">
            <Tabs defaultValue="ranking" className="w-full min-w-0">
              <ScrollArea className="w-full whitespace-nowrap min-w-0">
                <TabsList className="flex w-full p-1 rounded-xl bg-muted/30">
                  <TabsTrigger value="ranking" className="flex-shrink-0 px-4 py-2 rounded-lg transition-all bg-transparent data-[state=active]:bg-transparent data-[state=active]:shadow-none data-[state=active]:scale-125 hover:bg-transparent whitespace-nowrap flex items-center justify-center">
                    <Trophy className="h-5 w-5 text-slate-700 dark:text-slate-200" />
                  </TabsTrigger>
                  {showIdentifiedItems && (
                    <TabsTrigger value="identified-items" className="flex-shrink-0 px-4 py-2 rounded-lg transition-all bg-transparent data-[state=active]:bg-transparent data-[state=active]:shadow-none data-[state=active]:scale-125 hover:bg-transparent whitespace-nowrap flex items-center justify-center font-semibold text-primary">
                      Items ({identifiedItems?.length})
                    </TabsTrigger>
                  )}
                  {vendorNames.map((vendorName) => (
                    <TabsTrigger
                      key={vendorName}
                      value={vendorName}
                      className="flex-shrink-0 px-4 py-2 rounded-lg transition-all bg-transparent data-[state=active]:bg-transparent data-[state=active]:shadow-none data-[state=active]:scale-125 hover:bg-transparent whitespace-nowrap flex items-center gap-1 justify-center"
                      title={vendorName}
                      aria-label={vendorName}
                      style={{ minWidth: 30, minHeight: 30 }}
                    >
                      <RenderVendorLogo vendorName={vendorName} size={60} />
                      <span
                        style={{
                          position: "absolute",
                          width: 1,
                          height: 1,
                          padding: 0,
                          overflow: "hidden",
                          clip: "rect(0, 0, 0, 0)",
                          whiteSpace: "nowrap",
                          border: 0,
                        }}
                        aria-hidden="true"
                      >
                        {vendorName}
                      </span>
                    </TabsTrigger>
                  ))}
                </TabsList>
                <ScrollBar orientation="horizontal" className="h-1.5" />
              </ScrollArea>

              {/* Identified Items Tab */}
              {showIdentifiedItems && (
                <TabsContent value="identified-items" className="mt-4 min-w-0">
                  {renderIdentifiedItemsList()}
                </TabsContent>
              )}

              {/* Best Match Tab */}
              <TabsContent value="ranking" className="mt-4 min-w-0">
                <Card className="bg-gradient-card shadow-card rounded-lg min-w-0 flex flex-col">
                  <CardHeader className="pb-3 min-w-0">
                    <CardTitle className="text-sm font-semibold flex items-center min-w-0 flex-wrap"></CardTitle>
                  </CardHeader>
                  <CardContent className="space-y-3 min-w-0">
                    {overallRanking.rankedProducts.map((product, index) => {
                      const key = `${product.vendor}-${product.productName}`;
                      const imageData = analysisImages[key];
                      const isLoadingImage = false;
                      const rawProductImgUrl = imageData?.topImage?.url || (product as any).topImage?.url || (product as any).top_image?.url || product.imageUrl;
                      const productImgUrl = getAbsoluteImageUrl(rawProductImgUrl);
                      const priceReviews = getPriceReview(product.vendor, product.productName);
                      const fbKey = getProductKey(product.vendor, product.productName);
                      const fb = feedbackState[fbKey] ?? { type: null, comment: "", loading: false, submitted: false };

                      const overallScore = product.overallScore ?? 0;

                      return (
                        <div
                          key={`${product.vendor}-${product.productName}-${index}`}
                          className={`bg-white/60 dark:bg-slate-900/60 backdrop-blur-md rounded-2xl p-6 shadow-xl border-2 ${getBorderColor(overallScore)} w-full max-w-full overflow-hidden break-words hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-1`}
                          style={{ position: "relative" }}
                        >
                          <div className="flex items-center justify-between flex-wrap min-w-0 gap-2">
                            <div className="flex flex-col flex-1 min-w-0">
                              <div className="flex items-center gap-2">
                                {/* Product image before product name */}
                                {isLoadingImage ? (
                                  <div className="w-10 h-10 bg-gray-200 dark:bg-gray-700 rounded animate-pulse flex items-center justify-center flex-shrink-0">
                                    <span className="text-xs text-gray-500">...</span>
                                  </div>
                                ) : productImgUrl ? (
                                  <div className="relative flex-shrink-0">
                                    <img
                                      src={productImgUrl}
                                      alt={`${product.productName} thumbnail`}
                                      onMouseEnter={() => setHoveredImage(productImgUrl)}
                                      onMouseLeave={() => setHoveredImage(null)}
                                      className="w-10 h-10 rounded object-contain cursor-pointer hover:scale-110 transition-transform"
                                    />
                                    {imageData?.topImage?.source && (
                                      <div className="absolute -top-1 -right-1 text-xs">
                                        {getQualityBadge(imageData.topImage.source).charAt(0)}
                                      </div>
                                    )}
                                  </div>
                                ) : null}
                                <h2 className="font-bold truncate select-text text-xl text-slate-900 dark:text-slate-100">{product.productName}</h2>
                              </div>
                              <div className="flex items-center gap-2 mt-1">
                                <p className="text-base font-medium text-slate-600 dark:text-slate-400 truncate select-text">{product.vendor}</p>
                                {product.modelFamily && (
                                  <span className="text-xs px-2 py-0.5 bg-slate-100 dark:bg-slate-700 rounded-full text-slate-500 dark:text-slate-400">
                                    {product.modelFamily} Series
                                  </span>
                                )}
                              </div>
                            </div>
                            <div className="w-12 h-12 flex items-center justify-center relative">
                              <CircularProgressBarSVG score={overallScore} />
                            </div>
                          </div>

                          <div className="space-y-6 mt-3">
                            <div className="p-4 rounded-xl bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 border border-green-200 dark:border-green-700/50 shadow-sm transition-transform duration-300 hover:scale-[1.01]">
                              <div className="flex items-center gap-2 mb-3">
                                <span className="text-lg">✨</span>
                                <p className="text-lg font-bold text-green-800 dark:text-green-300 select-text">Highlights</p>
                              </div>
                              {renderMarkdownContent(product.keyStrengths)}
                            </div>

                            {product.concerns && (
                              <div className="p-4 rounded-xl bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 border border-amber-200 dark:border-amber-700/50 shadow-sm transition-transform duration-300 hover:scale-[1.01]">
                                <div className="flex items-center gap-2 mb-3">
                                  <span className="text-lg">⚠️</span>
                                  <p className="text-lg font-bold text-amber-800 dark:text-amber-300 select-text">Limitations</p>
                                </div>
                                {renderMarkdownContent(product.concerns)}
                              </div>
                            )}

                            {priceReviews.results.length > 0 && (
                              <div className="p-4 rounded-xl bg-gradient-to-r from-[#F5FAFC] to-[#EAF6FB] dark:from-sky-900/20 dark:to-sky-800/20 border border-[#5FB3E6]/30 dark:border-sky-700/50 shadow-sm transition-transform duration-300 hover:scale-[1.01]">
                                <div className="flex items-center gap-2 mb-3">
                                  <span className="text-lg">💰</span>
                                  <p className="text-lg font-bold text-[#0F6CBD] dark:text-sky-300 select-text">Pricing Information</p>
                                </div>
                                <div className="space-y-3">
                                  {priceReviews.results
                                    .filter((result) => result.price)
                                    .map((result, idx) => (
                                      <div key={idx} className="flex items-center justify-between p-3 bg-white dark:bg-slate-800 rounded-lg border border-[#5FB3E6]/20 dark:border-sky-800/50 transition-transform duration-300 hover:scale-[1.02]">
                                        <div className="flex flex-col">
                                          <span className="text-xl font-bold text-[#0F6CBD] dark:text-sky-400">{result.price}</span>
                                          {result.source && <span className="text-sm text-slate-600 dark:text-slate-400">Available on {result.source}</span>}
                                        </div>
                                        {result.link && (
                                          <a href={result.link} target="_blank" rel="noopener noreferrer" className="p-2 text-[#0F6CBD] hover:text-[#084275] dark:text-sky-400 dark:hover:text-sky-200 transition-all duration-300 hover:scale-110 flex items-center justify-center">
                                            <Eye className="h-5 w-5" />
                                          </a>
                                        )}
                                      </div>
                                    ))}
                                </div>
                              </div>
                            )}

                            {/* Standards Compliance Section */}
                            {(() => {
                              const compliance = (product as any).standardsCompliance || (product as any).standards_compliance;
                              const compliantStandards = compliance?.compliant_standards || compliance?.compliantStandards || [];
                              if (compliantStandards.length > 0) {
                                return (
                                  <div className="p-4 rounded-xl bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 border border-indigo-200 dark:border-indigo-700/50 shadow-sm transition-transform duration-300 hover:scale-[1.01]">
                                    <div className="flex items-center gap-2 mb-3">
                                      <span className="text-lg">🏅</span>
                                      <p className="text-lg font-bold text-indigo-800 dark:text-indigo-300 select-text">Standards Compliance</p>
                                    </div>
                                    <div className="flex flex-wrap gap-2">
                                      {compliantStandards.map((standard: string, idx: number) => (
                                        <span key={idx} className="px-3 py-1 bg-indigo-100 dark:bg-indigo-800/40 text-indigo-700 dark:text-indigo-200 rounded-full text-sm font-medium">
                                          {standard}
                                        </span>
                                      ))}
                                    </div>
                                    {compliance?.compliance_notes && (
                                      <p className="mt-2 text-sm text-indigo-600 dark:text-indigo-400">{compliance.compliance_notes}</p>
                                    )}
                                  </div>
                                );
                              }
                              return null;
                            })()}

                            {/* Recommendation Section */}
                            {(product as any).recommendation && (
                              <div className="p-4 rounded-xl bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 border border-blue-200 dark:border-blue-700/50 shadow-sm transition-transform duration-300 hover:scale-[1.01]">
                                <div className="flex items-center gap-2 mb-3">
                                  <span className="text-lg">💡</span>
                                  <p className="text-lg font-bold text-blue-800 dark:text-blue-300 select-text">Recommendation</p>
                                </div>
                                <p className="text-base text-blue-700 dark:text-blue-200 leading-relaxed">{(product as any).recommendation}</p>
                              </div>
                            )}

                            {/* Technical Specifications Section */}
                            {(() => {
                              const matchedReqs = (product as any).matchedRequirements || (product as any).matched_requirements;
                              if (matchedReqs && Object.keys(matchedReqs).length > 0) {
                                return (
                                  <div className="p-4 rounded-xl bg-gradient-to-r from-slate-50 to-gray-50 dark:from-slate-800/50 dark:to-gray-800/50 border border-slate-200 dark:border-slate-700/50 shadow-sm transition-transform duration-300 hover:scale-[1.01]">
                                    <div className="flex items-center gap-2 mb-3">
                                      <span className="text-lg">📋</span>
                                      <p className="text-lg font-bold text-slate-800 dark:text-slate-200 select-text">Technical Specifications</p>
                                    </div>
                                    <div className="space-y-2">
                                      {Object.entries(matchedReqs).slice(0, 8).map(([key, req]: [string, any]) => (
                                        <div key={key} className="flex items-start justify-between p-2 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700/50">
                                          <div className="flex-1 min-w-0">
                                            <div className="flex items-center gap-2">
                                              <span className={`w-2 h-2 rounded-full flex-shrink-0 ${req.status === "MATCHED" ? "bg-green-500" :
                                                req.status === "PARTIAL" ? "bg-yellow-500" : "bg-red-500"
                                                }`}></span>
                                              <span className="text-sm font-medium text-slate-700 dark:text-slate-300 truncate">
                                                {req.parameter_name || key}
                                              </span>
                                              {req.is_critical && (
                                                <span className="text-xs px-1.5 py-0.5 bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 rounded">Critical</span>
                                              )}
                                            </div>
                                            <div className="ml-4 mt-1 text-xs text-slate-500 dark:text-slate-400">
                                              <span>Required: {req.user_requirement}</span>
                                              <span className="mx-2">→</span>
                                              <span className="text-slate-700 dark:text-slate-300 font-medium">{req.product_specification}</span>
                                            </div>
                                          </div>
                                        </div>
                                      ))}
                                      {Object.keys(matchedReqs).length > 8 && (
                                        <p className="text-xs text-slate-500 dark:text-slate-400 text-center pt-2">
                                          + {Object.keys(matchedReqs).length - 8} more specifications
                                        </p>
                                      )}
                                    </div>
                                  </div>
                                );
                              }
                              return null;
                            })()}

                            {/* Feedback Form for this product */}
                            <div className="mt-4 p-4 rounded-xl bg-gradient-to-r from-slate-50 to-gray-50 dark:from-slate-800 dark:to-slate-900 border border-slate-200 dark:border-slate-700 shadow-sm transition-transform duration-300 hover:scale-[1.01]">
                              <div className="flex items-center justify-between mb-2">
                                <div className="flex items-center gap-2">
                                  <span className="text-lg">💬</span>
                                  <p className="text-base font-semibold text-slate-800 dark:text-slate-200">Feedback</p>
                                </div>
                                <div className="flex gap-2">
                                  <Button variant="ghost" size="sm" onClick={() => setFeedbackType(fbKey, "positive")} className="h-10 w-10 p-0 rounded-full hover:bg-transparent bg-transparent border-0">
                                    <img src="/icon-thumbs-3d.png" alt="Thumbs Up" className={`h-8 w-8 object-contain transition-transform duration-300 hover:scale-125 ${fb.type === "positive" ? "scale-125 drop-shadow-md brightness-75" : "opacity-100"}`} />
                                  </Button>
                                  <Button variant="ghost" size="sm" onClick={() => setFeedbackType(fbKey, "negative")} className="h-10 w-10 p-0 rounded-full hover:bg-transparent bg-transparent border-0">
                                    <img src="/icon-thumbsdown-3d.png" alt="Thumbs Down" className={`h-8 w-8 object-contain transition-transform duration-300 hover:scale-125 ${fb.type === "negative" ? "scale-125 drop-shadow-md brightness-75" : "opacity-100"}`} />
                                  </Button>
                                </div>
                              </div>
                              {fb.submitted ? (
                                <div className="flex items-center gap-2 text-base text-green-700 dark:text-green-400 bg-green-50 dark:bg-green-900/20 p-3 rounded-lg animate-in fade-in duration-300">
                                  <span>{fb.response ? ` ${fb.response}` : "!"}</span>
                                </div>
                              ) : (
                                fb.type && (
                                  <div className="space-y-3 animate-in fade-in slide-in-from-top-1 duration-300 pt-1">
                                    <div className="relative group/textarea transition-transform duration-300 hover:scale-[1.01] origin-center">
                                      <Button
                                        variant="ghost"
                                        size="icon"
                                        onClick={() => setFeedbackType(fbKey, null)}
                                        className="absolute top-1 right-2 h-7 w-7 rounded-full bg-transparent hover:bg-transparent z-20 transition-transform duration-200 hover:scale-125 group"
                                      >
                                        <span className="text-sm font-bold leading-none text-slate-400 dark:text-slate-500 group-hover:text-slate-600 dark:group-hover:text-slate-300">✕</span>
                                      </Button>
                                      <Textarea
                                        value={fb.comment}
                                        onChange={(e) => setFeedbackComment(fbKey, e.target.value)}
                                        onKeyDown={(e) => handleFeedbackKeyDown(e, fbKey, product.vendor, product.productName)}
                                        className="bg-white dark:bg-slate-800 border-slate-300 dark:border-slate-600 text-base min-h-[80px] pb-3 pr-12 transition-all duration-300 focus:ring-2 focus:ring-[#5FB3E6] resize-none hover:shadow-md"
                                        placeholder="Tell us more... (Optional)"
                                        autoFocus
                                      />
                                      <div className="absolute bottom-2 right-2 z-10 transition-transform duration-200">
                                        <Button
                                          size="icon"
                                          variant="ghost"
                                          onClick={() => submitFeedback(fbKey, product.vendor, product.productName)}
                                          disabled={fb.loading}
                                          className={`h-9 w-9 rounded-full bg-transparent hover:bg-transparent transition-all duration-300 hover:scale-125 ${fb.comment.trim() ? "text-[#5FB3E6]" : "text-slate-300 dark:text-slate-600"}`}
                                        >
                                          {fb.loading ? (
                                            <Loader2 className="h-5 w-5 animate-spin" />
                                          ) : (
                                            <Send className="h-5 w-5 ml-0.5" />
                                          )}
                                        </Button>
                                      </div>
                                    </div>
                                  </div>
                                )
                              )}
                            </div>
                          </div>

                          {index < overallRanking.rankedProducts.length - 1 && <Separator className="my-3" />}
                        </div>
                      );
                    })}
                  </CardContent>
                </Card>
              </TabsContent>

              {/* Vendor Specific Tabs */}
              {vendorNames.map((vendorName) => (
                <TabsContent key={vendorName} value={vendorName} className="mt-4 min-w-0">
                  <Card className="bg-gradient-card shadow-card rounded-lg min-w-0 flex flex-col">
                    <CardHeader className="pb-3 min-w-0">

                    </CardHeader>
                    <CardContent className="space-y-4 min-w-0">
                      {vendorsGrouped[vendorName]
                        .sort((a, b) => (b.matchScore ?? 0) - (a.matchScore ?? 0))
                        .map((productMatch, idx) => {
                          // Get product-specific image from analysis images
                          const key = `${productMatch.vendor}-${productMatch.productName}`;
                          const imageData = analysisImages[key];
                          const rawProductImgUrl = imageData?.topImage?.url || (productMatch as any).topImage?.url || (productMatch as any).top_image?.url || productMatch.imageUrl;
                          const productImgUrl = cleanImageUrl(rawProductImgUrl);
                          const priceReviews = getPriceReview(productMatch.vendor, productMatch.productName);
                          const matchScore = productMatch.matchScore ?? 0;

                          return (
                            <div
                              key={`${productMatch.vendor}-${productMatch.productName}-${idx}`}
                              className={`bg-white/60 dark:bg-slate-900/60 backdrop-blur-md rounded-2xl p-6 shadow-xl border-2 ${getBorderColor(
                                matchScore
                              )} w-full max-w-full overflow-hidden break-words hover:shadow-2xl transition-all duration-300 transform hover:-translate-y-1`}
                            >
                              <div className="flex items-center justify-between flex-wrap min-w-0 gap-2">
                                {/* Product image (not vendor logo) */}
                                {productImgUrl ? (
                                  <img
                                    src={productImgUrl ? (productImgUrl.startsWith("/") ? `${BASE_URL}${productImgUrl}` : productImgUrl) : undefined}
                                    alt={`${productMatch.productName} thumbnail`}
                                    onMouseEnter={() => setHoveredImage(productImgUrl)}
                                    onMouseLeave={() => setHoveredImage(null)}
                                    style={{
                                      width: 30,
                                      height: 30,
                                      borderRadius: 6,
                                      objectFit: "contain",
                                      cursor: "pointer",
                                      flexShrink: 0,
                                    }}
                                  />
                                ) : (
                                  <span className="flex-shrink-0 text-lg select-none">{productMatch.productName?.charAt(0) ?? "•"}</span>
                                )}
                                <div className="flex flex-col flex-1 min-w-0">
                                  <h2 className="font-bold truncate select-text text-xl text-slate-900 dark:text-slate-100 mb-1">{productMatch.productName}</h2>
                                  <div className="flex items-center gap-2">
                                    {(productMatch as any).modelFamily && (
                                      <span className="text-xs px-2 py-0.5 bg-slate-100 dark:bg-slate-700 rounded-full text-slate-500 dark:text-slate-400">
                                        {(productMatch as any).modelFamily} Series
                                      </span>
                                    )}
                                  </div>
                                </div>
                                <div className="w-12 h-12 flex items-center justify-center relative">
                                  <CircularProgressBarSVG score={matchScore} />
                                </div>
                              </div>

                              <div className="space-y-3 mt-3">
                                <div className="p-4 rounded-xl bg-gradient-to-r from-green-50 to-emerald-50 dark:from-green-900/20 dark:to-emerald-900/20 border border-green-200 dark:border-green-700/50 shadow-sm transition-transform duration-300 hover:scale-[1.01] mb-6">
                                  <div className="flex items-center gap-2 mb-3">
                                    <span className="text-lg">✨</span>
                                    <p className="text-lg font-bold text-green-800 dark:text-green-300 select-text">Highlights</p>
                                  </div>
                                  {renderMarkdownContent(productMatch.reasoning)}
                                </div>

                                {productMatch.limitations && (
                                  <div className="p-4 rounded-xl bg-gradient-to-r from-amber-50 to-orange-50 dark:from-amber-900/20 dark:to-orange-900/20 border border-amber-200 dark:border-amber-700/50 shadow-sm transition-transform duration-300 hover:scale-[1.01] mb-6">
                                    <div className="flex items-center gap-2 mb-3">
                                      <span className="text-lg">⚠️</span>
                                      <p className="text-lg font-bold text-amber-800 dark:text-amber-300 select-text">Limitations</p>
                                    </div>
                                    {renderMarkdownContent(productMatch.limitations)}
                                  </div>
                                )}

                                {priceReviews.results.length > 0 && (
                                  <div className="p-4 rounded-xl bg-gradient-to-r from-[#F5FAFC] to-[#EAF6FB] dark:from-sky-900/20 dark:to-sky-800/20 border border-[#5FB3E6]/30 dark:border-sky-700/50 shadow-sm transition-transform duration-300 hover:scale-[1.01] mb-6">
                                    <div className="flex items-center gap-2 mb-3">
                                      <span className="text-lg">💰</span>
                                      <p className="text-lg font-bold text-[#0F6CBD] dark:text-sky-300 select-text">Pricing Information</p>
                                    </div>
                                    <div className="space-y-3">
                                      {priceReviews.results
                                        .filter((result) => result.price)
                                        .map((result, idx) => (
                                          <div key={idx} className="flex items-center justify-between p-3 bg-white dark:bg-slate-800 rounded-lg border border-[#5FB3E6]/20 dark:border-sky-800/50 transition-transform duration-300 hover:scale-[1.02]">
                                            <div className="flex flex-col">
                                              <span className="text-xl font-bold text-[#0F6CBD] dark:text-sky-400">{result.price}</span>
                                              {result.source && <span className="text-sm text-slate-600 dark:text-slate-400">Available on {result.source}</span>}
                                            </div>
                                            {result.link && (
                                              <a href={result.link} target="_blank" rel="noopener noreferrer" className="p-2 text-[#0F6CBD] hover:text-[#084275] dark:text-sky-400 dark:hover:text-sky-200 transition-all duration-300 hover:scale-110 flex items-center justify-center" title="View Deal">
                                                <Eye className="h-5 w-5" />
                                              </a>
                                            )}
                                          </div>
                                        ))}
                                    </div>
                                  </div>
                                )}

                                {/* Standards Compliance Section */}
                                {(() => {
                                  const compliance = (productMatch as any).standardsCompliance || (productMatch as any).standards_compliance;
                                  const compliantStandards = compliance?.compliant_standards || compliance?.compliantStandards || [];
                                  if (compliantStandards.length > 0) {
                                    return (
                                      <div className="p-4 rounded-xl bg-gradient-to-r from-indigo-50 to-purple-50 dark:from-indigo-900/20 dark:to-purple-900/20 border border-indigo-200 dark:border-indigo-700/50 shadow-sm transition-transform duration-300 hover:scale-[1.01] mb-6">
                                        <div className="flex items-center gap-2 mb-3">
                                          <span className="text-lg">🏅</span>
                                          <p className="text-lg font-bold text-indigo-800 dark:text-indigo-300 select-text">Standards Compliance</p>
                                        </div>
                                        <div className="flex flex-wrap gap-2">
                                          {compliantStandards.map((standard: string, idx: number) => (
                                            <span key={idx} className="px-3 py-1 bg-indigo-100 dark:bg-indigo-800/40 text-indigo-700 dark:text-indigo-200 rounded-full text-sm font-medium">
                                              {standard}
                                            </span>
                                          ))}
                                        </div>
                                        {compliance?.compliance_notes && (
                                          <p className="mt-2 text-sm text-indigo-600 dark:text-indigo-400">{compliance.compliance_notes}</p>
                                        )}
                                      </div>
                                    );
                                  }
                                  return null;
                                })()}

                                {/* Recommendation Section */}
                                {(productMatch as any).recommendation && (
                                  <div className="p-4 rounded-xl bg-gradient-to-r from-blue-50 to-cyan-50 dark:from-blue-900/20 dark:to-cyan-900/20 border border-blue-200 dark:border-blue-700/50 shadow-sm transition-transform duration-300 hover:scale-[1.01] mb-6">
                                    <div className="flex items-center gap-2 mb-3">
                                      <span className="text-lg">💡</span>
                                      <p className="text-lg font-bold text-blue-800 dark:text-blue-300 select-text">Recommendation</p>
                                    </div>
                                    <p className="text-base text-blue-700 dark:text-blue-200 leading-relaxed">{(productMatch as any).recommendation}</p>
                                  </div>
                                )}

                                {/* Technical Specifications Section */}
                                {(() => {
                                  const matchedReqs = (productMatch as any).matchedRequirements || (productMatch as any).matched_requirements;
                                  if (matchedReqs && Object.keys(matchedReqs).length > 0) {
                                    return (
                                      <div className="p-4 rounded-xl bg-gradient-to-r from-slate-50 to-gray-50 dark:from-slate-800/50 dark:to-gray-800/50 border border-slate-200 dark:border-slate-700/50 shadow-sm transition-transform duration-300 hover:scale-[1.01] mb-6">
                                        <div className="flex items-center gap-2 mb-3">
                                          <span className="text-lg">📋</span>
                                          <p className="text-lg font-bold text-slate-800 dark:text-slate-200 select-text">Technical Specifications</p>
                                        </div>
                                        <div className="space-y-2">
                                          {Object.entries(matchedReqs).slice(0, 8).map(([key, req]: [string, any]) => (
                                            <div key={key} className="flex items-start justify-between p-2 bg-white dark:bg-slate-800 rounded-lg border border-slate-200 dark:border-slate-700/50">
                                              <div className="flex-1 min-w-0">
                                                <div className="flex items-center gap-2">
                                                  <span className={`w-2 h-2 rounded-full flex-shrink-0 ${req.status === "MATCHED" ? "bg-green-500" :
                                                    req.status === "PARTIAL" ? "bg-yellow-500" : "bg-red-500"
                                                    }`}></span>
                                                  <span className="text-sm font-medium text-slate-700 dark:text-slate-300 truncate">
                                                    {req.parameter_name || key}
                                                  </span>
                                                  {req.is_critical && (
                                                    <span className="text-xs px-1.5 py-0.5 bg-red-100 dark:bg-red-900/30 text-red-600 dark:text-red-400 rounded">Critical</span>
                                                  )}
                                                </div>
                                                <div className="ml-4 mt-1 text-xs text-slate-500 dark:text-slate-400">
                                                  <span>Required: {req.user_requirement}</span>
                                                  <span className="mx-2">→</span>
                                                  <span className="text-slate-700 dark:text-slate-300 font-medium">{req.product_specification}</span>
                                                </div>
                                              </div>
                                            </div>
                                          ))}
                                          {Object.keys(matchedReqs).length > 8 && (
                                            <p className="text-xs text-slate-500 dark:text-slate-400 text-center pt-2">
                                              + {Object.keys(matchedReqs).length - 8} more specifications
                                            </p>
                                          )}
                                        </div>
                                      </div>
                                    );
                                  }
                                  return null;
                                })()}
                              </div>

                              {idx < vendorsGrouped[vendorName].length - 1 && <Separator className="my-3" />}
                            </div>
                          );
                        })}
                    </CardContent>
                  </Card>
                </TabsContent>
              ))}
            </Tabs>
          </div>
        </ScrollArea>
      )}

      {hoveredImage && (
        <div
          onMouseLeave={() => setHoveredImage(null)}
          style={{
            position: "fixed",
            top: "50%",
            left: "50%",
            transform: "translate(-50%, -50%)",
            background: "rgba(255, 255, 255, 0.98)",
            padding: 24,
            borderRadius: 16,
            boxShadow: "0 20px 60px rgba(0,0,0,0.4)",
            zIndex: 9999,
            maxWidth: "600px",
            maxHeight: "600px",
            display: "flex",
            justifyContent: "center",
            alignItems: "center",
            border: "2px solid rgba(0,0,0,0.1)",
          }}
          className="dark:bg-slate-800/98 dark:border-slate-600"
        >
          <img
            src={hoveredImage ? (hoveredImage.startsWith("/") ? `${BASE_URL}${hoveredImage}` : hoveredImage) : undefined}
            alt="Product preview"
            style={{
              maxWidth: "550px",
              maxHeight: "550px",
              objectFit: "contain",
              borderRadius: 12,
            }}
          />
        </div>
      )}

      {/* Image Gallery Modal */}
      {imageGalleryOpen && (
        <ImageGallery
          images={imageGalleryOpen.images}
          isOpen={true}
          onClose={() => setImageGalleryOpen(null)}
          productName={imageGalleryOpen.productName}
        />
      )}

      <style>{`
        .custom-no-scrollbar::-webkit-scrollbar {
          display: none;
        }
        .custom-no-scrollbar {
          -ms-overflow-style: none;
          scrollbar-width: none;
        }
      `}</style>
    </div>
  );
};

export default RightPanel;