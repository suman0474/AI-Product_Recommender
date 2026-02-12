import React, { useState, useRef } from "react";
import { useNavigate, useLocation } from "react-router-dom";
import { Button } from "@/components/ui/button";
import { Card, CardContent } from "@/components/ui/card";
import {
  Upload,
  FileText,
  CheckCircle,
  AlertCircle,
  ArrowLeft,
  Eye,
  CloudUpload,
} from "lucide-react";
import { cn } from "@/lib/utils";
import { uploadPdfFile, searchPdfs, uploadPdfFromUrl, viewPdf } from "../components/AIRecommender/api";
import MainHeader from "@/components/MainHeader";

interface PdfSearchResult {
  title: string;
  url: string;
  snippet: string;
}

const PdfSearchAndUpload = () => {
  const navigate = useNavigate();
  const location = useLocation();

  const [searchQuery, setSearchQuery] = useState("");
  const [pdfList, setPdfList] = useState<PdfSearchResult[]>([]);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [status, setStatus] = useState<"idle" | "success" | "error" | "processing">("idle");
  const [message, setMessage] = useState("");
  const [results, setResults] = useState<any>(null);

  const fileInputRef = useRef<HTMLInputElement>(null);

  // --- Search PDFs ---
  const handleSearch = async () => {
    if (!searchQuery.trim()) return;
    setIsProcessing(true);
    setStatus("processing");
    setMessage("Searching for PDFs...");
    try {
      const searchResults: PdfSearchResult[] = await searchPdfs(searchQuery);
      setPdfList(searchResults);
      setStatus("success");
      setMessage(`${searchResults.length} PDFs found.`);
    } catch (err: any) {
      console.error(err);
      setMessage(err.response?.data?.error || "Failed to search for PDFs.");
      setStatus("error");
      setPdfList([]);
    } finally {
      setIsProcessing(false);
    }
  };

  // --- Handle a URL-based upload from a search result ---
  const handleUrlUpload = async (url: string) => {
    setIsProcessing(true);
    setStatus("processing");
    setMessage("Processing your file...");
    setResults(null);

    try {
      const data = await uploadPdfFromUrl(url);
      setResults(data);
      setStatus("success");
      setMessage("Success! Your data has been extracted.");
    } catch (error: any) {
      setStatus("error");
      setMessage(error.message || "An unexpected error occurred during processing.");
    } finally {
      setIsProcessing(false);
    }
  };

  // --- Handle a local file upload ---
  const handleLocalFileUpload = async (file: File) => {
    setIsProcessing(true);
    setStatus("processing");
    setMessage("Processing your file...");
    setResults(null);

    try {
      const data = await uploadPdfFile(file);
      setResults(data);
      setStatus("success");
      setMessage("Success! Your data has been extracted.");
    } catch (error: any) {
      setStatus("error");
      setMessage(error.message || "An unexpected error occurred during upload.");
    } finally {
      setIsProcessing(false);
    }
  };

  const getStatusIcon = () => {
    switch (status) {
      case "success":
        return <CheckCircle className="w-5 h-5 text-green-600" />;
      case "error":
        return <AlertCircle className="w-5 h-5 text-red-600" />;
      case "processing":
        return <div className="w-5 h-5 border-2 border-[#0F6CBD] border-t-transparent rounded-full animate-spin" />;
      default:
        return null;
    }
  };

  return (
    <div className="min-h-screen app-glass-gradient flex flex-col">
      <MainHeader>
        <div className="flex items-center gap-2">
          <h1 className="text-xl font-bold bg-clip-text text-transparent bg-gradient-to-r from-[#0F6CBD] to-[#004E8C]">
            PDF Explorer & Extractor
          </h1>
        </div>
      </MainHeader>

      <div className="flex-1 flex items-center justify-center p-4 pt-24">
        <div className="w-full max-w-3xl transition-all duration-300 hover:scale-[1.02]">
          <div className="acrylic-glass-pill backdrop-blur-3xl p-8 relative transition-all duration-300 hover:shadow-2xl">
            {/* Header */}
            <div className="text-center mb-6 space-y-4">
              <button
                onClick={() => {
                  if (location.pathname.includes('/solution/search')) {
                    navigate('/solution/search');
                  } else if (location.pathname.includes('/solution')) {
                    navigate('/solution');
                  } else if (location.pathname.includes('/search')) {
                    navigate('/search');
                  } else if (location.pathname.includes('/chat')) {
                    navigate('/chat');
                  } else {
                    navigate('/solution');
                  }
                }}
                className="absolute top-4 left-8 text-slate-700 hover:text-slate-900 dark:text-slate-400 dark:hover:text-slate-100 transition-transform hover:scale-125 bg-transparent p-0 border-none outline-none group"
                title="Return to Requirements"
              >
                <ArrowLeft className="w-5 h-5" />
              </button>
              <p className="text-muted-foreground">
                Search vendor/product PDFs, view them, or extract data.
              </p>
            </div>

            {/* Search Bar */}
            <div className="flex mb-6 gap-2">
              <input
                type="text"
                className="flex-1 form-glass-input rounded-xl border border-white/20 dark:border-slate-700/30 p-3 bg-white/50 dark:bg-slate-800/50 backdrop-blur-sm focus:outline-none focus:ring-2 focus:ring-[#5FB3E6] placeholder:text-muted-foreground transition-all duration-300 hover:scale-[1.01] hover:shadow-md"
                placeholder="Search vendor/product..."
                value={searchQuery}
                onChange={(e) => setSearchQuery(e.target.value)}
                onKeyDown={(e) => { if (e.key === "Enter") handleSearch(); }}
              />
              <Button onClick={handleSearch} className="px-6 btn-glass-primary text-white rounded-xl" disabled={isProcessing}>
                {isProcessing ? "Searching..." : "Search"}
              </Button>
            </div>

            {/* PDF List */}
            {pdfList.length > 0 && (
              <div className="grid grid-cols-1 sm:grid-cols-2 gap-4 mb-6">
                {pdfList.map((pdf) => (
                  <Card key={pdf.url} className="bg-gradient-to-br from-[#F5FAFC]/50 to-[#EAF6FB]/50 dark:from-slate-800/60 dark:to-slate-800/30 backdrop-blur-sm border border-white/30 dark:border-slate-700/30 rounded-xl hover:shadow-lg transition-all duration-300 hover:bg-white/60 dark:hover:bg-slate-800/60 hover:scale-[1.02]">
                    <CardContent className="p-4 flex flex-col justify-between h-full">
                      <div>
                        <p className="font-medium text-gray-900 mb-2">{pdf.title}</p>
                        <p className="text-sm text-gray-500 truncate">{pdf.url}</p>
                        <p className="text-sm text-gray-400 mt-1">{pdf.snippet}</p>
                      </div>
                      <div className="flex gap-2 mt-4">
                        <Button
                          variant="outline"
                          onClick={() => window.open(viewPdf(pdf.url), "_blank")}
                          className="flex-1"
                          disabled={isProcessing}
                        >
                          <Eye className="w-4 h-4 mr-2" /> View
                        </Button>
                        <Button
                          onClick={() => handleUrlUpload(pdf.url)}
                          className="flex-1 bg-[#5FB3E6] text-white hover:bg-[#4DA0D1]"
                          disabled={isProcessing}
                        >
                          <CloudUpload className="w-4 h-4 mr-2" />
                          {isProcessing ? "Processing..." : "Upload"}
                        </Button>
                      </div>
                    </CardContent>
                  </Card>
                ))}
              </div>
            )}

            {/* Local PDF Upload */}
            <Card className="bg-gradient-to-br from-[#F5FAFC]/50 to-[#EAF6FB]/50 dark:from-slate-800/60 dark:to-slate-800/30 backdrop-blur-sm border border-white/30 dark:border-slate-700/30 rounded-xl shadow-sm transition-all duration-300 hover:shadow-lg hover:scale-[1.02]">
              <CardContent className="p-6 text-center">
                <div
                  className={cn(
                    "border-2 border-dashed border-[#5FB3E6]/50 rounded-2xl p-8 cursor-pointer flex flex-col items-center justify-center mb-4 transition-colors hover:border-[#0F6CBD] hover:bg-white/40 dark:hover:bg-slate-800/40 group transition-all duration-300 hover:scale-[1.02] hover:shadow-xl",
                    selectedFile ? "bg-white/40 border-[#5FB3E6]" : ""
                  )}
                  onClick={() => fileInputRef.current?.click()}
                >
                  <img src="/icon-upload-3d.png" alt="Upload" className="w-24 h-24 mb-4 object-contain drop-shadow-md transition-transform duration-300 group-hover:scale-110" />
                  {selectedFile ? (
                    <div className="flex items-center justify-center space-x-3 mt-2">
                      <FileText className="w-8 h-8 text-[#0F6CBD]" />
                      <div className="text-left">
                        <p className="font-medium text-foreground">{selectedFile.name}</p>
                        <p className="text-sm text-muted-foreground">{(selectedFile.size / 1024 / 1024).toFixed(2)} MB</p>
                      </div>
                    </div>
                  ) : (
                    <>
                      <p className="text-xl font-extrabold text-foreground mb-1">Choose a PDF file</p>
                      <p className="text-muted-foreground">Click here to select your document</p>
                    </>
                  )}
                  <input
                    ref={fileInputRef}
                    type="file"
                    accept=".pdf"
                    onChange={(e) => {
                      const file = e.target.files?.[0];
                      if (file) setSelectedFile(file);
                    }}
                    className="hidden"
                  />
                </div>
                <Button
                  onClick={() => selectedFile && handleLocalFileUpload(selectedFile)}
                  disabled={!selectedFile || isProcessing}
                  className="w-full sm:w-auto px-8 rounded-full btn-glass-primary text-white disabled:opacity-50 disabled:cursor-not-allowed"
                >
                  {isProcessing ? "Processing..." : "Upload and Process"}
                </Button>
              </CardContent>
            </Card>

            {/* Status */}
            {status !== "idle" && (
              <div
                className={cn(
                  "p-4 rounded-lg flex items-center space-x-3 mt-4",
                  status === "success" && "bg-green-100 text-green-700",
                  status === "error" && "bg-red-100 text-red-700",
                  status === "processing" && "bg-sky-100 text-sky-700"
                )}
              >
                {getStatusIcon()}
                <p className="font-medium">{message}</p>
              </div>
            )}

            {/* Extraction Results */}
            {results && (
              <div className="mt-6 text-left">
                <h3 className="text-xl font-bold text-gray-900 mb-4">Extraction Results</h3>
                <div className="bg-gray-100 rounded-lg p-4 overflow-x-auto">
                  <pre className="text-sm text-gray-600 whitespace-pre-wrap">
                    {JSON.stringify(results, null, 2)}
                  </pre>
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
};

export default PdfSearchAndUpload;