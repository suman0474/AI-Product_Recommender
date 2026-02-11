import { Toaster } from "@/components/ui/toaster";
import { Toaster as Sonner } from "@/components/ui/sonner";
import { TooltipProvider } from "@/components/ui/tooltip";
import { QueryClient, QueryClientProvider } from "@tanstack/react-query";
import { BrowserRouter, Routes, Route, Navigate } from "react-router-dom";
import { AuthProvider, useAuth } from "./contexts/AuthContext";
import React, { Suspense, lazy } from "react";

// Lazy-loaded page components for code splitting
// Each page will be loaded only when navigated to, reducing initial bundle size
const Index = lazy(() => import("./pages/Index"));
const NotFound = lazy(() => import("./pages/NotFound"));
const Landing = lazy(() => import("./pages/Landing"));
const Login = lazy(() => import("./pages/Login"));
const Signup = lazy(() => import("./pages/Signup"));
const AdminDashboard = lazy(() => import("./pages/AdminDashboard"));
const Project = lazy(() => import("./pages/Project"));
const EnGenieChat = lazy(() => import("./pages/EnGenieChat"));
const Uploading = lazy(() => import("./pages/Uploading"));

// Loading fallback component for Suspense
const PageLoader = () => (
  <div className="flex items-center justify-center min-h-screen">
    <div className="animate-spin rounded-full h-8 w-8 border-b-2 border-primary"></div>
  </div>
);

const queryClient = new QueryClient();

// Enhanced ProtectedRoute to allow optional admin-only access
interface ProtectedRouteProps {
  children: React.ReactNode;
  requireAdmin?: boolean;
}

const ProtectedRoute = ({ children, requireAdmin = false }: ProtectedRouteProps) => {
  const { isAuthenticated, isLoading, user } = useAuth();

  if (isLoading) {
    // Render a loading state while auth info is loading
    return <div>Loading...</div>;
  }

  if (!isAuthenticated) {
    return <Navigate to="/login" />;
  }

  if (requireAdmin && user?.role !== "admin") {
    // Redirect non-admins away from admin routes (you can customize the path)
    return <Navigate to="/search" />;
  }

  return <>{children}</>;
};

const App = () => (
  <QueryClientProvider client={queryClient}>
    <TooltipProvider>
      <AuthProvider>
        <Toaster />
        <Sonner />
        <BrowserRouter
          future={{
            v7_startTransition: true,
            v7_relativeSplatPath: true,
          }}
        >
          <Suspense fallback={<PageLoader />}>
            <Routes>
              {/* Public routes */}
              <Route path="/" element={<Landing />} />
              <Route path="/login" element={<Login />} />
              <Route path="/signup" element={<Signup />} />

              {/* Solution routes */}
              <Route
                path="/solution"
                element={
                  <ProtectedRoute>
                    <Project />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/solution/search"
                element={
                  <ProtectedRoute>
                    <Project />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/solution/search/upload"
                element={
                  <ProtectedRoute requireAdmin>
                    <Uploading />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/solution/search/admin"
                element={
                  <ProtectedRoute requireAdmin>
                    <AdminDashboard />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/solution/upload"
                element={
                  <ProtectedRoute requireAdmin>
                    <Uploading />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/solution/admin"
                element={
                  <ProtectedRoute requireAdmin>
                    <AdminDashboard />
                  </ProtectedRoute>
                }
              />

              {/* Search routes */}
              <Route
                path="/search"
                element={
                  <ProtectedRoute>
                    <Index />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/search/upload"
                element={
                  <ProtectedRoute requireAdmin>
                    <Uploading />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/search/admin"
                element={
                  <ProtectedRoute requireAdmin>
                    <AdminDashboard />
                  </ProtectedRoute>
                }
              />

              {/* Chat routes */}
              <Route
                path="/chat"
                element={
                  <ProtectedRoute>
                    <EnGenieChat />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/chat/upload"
                element={
                  <ProtectedRoute requireAdmin>
                    <Uploading />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/chat/admin"
                element={
                  <ProtectedRoute requireAdmin>
                    <AdminDashboard />
                  </ProtectedRoute>
                }
              />

              {/* Legacy routes */}
              <Route
                path="/admin"
                element={
                  <ProtectedRoute requireAdmin>
                    <AdminDashboard />
                  </ProtectedRoute>
                }
              />
              <Route
                path="/upload"
                element={
                  <ProtectedRoute requireAdmin>
                    <Uploading />
                  </ProtectedRoute>
                }
              />

              {/* 404 */}
              <Route path="*" element={<NotFound />} />
            </Routes>
            <div className="hidden">
              {/* Hidden Routes for backward compatibility/aliases if needed in future */}
              {/* <Route path="/project" element={<Navigate to="/solution" replace />} /> */}
            </div>
          </Suspense>
        </BrowserRouter>
      </AuthProvider>
    </TooltipProvider>
  </QueryClientProvider>
);

export default App;
