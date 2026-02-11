import React, { useState, useEffect } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { useAuth } from '../contexts/AuthContext';
import { Button } from '../components/ui/button';
import { Brain, CheckCircle, XCircle, ArrowLeft } from 'lucide-react';
import { GlassCard, GlassCardContent, GlassCardHeader, GlassCardTitle } from '../components/ui/glass-card';
import { BASE_URL } from '../components/AIRecommender/api';
const API_BASE_URL = BASE_URL;

const AdminDashboard = () => {
  const { user, isLoading: authLoading, isAuthenticated } = useAuth();
  const navigate = useNavigate();
  const location = useLocation();
  const [pendingUsers, setPendingUsers] = useState([]);
  const [dataLoading, setDataLoading] = useState(true);
  const [approvingIds, setApprovingIds] = useState(new Set());
  const [error, setError] = useState(null);

  // State for modal
  const [modalOpen, setModalOpen] = useState(false);
  const [modalAction, setModalAction] = useState(null); // 'approve' or 'reject'
  const [modalUser, setModalUser] = useState(null); // user object for which modal is open

  useEffect(() => {
    if (!authLoading && (!isAuthenticated || user?.role !== 'admin')) {
      navigate('/');
    }
  }, [authLoading, isAuthenticated, user, navigate]);

  const fetchPendingUsers = async () => {
    setDataLoading(true);
    setError(null);
    try {
      const response = await fetch(`${API_BASE_URL}/admin/pending_users`, {
        method: 'GET',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
      });

      if (!response.ok) {
        const errorText = await response.text();
        console.error('Failed API response:', errorText);
        throw new Error(`HTTP error! Status: ${response.status}. Check console for details.`);
      }

      const data = await response.json();
      if (!data.pending_users || !Array.isArray(data.pending_users)) {
        throw new Error("API response is not in the expected format. Expected an object with a 'pending_users' array.");
      }

      setPendingUsers(data.pending_users);
    } catch (error) {
      console.error('Failed to fetch pending users:', error);
      setError(error.message);
      setPendingUsers([]);
    } finally {
      setDataLoading(false);
    }
  };

  useEffect(() => {
    if (user?.role === 'admin') {
      fetchPendingUsers();
    }
  }, [user]);

  const performAction = async (userId, action) => {
    setApprovingIds(prev => new Set(prev).add(userId));
    setModalOpen(false);
    try {
      const response = await fetch(`${API_BASE_URL}/admin/approve_user`, {
        method: 'POST',
        credentials: 'include',
        headers: {
          'Content-Type': 'application/json',
        },
        body: JSON.stringify({ user_id: userId, action }),
      });

      if (!response.ok) {
        throw new Error(`HTTP error! status: ${response.status}`);
      }

      setPendingUsers(prev => prev.filter(u => u.id !== userId));
    } catch (error) {
      console.error(`Failed to ${action} user:`, error);
    } finally {
      setApprovingIds(prev => {
        const newSet = new Set(prev);
        newSet.delete(userId);
        return newSet;
      });
    }
  };

  const openConfirmationModal = (user, action) => {
    setModalUser(user);
    setModalAction(action);
    setModalOpen(true);
  };

  if (authLoading || dataLoading) {
    return <div className="min-h-screen app-glass-gradient flex items-center justify-center text-foreground">Loading...</div>;
  }

  if (!user || user.role !== 'admin') {
    return null;
  }

  return (
    <div className="min-h-screen app-glass-gradient text-foreground flex flex-col items-center justify-center p-6 relative overflow-hidden">
      {/* Subtle gradient background */}
      <div className="absolute inset-0 bg-gradient-to-br from-ai-primary/10 via-ai-primary/5 to-background"></div>

      <div className="relative w-full max-w-4xl transition-all duration-300 hover:scale-[1.02]">
        <div className="acrylic-glass-pill backdrop-blur-3xl p-8 relative transition-all duration-300 hover:shadow-2xl">
          <div className="text-center space-y-4">
            <div className="w-20 h-20 mx-auto">
              <img src="/ChatGPT Image Nov 11, 2025, 11_58_30 AM.png" alt="Logo" className="w-full h-full rounded-full object-cover shadow-lg" />
            </div>
            <div>
              <h1 className="text-3xl font-bold text-transparent bg-clip-text bg-gradient-to-r from-[#0F6CBD] to-[#004E8C]">
                Admin Dashboard
              </h1>
            </div>
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
              title="Return"
            >
              <ArrowLeft className="w-5 h-5" />
            </button>

            <p className="text-muted-foreground mt-2 text-base">
              Manage user accounts pending approval.
            </p>
          </div>

          <div className="mt-8">
            <h3 className="text-xl font-semibold text-foreground mb-4">Pending Users</h3>
            {error ? (
              <p className="text-ai-error text-center">{error}</p>
            ) : pendingUsers.length === 0 ? (
              <p className="text-muted-foreground text-center">No new users are pending approval.</p>
            ) : (
              <div className="space-y-4">
                {pendingUsers.map((pendingUser) => (
                  <div
                    key={pendingUser.id}
                    className="flex items-center justify-between p-4 bg-gradient-to-br from-[#F5FAFC]/50 to-[#EAF6FB]/50 dark:from-slate-800/60 dark:to-slate-800/30 backdrop-blur-sm border border-white/30 dark:border-slate-700/30 rounded-xl hover:shadow-lg transition-all duration-300 hover:bg-white/20 dark:hover:bg-slate-800/40 hover:scale-[1.02]"
                  >
                    <div>
                      <p className="text-foreground font-medium">{pendingUser.username}</p>
                      <p className="text-muted-foreground text-sm">{pendingUser.email}</p>
                    </div>
                    <div className="flex gap-2">
                      <Button
                        onClick={() => openConfirmationModal(pendingUser, 'approve')}
                        disabled={approvingIds.has(pendingUser.id)}
                        className="bg-ai-success hover:bg-ai-success text-black transition-transform hover:scale-105"
                      >
                        <CheckCircle className="mr-2 h-4 w-4" />
                        Approve
                      </Button>
                      <Button
                        variant="ghost"
                        onClick={() => openConfirmationModal(pendingUser, 'reject')}
                        disabled={approvingIds.has(pendingUser.id)}
                        className="text-ai-error hover:text-red-700 hover:bg-transparent transition-transform hover:scale-105"
                      >
                        <XCircle className="mr-2 h-4 w-4" />
                        Reject
                      </Button>
                    </div>
                  </div>
                ))}
              </div>
            )}
          </div>
        </div>
      </div>

      {/* Confirmation Modal */}
      {modalOpen && (
        <div className="fixed inset-0 flex items-center justify-center z-50 bg-black/70 backdrop-blur-md">
          <div className="rounded-2xl p-8 w-[380px] max-w-full text-center shadow-2xl border border-white/20 dark:border-slate-700/30 bg-gradient-to-br from-[#F5FAFC]/90 to-[#EAF6FB]/90 dark:from-slate-900/90 dark:to-slate-900/50 backdrop-blur-2xl text-foreground">
            <h2 className="text-2xl font-bold mb-4">
              Confirm {modalAction === 'approve' ? 'Approval' : 'Rejection'}
            </h2>
            <p className="mb-6 text-lg">
              Are you sure you want to {modalAction} <span className="font-bold">{modalUser?.username}</span>?
            </p>
            <div className="flex justify-center items-center gap-8 mt-6">

              <button
                onClick={() => setModalOpen(false)}
                className="text-muted-foreground hover:text-slate-900 dark:hover:text-slate-100 font-medium transition-transform hover:scale-125 bg-transparent border-none cursor-pointer"
              >
                Cancel
              </button>
              <button
                className={`font-semibold flex items-center gap-2 transition-transform hover:scale-125 bg-transparent border-none cursor-pointer ${modalAction === 'approve'
                  ? 'text-ai-success hover:text-green-700 dark:hover:text-green-400'
                  : 'text-ai-error hover:text-red-700 dark:hover:text-red-400'
                  }`}
                onClick={() => performAction(modalUser.id, modalAction)}
                disabled={approvingIds.has(modalUser.id)}
              >
                {modalAction === 'approve' ? (
                  <>
                    <CheckCircle className="inline mr-1 h-5 w-5" /> Approve
                  </>
                ) : (
                  <>
                    <XCircle className="inline mr-1 h-5 w-5" /> Reject
                  </>
                )}
              </button>
            </div>
          </div>
        </div>
      )}



    </div>
  );
};

export default AdminDashboard;
