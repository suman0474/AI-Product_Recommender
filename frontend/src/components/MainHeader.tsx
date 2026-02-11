import { useState } from 'react';
import { useNavigate, useLocation } from 'react-router-dom';
import { Button } from '@/components/ui/button';
import { LogOut, User, Upload, Bot, Save, FileText, FolderOpen } from 'lucide-react';
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
    TooltipTrigger,
} from "@/components/ui/tooltip";
import { useAuth } from '@/contexts/AuthContext';
import { ProfileEditDialog } from '@/components/ProfileEditDialog';
import ProjectListDialog from '@/components/ProjectListDialog';

interface MainHeaderProps {
    /** Optional: Additional content to render between the logo and profile section */
    children?: React.ReactNode;
    /** Optional: Show project action buttons (Save, New, Open) - default true for all pages */
    showProjectActions?: boolean;
    /** Optional: Project action button handlers */
    onSave?: () => void;
    onNew?: () => void;
    onProjectSelect?: (project: any) => void;
    onProjectDelete?: (projectId: string) => void;
    /** Optional: Custom class name for the header */
    className?: string;
}

const MainHeader = ({
    children,
    showProjectActions = true,
    onSave,
    onNew,
    onProjectSelect,
    onProjectDelete,
    className = ""
}: MainHeaderProps) => {
    const navigate = useNavigate();
    const { user, logout } = useAuth();
    const [isProfileEditOpen, setIsProfileEditOpen] = useState(false);
    const [isProjectListOpen, setIsProjectListOpen] = useState(false);

    // Compute profile display values
    const profileFullName = user?.firstName && user?.lastName
        ? `${user.firstName} ${user.lastName}`
        : user?.email || "User";
    const profileButtonLabel = user?.firstName || user?.email || "User";

    // Get current location to determine base path for nested routes
    const location = useLocation();
    const getBasePath = () => {
        const path = location.pathname;
        // Check more specific paths first
        if (path.startsWith('/solution/search')) return '/solution/search';
        if (path.startsWith('/solution')) return '/solution';
        if (path.startsWith('/search')) return '/search';
        if (path.startsWith('/chat')) return '/chat';
        return '/solution'; // default fallback
    };

    // Default handlers that navigate to Solution page if not provided
    const handleSave = () => {
        if (onSave) {
            onSave();
        } else {
            // Navigate to Solution page for save functionality
            navigate('/solution');
        }
    };

    const handleNew = () => {
        if (onNew) {
            onNew();
        } else {
            // Navigate to a fresh page based on current context
            const currentPath = location.pathname;
            if (currentPath.startsWith('/chat')) {
                // Signal Chat page to clear its storage
                sessionStorage.setItem('clear_chat_state', 'true');
                navigate('/chat');
                window.location.reload();
            } else if (currentPath.startsWith('/search') && !currentPath.includes('/solution')) {
                // Signal Search page to clear its storage
                sessionStorage.setItem('clear_search_state', 'true');
                navigate('/search');
                window.location.reload();
            } else {
                // Default: navigate to Solution page for new project
                navigate('/solution');
            }
        }
    };

    const handleProjectSelect = (project: any) => {
        if (onProjectSelect) {
            onProjectSelect(project);
        } else {
            // Navigate to Solution page with project
            navigate('/solution');
        }
    };

    const handleProjectDelete = (projectId: string) => {
        if (onProjectDelete) {
            onProjectDelete(projectId);
        }
    };

    return (
        <>
            <header className={`glass-header px-6 py-4 fixed top-0 w-full z-50 ${className}`}>
                <div className="flex items-center justify-between">
                    {/* Left side - Logo and optional children (tabs, etc.) */}
                    <div className="flex items-center gap-4">
                        <div className="w-16 h-16 rounded-full overflow-hidden shadow-lg border-2 border-white/50">
                            <video
                                src="/animation.mp4"
                                autoPlay
                                muted
                                playsInline
                                disablePictureInPicture
                                controls={false}
                                onContextMenu={(e) => e.preventDefault()}
                                onError={(e) => {
                                    // Retry loading on error (handles 304 cache issues)
                                    const video = e.currentTarget;
                                    video.load();
                                    video.play().catch(() => { });
                                }}
                                className="w-full h-full object-cover pointer-events-none"
                            />
                        </div>

                        {/* Optional children slot - can be used for tabs, navigation, etc. */}
                        {children}
                    </div>

                    {/* Right side - Action Buttons and Profile */}
                    <div className="flex items-center gap-2">
                        {/* Project Action Buttons */}
                        {showProjectActions && (
                            <>
                                <Tooltip>
                                    <TooltipTrigger asChild>
                                        <Button
                                            variant="outline"
                                            size="sm"
                                            onClick={handleSave}
                                            className="rounded-lg p-2 hover:bg-transparent transition-transform hover:scale-[1.2]"
                                        >
                                            <Save className="h-4 w-4" />
                                        </Button>
                                    </TooltipTrigger>
                                    <TooltipContent><p>Save</p></TooltipContent>
                                </Tooltip>

                                <Tooltip>
                                    <TooltipTrigger asChild>
                                        <Button
                                            variant="outline"
                                            size="sm"
                                            onClick={handleNew}
                                            className="rounded-lg p-2 hover:bg-transparent transition-transform hover:scale-[1.2]"
                                        >
                                            <FileText className="h-4 w-4" />
                                        </Button>
                                    </TooltipTrigger>
                                    <TooltipContent><p>New</p></TooltipContent>
                                </Tooltip>

                                <ProjectListDialog
                                    open={isProjectListOpen}
                                    onOpenChange={setIsProjectListOpen}
                                    onProjectSelect={handleProjectSelect}
                                    onProjectDelete={handleProjectDelete}
                                >
                                    <Tooltip>
                                        <TooltipTrigger asChild>
                                            <Button
                                                variant="outline"
                                                size="sm"
                                                className="rounded-lg p-2 hover:bg-transparent transition-transform hover:scale-[1.2]"
                                                onClick={() => setIsProjectListOpen(true)}
                                            >
                                                <FolderOpen className="h-4 w-4" />
                                            </Button>
                                        </TooltipTrigger>
                                        <TooltipContent><p>Open</p></TooltipContent>
                                    </Tooltip>
                                </ProjectListDialog>
                            </>
                        )}

                        {/* Profile Dropdown */}
                        <DropdownMenu>
                            <Tooltip>
                                <TooltipTrigger asChild>
                                    <DropdownMenuTrigger asChild>
                                        <Button
                                            variant="outline"
                                            className="text-sm font-semibold text-muted-foreground p-2 hover:bg-transparent transition-transform hover:scale-[1.2]"
                                        >
                                            <div className="w-7 h-7 rounded-full bg-[#0F6CBD] flex items-center justify-center text-white font-bold">
                                                {profileButtonLabel.charAt(0)}
                                            </div>
                                        </Button>
                                    </DropdownMenuTrigger>
                                </TooltipTrigger>
                                <TooltipContent><p>Profile</p></TooltipContent>
                            </Tooltip>
                            <DropdownMenuContent
                                className="w-56 mt-1 rounded-xl bg-gradient-to-br from-[#F5FAFC]/90 to-[#EAF6FB]/90 dark:from-slate-900/90 dark:to-slate-900/50 backdrop-blur-2xl border border-white/20 dark:border-slate-700/30 shadow-2xl"
                                align="end"
                            >
                                <DropdownMenuLabel className="p-0 font-normal">
                                    <button
                                        onClick={() => setIsProfileEditOpen(true)}
                                        className="w-full flex items-center gap-2 px-2 py-1.5 hover:bg-muted/50 transition-colors text-sm font-semibold rounded-md text-left outline-none cursor-pointer"
                                        title="Click to edit profile"
                                    >
                                        <User className="w-4 h-4" />
                                        {profileFullName}
                                    </button>
                                </DropdownMenuLabel>
                                <DropdownMenuSeparator />

                                {user?.role?.toLowerCase() === "admin" && (
                                    <>
                                        <DropdownMenuItem className="flex gap-2 focus:bg-transparent cursor-pointer focus:text-slate-900 dark:focus:text-slate-100" onClick={() => navigate(`${getBasePath()}/admin`)}>
                                            <Bot className="h-4 w-4" />
                                            Approve Sign Ups
                                        </DropdownMenuItem>
                                        <DropdownMenuItem className="flex gap-2 focus:bg-transparent cursor-pointer focus:text-slate-900 dark:focus:text-slate-100" onClick={() => navigate(`${getBasePath()}/upload`)}>
                                            <Upload className="h-4 w-4" />
                                            Upload
                                        </DropdownMenuItem>
                                        <DropdownMenuSeparator />
                                    </>
                                )}

                                <DropdownMenuItem className="flex gap-2 focus:bg-transparent cursor-pointer focus:text-slate-900 dark:focus:text-slate-100" onClick={logout}>
                                    <LogOut className="h-4 w-4" />
                                    Logout
                                </DropdownMenuItem>
                            </DropdownMenuContent>
                        </DropdownMenu>
                    </div>
                </div>
            </header>

            <ProfileEditDialog
                open={isProfileEditOpen}
                onOpenChange={setIsProfileEditOpen}
            />
        </>
    );
};

export default MainHeader;
