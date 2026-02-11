import { useState } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Eye, EyeOff, Brain, ArrowLeft } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { Button } from '@/components/ui/button';
import { Input } from '@/components/ui/input';
import { Label } from '@/components/ui/label';

const Login = () => {
  const [username, setUsername] = useState('');
  const [password, setPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const { login, isLoading } = useAuth();
  const navigate = useNavigate();

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!username.trim() || !password.trim()) return;

    try {
      await login({ username: username.trim(), password });
      navigate('/solution');
    } catch (error) {
      // handled in AuthContext
    }
  };

  return (
    <div className="min-h-screen app-glass-gradient flex items-center justify-center px-6">
      <div className="w-full max-w-md transition-all duration-300 hover:scale-[1.02]">
        <div className="acrylic-glass-pill backdrop-blur-3xl p-8 relative transition-all duration-300 hover:shadow-2xl">
          {/* Back Button - Top Left */}
          <button
            onClick={() => navigate('/')}
            className="absolute top-10 left-8 text-muted-foreground hover:text-foreground transition-colors p-2 -ml-2 -mt-2 z-50"
          // title="Back to Home"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>
          {/* Header */}
          <div className="text-center space-y-4 animate-item stagger-1">
            <div className="w-16 h-16 mx-auto rounded-full overflow-hidden shadow-lg">
              <video
                muted
                playsInline
                className="w-full h-full object-cover"
              >
                <source src="/animation.mp4" type="video/mp4" />
              </video>
            </div>
            <div>
              <h1 className="text-3xl font-bold">
                Welcome Back
              </h1>
              <p className="text-muted-foreground mt-2 text-base">
                Sign in to access EnGenie
              </p>
            </div>
          </div>

          {/* Form */}
          <div>
            <form onSubmit={handleSubmit} className="space-y-6">
              {/* Username */}
              <div className="space-y-2 animate-item stagger-2">
                <Label htmlFor="username" className="font-medium">
                  Username
                </Label>
                <Input
                  id="username"
                  type="text"
                  placeholder="Enter your username"
                  value={username}
                  onChange={(e) => setUsername(e.target.value)}
                  className="form-glass-input rounded-xl"
                  required
                />
              </div>

              {/* Password */}
              <div className="space-y-2 animate-item stagger-3">
                <Label htmlFor="password" className="font-medium">
                  Password
                </Label>
                <div className="relative hover:scale-[1.02] transition-all duration-300">
                  <Input
                    id="password"
                    type={showPassword ? 'text' : 'password'}
                    placeholder="Enter your password"
                    value={password}
                    onChange={(e) => setPassword(e.target.value)}
                    className="form-glass-input pr-12 rounded-xl hover:scale-100"
                    required
                  />
                  <Button
                    type="button"
                    variant="ghost"
                    size="sm"
                    className="absolute right-0 top-0 h-full px-3 text-muted-foreground hover:text-foreground transition-colors hover:bg-transparent transition-transform hover:scale-110 active:scale-95"
                    onClick={() => setShowPassword(!showPassword)}
                  >
                    {showPassword ? (
                      <EyeOff className="h-4 w-4" />
                    ) : (
                      <Eye className="h-4 w-4" />
                    )}
                  </Button>
                </div>
              </div>

              {/* Submit Button */}
              <button
                type="submit"
                className="btn-glass-primary w-full font-semibold rounded-xl px-4 py-3 animate-item stagger-4"
                disabled={isLoading || !username.trim() || !password.trim()}
              >
                {isLoading ? (
                  <span className="animate-pulse">Signing in...</span>
                ) : (
                  'Sign In'
                )}
              </button>

              {/* Signup link */}
              <div className="text-center pt-4 animate-item stagger-5">
                <p className="text-muted-foreground text-sm">
                  Don't have an account?{' '}
                  <Link
                    to="/signup"
                    className="font-semibold group relative inline-block"
                  >
                    <span className="relative z-10 text-secondary group-hover:text-primary transition-colors duration-200">
                      Sign up
                    </span>
                    <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-primary group-hover:w-full transition-all duration-300 ease-out"></span>
                  </Link>
                </p>
              </div>
            </form>
          </div>
        </div>
      </div>
    </div>
  );
};

export default Login;
