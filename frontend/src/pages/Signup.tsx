import { useState, useRef } from 'react';
import { useNavigate, Link } from 'react-router-dom';
import { Eye, EyeOff, ArrowLeft, Upload, FileText } from 'lucide-react';
import { useAuth } from '../contexts/AuthContext';
import { Button } from '../components/ui/button';
import { Input } from '../components/ui/input';
import { Label } from '../components/ui/label';
import { uploadStrategyFile, uploadStandardsDocument } from '../components/AIRecommender/api';

const Signup = () => {
  const [step, setStep] = useState(1);

  // Step 1 Fields
  const [firstName, setFirstName] = useState('');
  const [lastName, setLastName] = useState('');
  const [email, setEmail] = useState('');
  const [username, setUsername] = useState('');

  // Step 2 Fields
  const [companyName, setCompanyName] = useState('');
  const [location, setLocation] = useState('');
  const [strategy, setStrategy] = useState('');

  // Strategy file upload
  const [strategyFile, setStrategyFile] = useState<File | null>(null);
  const strategyFileInputRef = useRef<HTMLInputElement>(null);

  // Standards file upload
  const [standardsFile, setStandardsFile] = useState<File | null>(null);
  const standardsFileInputRef = useRef<HTMLInputElement>(null);

  // Step 3 Fields
  const [password, setPassword] = useState('');
  const [confirmPassword, setConfirmPassword] = useState('');
  const [showPassword, setShowPassword] = useState(false);
  const [showConfirmPassword, setShowConfirmPassword] = useState(false);

  // Status
  const [isPending, setIsPending] = useState(false);
  const { signup, isLoading } = useAuth();
  const navigate = useNavigate();

  const handleNext = () => {
    // Basic validation per step
    if (step === 1) {
      if (firstName && lastName && email) setStep(2);
    } else if (step === 2) {
      // Step 2 fields are now optional
      setStep(3);
    }
  };

  const handlePrev = () => {
    setStep(step - 1);
  };

  // Strategy file handlers
  const handleStrategyFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setStrategyFile(e.target.files[0]);
    }
  };

  const removeStrategyFile = () => {
    setStrategyFile(null);
    if (strategyFileInputRef.current) {
      strategyFileInputRef.current.value = '';
    }
  };

  // Standards file handlers
  const handleStandardsFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    if (e.target.files && e.target.files[0]) {
      setStandardsFile(e.target.files[0]);
    }
  };

  const removeStandardsFile = () => {
    setStandardsFile(null);
    if (standardsFileInputRef.current) {
      standardsFileInputRef.current.value = '';
    }
  };

  const handleSubmit = async (e: React.FormEvent) => {
    e.preventDefault();

    if (!firstName || !lastName || !email || !username || !password || password !== confirmPassword) {
      return;
    }

    try {
      // Create FormData to handle signup
      const formData = new FormData();
      formData.append('first_name', firstName.trim());
      formData.append('last_name', lastName.trim());
      formData.append('email', email.trim());
      formData.append('username', username.trim());
      formData.append('password', password);

      // Optional fields
      if (companyName) formData.append('company_name', companyName.trim());
      if (location) formData.append('location', location.trim());
      if (strategy) formData.append('strategy', strategy.trim());

      // Cast to any since AuthContext expects UserCredentials object but we are sending FormData
      // Backend is updated to handle FormData
      await signup(formData as any);

      // Upload strategy document if provided (after signup)
      if (strategyFile) {
        try {
          await uploadStrategyFile(strategyFile);
          console.log("Strategy document uploaded successfully");
        } catch (uploadError) {
          console.error("Strategy document upload failed:", uploadError);
          // Don't block signup success for document upload failure
        }
      }

      // Upload standards document if provided (after signup)
      if (standardsFile) {
        try {
          await uploadStandardsDocument(standardsFile);
          console.log("Standards document uploaded successfully");
        } catch (uploadError) {
          console.error("Standards document upload failed:", uploadError);
          // Don't block signup success for document upload failure
        }
      }

      setIsPending(true);
    } catch (error) {
      console.error("Signup failed:", error);
      setIsPending(false);
    }
  };

  // Step Validations
  const isStep1Valid = firstName && lastName && email;
  const isStep2Valid = true; // Optional
  const isFormValid = isStep1Valid && username && password && confirmPassword && (password === confirmPassword);

  if (isPending) {
    return (
      <div className="h-screen overflow-y-auto custom-no-scrollbar app-glass-gradient flex items-center justify-center p-6">
        <div className="w-full max-w-md">
          <div className="acrylic-glass-pill backdrop-blur-3xl p-8 text-center relative">
            <div className="w-16 h-16 mx-auto mb-6 rounded-full overflow-hidden shadow-lg">
              <video
                muted
                playsInline
                className="w-full h-full object-cover"
              >
                <source src="/animation.mp4" type="video/mp4" />
              </video>
            </div>
            <h2 className="text-2xl font-bold mb-4">Account Created!</h2>
            <p className="text-muted-foreground mb-6">
              Your account has been created and is awaiting admin approval. You will receive an email once your account is active.
            </p>
            <button onClick={() => navigate('/login')} className="btn-glass-primary w-full px-4 py-3 rounded-lg">Go to Login</button>
          </div>
        </div>
      </div>
    );
  }

  return (
    <div className="min-h-screen app-glass-gradient flex items-center justify-center py-6 px-6">
      <div className="w-full max-w-md transition-all duration-300 hover:scale-[1.02]">
        <div className="acrylic-glass-pill backdrop-blur-3xl p-8 relative transition-all duration-300 hover:shadow-2xl flex flex-col">
          {/* Back Button */}
          <button
            onClick={() => navigate('/')}
            className="absolute top-10 left-8 text-muted-foreground hover:text-foreground transition-colors p-2 -ml-2 -mt-2 z-50"
            title="Back to Home"
          >
            <ArrowLeft className="h-5 w-5" />
          </button>

          {/* Header */}
          <div className="text-center space-y-4 mb-8 animate-item stagger-1">
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
              <h1 className="text-3xl font-bold">Create Account</h1>
              <p className="text-muted-foreground mt-2 text-base">Step {step} of 3</p>
            </div>
          </div>

          <form onSubmit={handleSubmit} className="flex-1 flex flex-col justify-between space-y-6">

            {/* STEP 1: Personal Info */}
            {step === 1 && (
              <div className="space-y-4">
                <div className="space-y-2 animate-item stagger-2">
                  <Label htmlFor="firstName">First Name</Label>
                  <Input id="firstName" value={firstName} onChange={(e) => setFirstName(e.target.value)} className="form-glass-input rounded-xl" placeholder="John" required />
                </div>
                <div className="space-y-2 animate-item stagger-3">
                  <Label htmlFor="lastName">Last Name</Label>
                  <Input id="lastName" value={lastName} onChange={(e) => setLastName(e.target.value)} className="form-glass-input rounded-xl" placeholder="Doe" required />
                </div>
                <div className="space-y-2 animate-item stagger-4">
                  <Label htmlFor="email">Email</Label>
                  <Input id="email" type="email" value={email} onChange={(e) => setEmail(e.target.value)} className="form-glass-input rounded-xl" placeholder="john@example.com" required />
                </div>
              </div>
            )}

            {/* STEP 2: Company & Strategy */}
            {step === 2 && (
              <div className="space-y-4">
                <div className="space-y-2 animate-item stagger-2">
                  <Label htmlFor="companyName">Company Name</Label>
                  <Input id="companyName" value={companyName} onChange={(e) => setCompanyName(e.target.value)} className="form-glass-input rounded-xl" placeholder="Acme Corp" />
                </div>
                <div className="space-y-2 animate-item stagger-3">
                  <Label htmlFor="location">Location</Label>
                  <Input id="location" value={location} onChange={(e) => setLocation(e.target.value)} className="form-glass-input rounded-xl" placeholder="New York" />
                </div>

                {/* Strategy Input with File Upload */}
                <div className="space-y-2 animate-item stagger-4">
                  <Label htmlFor="strategy">Strategy Document</Label>
                  <div className="relative hover:scale-[1.02] transition-all duration-300">
                    <Input
                      id="strategy"
                      value={strategy}
                      onChange={(e) => setStrategy(e.target.value)}
                      className="form-glass-input rounded-xl pr-12 hover:scale-100"
                      placeholder="Procurement Strategy"
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="absolute right-0 top-0 h-full px-3 text-muted-foreground hover:text-foreground transition-colors hover:bg-transparent transition-transform hover:scale-110 active:scale-95"
                      onClick={() => strategyFileInputRef.current?.click()}
                      title="Upload Strategy Document"
                    >
                      <Upload className={`h-4 w-4 ${strategyFile ? "text-primary" : ""}`} />
                    </Button>
                    <Input
                      type="file"
                      ref={strategyFileInputRef}
                      className="hidden"
                      onChange={handleStrategyFileChange}
                      accept=".pdf,.doc,.docx,.xls,.xlsx,.csv,.txt"
                    />
                  </div>
                  {strategyFile && (
                    <div className="flex items-center justify-between text-xs text-muted-foreground mt-1 px-1">
                      <span>Strategy: {strategyFile.name}</span>
                      <button type="button" onClick={removeStrategyFile} className="text-red-500 hover:text-red-700">Remove</button>
                    </div>
                  )}
                </div>

                {/* Standards Document Upload */}
                <div className="space-y-2 animate-item stagger-5">
                  <Label htmlFor="standards">Standards Document</Label>
                  <div className="relative hover:scale-[1.02] transition-all duration-300">
                    <Input
                      id="standards"
                      readOnly
                      value={standardsFile ? standardsFile.name : ""}
                      className="form-glass-input rounded-xl pr-12 hover:scale-100 cursor-pointer"
                      placeholder="Upload Standards Document"
                      onClick={() => standardsFileInputRef.current?.click()}
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="absolute right-0 top-0 h-full px-3 text-muted-foreground hover:text-foreground transition-colors hover:bg-transparent transition-transform hover:scale-110 active:scale-95"
                      onClick={() => standardsFileInputRef.current?.click()}
                      title="Upload Standards Document"
                    >
                      <Upload className={`h-4 w-4 ${standardsFile ? "text-primary" : ""}`} />
                    </Button>
                    <Input
                      type="file"
                      ref={standardsFileInputRef}
                      className="hidden"
                      onChange={handleStandardsFileChange}
                      accept=".pdf,.doc,.docx,.xls,.xlsx,.csv,.txt"
                    />
                  </div>
                  {standardsFile && (
                    <div className="flex items-center justify-between text-xs text-muted-foreground mt-1 px-1">
                      <span>Standards: {standardsFile.name}</span>
                      <button type="button" onClick={removeStandardsFile} className="text-red-500 hover:text-red-700">Remove</button>
                    </div>
                  )}
                </div>
              </div>
            )}

            {/* STEP 3: Security */}
            {step === 3 && (
              <div className="space-y-4">
                <div className="space-y-2 animate-item stagger-1">
                  <Label htmlFor="username">Username</Label>
                  <Input id="username" value={username} onChange={(e) => setUsername(e.target.value)} className="form-glass-input rounded-xl" placeholder="johndoe" required />
                </div>

                <div className="space-y-2 animate-item stagger-2">
                  <Label htmlFor="password">Password</Label>
                  <div className="relative hover:scale-[1.02] transition-all duration-300">
                    <Input
                      id="password"
                      type={showPassword ? 'text' : 'password'}
                      value={password}
                      onChange={(e) => setPassword(e.target.value)}
                      className="form-glass-input pr-12 rounded-xl hover:scale-100"
                      placeholder="Create password"
                      required
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="absolute right-0 top-0 h-full px-3 text-muted-foreground hover:text-foreground transition-colors hover:bg-transparent transition-transform hover:scale-110 active:scale-95"
                      onClick={() => setShowPassword(!showPassword)}
                    >
                      {showPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </Button>
                  </div>
                </div>

                <div className="space-y-2 animate-item stagger-3">
                  <Label htmlFor="confirmPassword">Confirm Password</Label>
                  <div className="relative hover:scale-[1.02] transition-all duration-300">
                    <Input
                      id="confirmPassword"
                      type={showConfirmPassword ? 'text' : 'password'}
                      value={confirmPassword}
                      onChange={(e) => setConfirmPassword(e.target.value)}
                      className="form-glass-input pr-12 rounded-xl hover:scale-100"
                      placeholder="Confirm password"
                      required
                    />
                    <Button
                      type="button"
                      variant="ghost"
                      size="sm"
                      className="absolute right-0 top-0 h-full px-3 text-muted-foreground hover:text-foreground transition-colors hover:bg-transparent transition-transform hover:scale-110 active:scale-95"
                      onClick={() => setShowConfirmPassword(!showConfirmPassword)}
                    >
                      {showConfirmPassword ? <EyeOff className="h-4 w-4" /> : <Eye className="h-4 w-4" />}
                    </Button>
                  </div>
                  {confirmPassword && password !== confirmPassword && (
                    <p className="text-sm text-red-500 mt-1">Passwords do not match</p>
                  )}
                </div>
              </div>
            )}

            {/* Navigation Buttons */}
            <div className="flex justify-between items-center pt-4 animate-item stagger-6">
              {/* Left Button (Hidden on Step 1) */}
              <div className="w-1/3">
                {step > 1 && (
                  <Button type="button" variant="ghost" onClick={handlePrev} className="text-muted-foreground hover:text-foreground rounded-xl">
                    Previous
                  </Button>
                )}
              </div>

              {/* Step Indicators (Optional visual) */}
              <div className="flex gap-2">
                {[1, 2, 3].map(i => (
                  <div key={i} className={`h-2 w-2 rounded-full transition-colors ${step === i ? 'bg-primary' : 'bg-gray-300/50'}`} />
                ))}
              </div>

              {/* Right Button (Next or Submit) */}
              <div className="w-1/3 flex justify-end">
                {step < 3 ? (
                  <Button
                    type="button"
                    onClick={handleNext}
                    className="btn-glass-primary w-full font-semibold rounded-xl px-4 py-3 animate-item"
                    disabled={step === 1 ? !isStep1Valid : !isStep2Valid}
                  >
                    Next
                  </Button>
                ) : (
                  <Button
                    type="submit"
                    className="btn-glass-primary rounded-xl px-8 py-3 h-auto font-semibold"
                    disabled={isLoading || !isFormValid}
                  >
                    {isLoading ? 'Creating...' : 'Sign Up'}
                  </Button>
                )}
              </div>
            </div>

            {/* Login Link */}
            <div className="text-center pt-2 animate-item stagger-7">
              <p className="text-muted-foreground text-sm">
                Already have an account?{' '}
                <Link
                  to="/login"
                  className="font-semibold group relative inline-block"
                >
                  <span className="relative z-10 text-secondary group-hover:text-primary transition-colors duration-200">
                    Sign in
                  </span>
                  <span className="absolute bottom-0 left-0 w-0 h-0.5 bg-primary group-hover:w-full transition-all duration-300 ease-out"></span>
                </Link>
              </p>
            </div>
          </form>
        </div>
      </div>
    </div>
  );
};

export default Signup;
