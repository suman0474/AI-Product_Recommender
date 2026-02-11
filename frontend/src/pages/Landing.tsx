import React from 'react';
import { useNavigate } from 'react-router-dom';
import { Building2, Package, BookOpen, Search, MessageSquare, CheckCircle, Shield, Grid, Wrench, ChevronRight, Linkedin, Twitter, Github, Brain, Zap, BarChart3 } from 'lucide-react';
import ScrollAnimation from '../components/ScrollAnimation';

const Landing = () => {
  const navigate = useNavigate();

  const mainFeatures = [
    {
      image: "/icon-openbook-3d.png",
      title: "Solution Engineering",
      description: "Transform your greenfield or brownfield project requirements into precise, actionable specifications. Our AI analyzes your unique needs and proposes the optimal instruments and accessories.",
      benefits: [
        "Greenfield & brownfield project analysis",
        "Complete bill of materials generation",
        "Integration compatibility assessment",
        "Technical specification alignment"
      ],
      badge: "Enterprise & Supplier"
    },
    {
      image: "/icon-search-3d.png",
      title: "Intelligent Search",
      description: "Empower buyers and requisitioners with AI-driven specification assistance. Get intelligent recommendations backed by detailed rationale—not just results, but reasoning.",
      benefits: [
        "Natural language specification input",
        "AI-powered best-match recommendations",
        "Transparent decision rationale",
        "Alternative options comparison"
      ],
      badge: "Enterprise & Supplier"
    },
    {
      image: "/icon-chat-3d.png",
      title: "Quick Chat Assistant",
      description: "Get instant, expert-level answers on instruments, accessories, or general industrial knowledge. Your on-demand technical consultant available 24/7.",
      benefits: [
        "Instant technical Q&A",
        "Industrial knowledge base access",
        "Product comparison guidance",
        "Troubleshooting support"
      ],
      badge: "Enterprise & Supplier",
      badgeColor: "green"
    }
  ];



  const stats = [
    { value: "85%", label: "Faster Specification Time" },
    { value: "50+", label: "Instruments Cataloged" },
    { value: "40%", label: "Cost Savings" },
    { value: "99.2%", label: "Specification Accuracy" }
  ];

  const features = [
    {
      image: "/icon-brain-3d.png",
      title: "Company-Personalized Matching",
      description: "Recommendations aligned to your approved strategy, engineering standards, and inventory availability."
    },
    {
      image: "/icon-chart-3d.png",
      title: "Intelligent Vendor Analysis",
      description: "Side-by-side comparison and scoring across technical fit, compliance, and commercial factors."
    },
    {
      image: "/icon-lightning-3d.png",
      title: "Real-time Validation",
      description: "Instant requirement checks, missing-field prompts, and fast shortlisting."
    },
    {
      image: "/icon-shield-3d.png",
      title: "Secure & Reliable",
      description: "Enterprise-grade security with consistent, explainable outputs you can trust."
    }
  ];

  const productTypes = [
    "Pressure Transmitter",
    "Temperature Transmitter",
    "Humidity Transmitter",
    "Flow Meter",
    "Level Transmitter",
    "pH Sensors"
  ];

  const steps = [
    {
      number: 1,
      title: "Define Requirements",
      description: "Input your project specifications naturally—whether greenfield construction or brownfield upgrades. Our AI understands context."
    },
    {
      number: 2,
      title: "AI Analysis",
      description: "EnGenie cross-references your needs against supplier strategies, category frameworks, and organizational standards."
    },
    {
      number: 3,
      title: "Smart Recommendations",
      description: "Receive curated instrument and accessory recommendations with transparent rationale and compliance verification."
    },
    {
      number: 4,
      title: "Procure with Confidence",
      description: "Move forward with specifications that align with your organization's strategic sourcing objectives and technical standards."
    }
  ];

  const benefits = [
    {
      icon: Shield,
      title: "Supplier Strategy Integration",
      description: "Recommendations align with your preferred supplier agreements, volume commitments, and strategic partnerships."
    },
    {
      icon: Grid,
      title: "Category Framework Compliance",
      description: "Automatic adherence to category strategies, spending policies, and procurement guidelines."
    },
    {
      icon: Wrench,
      title: "Organizational Standards Built-In",
      description: "Technical specifications automatically verified against your engineering standards and safety requirements."
    }
  ];

  return (
    <div className="min-h-screen app-glass-gradient text-foreground">
      {/* Enhanced Header with Navigation */}
      <header className="glass-header">
        <div className="max-w-7xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-12 h-12 rounded-full overflow-hidden shadow-lg">
              <video
                autoPlay

                muted
                playsInline
                className="w-full h-full object-cover"
              >
                <source src="/animation.mp4" type="video/mp4" />
              </video>
            </div>
            <span className="text-2xl font-bold text-gradient-copilot">EnGenie</span>
          </div>
          <nav className="hidden md:flex items-center gap-8">
            {['Solutions', 'Features', 'How It Works', 'Benefits', 'Contact'].map((item) => (
              <a
                key={item}
                href={`#${item.toLowerCase().replace(/\s+/g, '-')}`}
                className="text-sm font-medium hover:text-primary transition-colors relative group"
              >
                {item}
                <span className="absolute left-0 -bottom-1 w-0 h-0.5 bg-primary transition-all duration-300 group-hover:w-full"></span>
              </a>
            ))}
          </nav>
          <div className="flex items-center gap-4">
            <button className="btn-glass-secondary px-6 py-2 rounded-full font-medium" onClick={() => navigate('/login')}>Login</button>
            <button className="btn-glass-primary px-6 py-2 rounded-full font-medium shadow-lg hover:shadow-xl" onClick={() => navigate('/signup')}>Sign Up</button>
          </div>
        </div>
      </header>

      {/* Hero Section */}
      <ScrollAnimation>
        <section className="pt-12 pb-6 md:pt-16 md:pb-20 px-6 max-w-7xl mx-auto">
          <div className="grid lg:grid-cols-2 gap-12 items-center">
            {/* Left Column: Text Content */}
            <div className="text-left space-y-8">
              <ScrollAnimation>
                <div className="inline-flex items-center gap-2 glass-pill mb-4">
                  <span className="w-2 h-2 bg-primary rounded-full animate-pulse"></span>
                  <span className="text-sm font-semibold text-primary">AI-Powered Procurement Intelligence</span>
                </div>
                <h1 className="text-5xl md:text-6xl font-extrabold leading-[1.15] tracking-tight">
                  Transform Your <br className="hidden md:block" />
                  <span className="text-gradient-copilot leading-[1.3] pb-2 inline-block">
                    Industrial Procurement
                  </span>{' '}
                  <br className="hidden md:block" />
                  With Intelligent AI
                </h1>
                <p className="text-lg md:text-xl text-muted-foreground leading-relaxed max-w-xl mt-8">
                  EnGenie revolutionizes how enterprises source and suppliers sell industrial instruments and accessories. Leverage AI that understands requirements, organizational standards, and supplier strategies—delivering precision recommendations in seconds.
                </p>
                <div className="flex flex-col sm:flex-row gap-4 pt-4">
                  <button className="btn-glass-primary px-8 py-4 text-lg rounded-full inline-flex items-center justify-center shadow-lg hover:shadow-xl hover:scale-105 transition-all" onClick={() => navigate('/signup')}>
                    Get Started
                    <ChevronRight className="ml-2 w-5 h-5" />
                  </button>
                  <button className="px-8 py-4 text-lg rounded-full border-2 border-foreground/10 hover:bg-foreground/5 hover:border-foreground/30 transition-all font-semibold" onClick={() => navigate('/login')}>
                    Sign In
                  </button>
                </div>
              </ScrollAnimation>
            </div>

            {/* Right Column: Animated Visuals */}
            <div className="relative hidden lg:block h-[400px]">
              <ScrollAnimation className="h-full">
                {/* Very light background container */}
                <div className="absolute inset-0 bg-gradient-to-br from-slate-50 to-blue-50/20 rounded-3xl p-8 flex items-center justify-center">
                  {/* Dashboard Background - Glass Card */}
                  <div className="relative w-full h-full glass-card backdrop-blur-md rounded-2xl shadow-lg border border-white/50 p-8 flex flex-col gap-5">
                    {/* Header with Search Icon and Input Boxes */}
                    <div className="flex gap-3 items-center">
                      <div className="w-12 h-12 rounded-xl bg-blue-50 flex items-center justify-center border border-blue-100">
                        <Search className="w-5 h-5 text-blue-500" />
                      </div>
                      <div className="flex-1 h-12 bg-slate-50 rounded-xl border border-gray-200"></div>
                      <div className="w-16 h-12 bg-slate-50 rounded-xl border border-gray-200"></div>
                    </div>

                    {/* Blue Progress Bar */}
                    <div className="w-1/3 h-6 bg-[#0F6CBD] rounded-lg"></div>

                    {/* Main Content Grid (2x2) - All Empty with Content Lines */}
                    <div className="grid grid-cols-2 gap-5 flex-1">
                      {/* Top Left */}
                      <div className="bg-slate-50/40 rounded-xl border border-gray-200/60 p-5 flex flex-col gap-3">
                        <div className="h-2.5 bg-gray-200 rounded w-3/4"></div>
                        <div className="h-2.5 bg-gray-200 rounded w-1/2"></div>
                      </div>

                      {/* Top Right */}
                      <div className="bg-slate-50/40 rounded-xl border border-gray-200/60 p-5 flex flex-col gap-3">
                        <div className="h-2.5 bg-gray-200 rounded w-2/3"></div>
                        <div className="h-2.5 bg-gray-200 rounded w-1/2"></div>
                      </div>

                      {/* Bottom Left */}
                      <div className="bg-slate-50/40 rounded-xl border border-gray-200/60 p-5 flex flex-col gap-3">
                        <div className="h-2.5 bg-gray-200 rounded w-3/5"></div>
                        <div className="h-2.5 bg-gray-200 rounded w-4/5"></div>
                      </div>

                      {/* Bottom Right - Empty (Standards Compliant will overlap this) */}
                      <div className="bg-slate-50/40 rounded-xl border border-gray-200/60 p-5 flex flex-col gap-3">
                        <div className="h-2.5 bg-gray-200 rounded w-2/3"></div>
                        <div className="h-2.5 bg-gray-200 rounded w-3/5"></div>
                      </div>
                    </div>
                  </div>

                  {/* Floating Card 1: Smart Search (Left Top) */}
                  <div className="absolute top-[12%] -left-8 glass-card backdrop-blur-md p-3.5 rounded-xl shadow-lg flex items-center gap-3 animate-float border border-white/50 z-20">
                    <div className="w-11 h-11 rounded-xl bg-blue-50 flex items-center justify-center flex-shrink-0 border border-blue-100">
                      <Search className="w-5 h-5 text-blue-500" />
                    </div>
                    <div>
                      <div className="font-semibold text-sm text-gray-900">Smart Search</div>
                      <div className="text-xs text-gray-500">AI-powered discovery</div>
                    </div>
                  </div>

                  {/* Floating Card 2: Quick Chat (Left Bottom) */}
                  <div className="absolute top-[55%] -left-6 glass-card backdrop-blur-md p-3.5 rounded-xl shadow-lg flex items-center gap-3 animate-float animation-delay-2000 border border-white/50 z-20">
                    <div className="w-11 h-11 rounded-xl bg-orange-50 flex items-center justify-center flex-shrink-0 border border-orange-100">
                      <MessageSquare className="w-5 h-5 text-orange-500" />
                    </div>
                    <div>
                      <div className="font-semibold text-sm text-gray-900">Quick Chat</div>
                      <div className="text-xs text-gray-500">Instant answers</div>
                    </div>
                  </div>

                  {/* Floating Card 3: Standards Compliant (Overlapping Bottom Right) */}
                  <div className="absolute bottom-12 -right-4 glass-card backdrop-blur-md p-3.5 rounded-xl shadow-lg flex items-center gap-3 animate-float animation-delay-4000 border border-white/50 z-20 min-w-[240px]">
                    <div className="w-11 h-11 rounded-xl bg-teal-50 flex items-center justify-center flex-shrink-0 border border-teal-100">
                      <Shield className="w-5 h-5 text-teal-600" />
                    </div>
                    <div className="flex-1 min-w-0">
                      <div className="font-semibold text-sm text-gray-900">Standards Compliant</div>
                      <div className="text-xs text-gray-500">Org policies built-in</div>
                    </div>
                  </div>
                </div>
              </ScrollAnimation>
            </div>
          </div>
        </section>
      </ScrollAnimation>

      {/* Powerful AI-Driven Features Section */}
      <section className="pt-4 pb-16">
        <div className="max-w-7xl mx-auto px-6">
          <ScrollAnimation>
            <div className="text-center mb-16">
              <h2 className="text-4xl font-bold mb-4 text-gradient inline-block">
                Powerful AI-Driven Features
              </h2>
              <p className="text-lg text-muted-foreground">
                Experience the next generation of product recommendation technology
              </p>
            </div>
          </ScrollAnimation>

          <div className="grid md:grid-cols-2 lg:grid-cols-4 gap-8">
            {features.map((feature, index) => (
              <ScrollAnimation key={index}>
                <div
                  className="glass-card p-6 group hover:scale-105 transition-transform duration-300 border-white/20 !bg-white/10 hover:!bg-white/20 h-full"
                >
                  <div className="text-center">
                    <div className="mb-6 relative flex items-center justify-center transform transition-transform duration-300 group-hover:scale-110">
                      <img
                        src={feature.image}
                        alt={feature.title}
                        className="w-24 h-24 object-contain mix-blend-multiply filter contrast-125 drop-shadow-xl"
                      />
                    </div>
                    <h3 className="text-lg font-semibold mb-2">{feature.title}</h3>
                  </div>
                  <div>
                    <p className="text-muted-foreground text-center text-sm leading-relaxed">{feature.description}</p>
                  </div>
                </div>
              </ScrollAnimation>
            ))}
          </div>
        </div>
      </section>

      {/* Stats Bar */}
      <section className="py-16 bg-gradient-to-br from-primary/10 to-secondary/10 backdrop-blur-sm">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid grid-cols-2 md:grid-cols-4 gap-8">
            {stats.map((stat, index) => (
              <ScrollAnimation key={index}>
                <div className="text-center">
                  <div className="text-4xl md:text-5xl font-extrabold text-gradient-copilot mb-2">{stat.value}</div>
                  <div className="text-sm text-muted-foreground font-medium">{stat.label}</div>
                </div>
              </ScrollAnimation>
            ))}
          </div>
        </div>
      </section>

      {/* Who It's For Section */}
      <section id="solutions" className="py-16">
        <div className="max-w-7xl mx-auto px-6">
          <ScrollAnimation>
            <div className="text-center mb-16">
              <span className="inline-block glass-pill text-primary font-semibold mb-4">Built For You</span>
              <h2 className="text-4xl md:text-5xl font-bold mb-4">
                Powering Success for Enterprises & Suppliers
              </h2>
              <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
                EnGenie creates value across the entire procurement ecosystem—streamlining operations for buyers while accelerating sales for suppliers.
              </p>
            </div>
          </ScrollAnimation>

          <div className="grid md:grid-cols-2 gap-8">
            {/* Enterprise Card */}
            <ScrollAnimation>
              <div className="glass-card p-8 border-l-4 border-primary hover:scale-[1.02] transition-all duration-300">
                <div className="flex items-center gap-4 mb-6">
                  <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-primary to-secondary flex items-center justify-center shadow-lg">
                    <Building2 className="w-8 h-8 text-white" />
                  </div>
                  <div>
                    <h3 className="text-2xl font-bold">For Enterprises</h3>
                    <p className="text-sm text-muted-foreground">Procurement teams & buyers</p>
                  </div>
                </div>
                <p className="text-muted-foreground mb-6 leading-relaxed">
                  Transform your procurement operations with AI that understands your organizational DNA—from supplier strategies to category frameworks and compliance requirements.
                </p>
                <ul className="space-y-3">
                  {[
                    "Reduce specification time by up to 85%",
                    "Ensure compliance with organizational standards",
                    "Leverage preferred supplier agreements automatically",
                    "Get AI-powered recommendations with transparent rationale",
                    "Streamline greenfield & brownfield project specifications"
                  ].map((benefit, idx) => (
                    <li key={idx} className="flex items-start gap-3">
                      <CheckCircle className="w-5 h-5 text-secondary flex-shrink-0 mt-0.5" />
                      <span className="text-sm">{benefit}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </ScrollAnimation>

            {/* Supplier Card */}
            <ScrollAnimation>
              <div className="glass-card p-8 border-l-4 border-secondary hover:scale-[1.02] transition-all duration-300">
                <div className="flex items-center gap-4 mb-6">
                  <div className="w-16 h-16 rounded-2xl bg-gradient-to-br from-secondary to-primary flex items-center justify-center shadow-lg">
                    <Package className="w-8 h-8 text-white" />
                  </div>
                  <div>
                    <h3 className="text-2xl font-bold">For Suppliers</h3>
                    <p className="text-sm text-muted-foreground">Manufacturers & distributors</p>
                  </div>
                </div>
                <p className="text-muted-foreground mb-6 leading-relaxed">
                  Get your products in front of the right buyers at the right time. EnGenie's AI ensures your instruments and accessories are recommended when they best match customer requirements.
                </p>
                <ul className="space-y-3">
                  {[
                    "Increase product visibility to qualified buyers",
                    "Get matched to projects based on technical specifications",
                    "Benefit from AI-powered product recommendations",
                    "Strengthen strategic partnerships with enterprise buyers",
                    "Access detailed market insights and demand patterns"
                  ].map((benefit, idx) => (
                    <li key={idx} className="flex items-start gap-3">
                      <CheckCircle className="w-5 h-5 text-secondary flex-shrink-0 mt-0.5" />
                      <span className="text-sm">{benefit}</span>
                    </li>
                  ))}
                </ul>
              </div>
            </ScrollAnimation>
          </div>
        </div>
      </section>

      {/* Features Section - 3 Main Modules */}
      <section id="features" className="py-16 bg-gradient-to-br from-primary/5 to-secondary/5">
        <div className="max-w-7xl mx-auto px-6">
          <ScrollAnimation>
            <div className="text-center mb-16">
              <span className="inline-block glass-pill text-primary font-semibold mb-4">Powerful Capabilities</span>
              <h2 className="text-4xl md:text-5xl font-bold mb-4">
                Three Intelligent Modules, One Powerful Platform
              </h2>
              <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
                EnGenie combines advanced AI with deep industrial knowledge to deliver comprehensive procurement intelligence tailored to your organization.
              </p>
            </div>
          </ScrollAnimation>

          <div className="grid md:grid-cols-3 gap-8">
            {mainFeatures.map((feature, index) => (
              <ScrollAnimation key={index}>
                <div className="glass-card p-8 h-full hover:scale-[1.02] transition-all duration-300 flex flex-col group">
                  <div className="w-28 h-28 mx-auto flex items-center justify-center mb-2 transform transition-transform duration-300 group-hover:scale-110">
                    <img
                      src={feature.image}
                      alt={feature.title}
                      className="w-full h-full object-contain mix-blend-multiply filter contrast-125 drop-shadow-xl"
                    />
                  </div>
                  <h3 className="text-2xl font-bold mb-4">{feature.title}</h3>
                  <p className="text-muted-foreground mb-6 leading-relaxed flex-grow">
                    {feature.description}
                  </p>
                  <ul className="space-y-3 mb-6">
                    {feature.benefits.map((benefit, idx) => (
                      <li key={idx} className="flex items-start gap-3">
                        <CheckCircle className="w-5 h-5 text-secondary flex-shrink-0 mt-0.5" />
                        <span className="text-sm">{benefit}</span>
                      </li>
                    ))}
                  </ul>
                  <div className={`inline-flex items-center gap-2 px-4 py-2 rounded-full text-xs font-semibold ${feature.badgeColor === 'green'
                    ? 'bg-secondary/20 text-secondary'
                    : 'bg-primary/20 text-primary'
                    }`}>
                    {feature.badge}
                  </div>
                </div>
              </ScrollAnimation>
            ))}
          </div>
        </div>
      </section>

      {/* How It Works Section */}
      <section id="how-it-works" className="py-16 relative overflow-hidden">
        <div className="absolute inset-0 bg-gradient-to-br from-primary/10 to-secondary/10 backdrop-blur-3xl"></div>
        <div className="relative max-w-7xl mx-auto px-6">
          <ScrollAnimation>
            <div className="text-center mb-16">
              <span className="inline-block glass-pill text-secondary font-semibold mb-4">Seamless Process</span>
              <h2 className="text-4xl md:text-5xl font-bold mb-4">
                Intelligence Built Into Every Step
              </h2>
              <p className="text-lg text-muted-foreground max-w-3xl mx-auto">
                EnGenie integrates your organizational DNA—supplier strategies, category standards, and compliance requirements—into every recommendation.
              </p>
            </div>
          </ScrollAnimation>

          <div className="grid md:grid-cols-4 gap-8">
            {steps.map((step, index) => (
              <ScrollAnimation key={index}>
                <div className="text-center relative">
                  <div className="w-16 h-16 mx-auto mb-6 rounded-full bg-gradient-to-br from-primary to-secondary flex items-center justify-center text-white text-2xl font-bold shadow-lg">
                    {step.number}
                  </div>
                  {index < steps.length - 1 && (
                    <div className="hidden md:block absolute top-8 left-[60%] w-[80%] h-0.5 bg-gradient-to-r from-primary/50 to-transparent"></div>
                  )}
                  <h3 className="text-xl font-bold mb-3">{step.title}</h3>
                  <p className="text-sm text-muted-foreground leading-relaxed">
                    {step.description}
                  </p>
                </div>
              </ScrollAnimation>
            ))}
          </div>
        </div>
      </section>

      {/* Benefits Section */}
      <section id="benefits" className="py-16">
        <div className="max-w-7xl mx-auto px-6">
          <div className="grid md:grid-cols-2 gap-12 items-center">
            {/* Visual Side (Now on Left) */}
            <ScrollAnimation>
              <div className="relative rounded-3xl overflow-hidden shadow-2xl hover:scale-[1.02] transition-transform duration-500">
                <div className="w-full aspect-[4/3] bg-gradient-to-br from-[#0F6CBD] to-[#5FB3E6] p-8 md:p-12 flex items-center justify-center relative group">
                  {/* Main Glass Card */}
                  <div className="w-full h-full glass-card hover:bg-white/55 backdrop-blur-md border border-white/40 rounded-2xl p-6 md:p-8 flex flex-col gap-6 shadow-2xl relative overflow-hidden">

                    {/* Header Badge */}
                    <div className="inline-flex px-4 py-1.5 rounded-lg bg-white/80 backdrop-blur-md border border-white/60 self-start shadow-sm">
                      <span className="text-[#0F6CBD] font-bold text-sm">Strategy Alignment</span>
                    </div>

                    {/* Skeleton Content */}
                    <div className="space-y-4 flex-1">
                      {/* Row 1 - Glassy Content Box */}
                      <div className="bg-slate-50/40 backdrop-blur-sm rounded-xl p-4 space-y-3 border border-white/30 hover:scale-[1.02] transition-transform duration-300">
                        <div className="h-3 bg-gray-400/20 rounded w-1/3"></div>
                        <div className="h-2.5 bg-gray-400/20 rounded w-3/4"></div>
                      </div>

                      {/* Row 2 - Glassy Content Box */}
                      <div className="bg-slate-50/40 backdrop-blur-sm rounded-xl p-4 space-y-3 border border-white/30 hover:scale-[1.02] transition-transform duration-300">
                        <div className="h-3 bg-gray-400/20 rounded w-1/4"></div>
                        <div className="h-2.5 bg-gray-400/20 rounded w-2/3"></div>
                      </div>

                      {/* Row 3 - Selected Item (Active State) */}
                      <div className="bg-blue-50/80 backdrop-blur-md rounded-xl p-4 border border-[#0F6CBD]/50 flex items-center hover:scale-[1.02] transition-all duration-300 gap-4 shadow-sm">
                        <div className="flex-shrink-0">
                          <div className="w-8 h-8 rounded-full bg-[#0F6CBD] flex items-center justify-center shadow-sm">
                            <svg className="w-5 h-5 text-white" fill="none" viewBox="0 0 24 24" stroke="currentColor" strokeWidth="3">
                              <path strokeLinecap="round" strokeLinejoin="round" d="M5 13l4 4L19 7" />
                            </svg>
                          </div>
                        </div>
                        <div className="space-y-3 flex-1">
                          <div className="h-3 bg-[#0F6CBD]/20 rounded w-1/3"></div>
                          <div className="h-2.5 bg-[#0F6CBD]/10 rounded w-3/4"></div>
                        </div>
                        <div className="bg-[#0F6CBD] text-white text-[10px] font-bold px-3 py-1.5 rounded-md shadow-sm">
                          SELECT
                        </div>
                      </div>
                    </div>
                  </div>
                </div>
              </div>
            </ScrollAnimation>

            {/* Text Side (Now on Right) */}
            <ScrollAnimation>
              <div>
                <h2 className="text-4xl md:text-5xl font-bold mb-6 text-foreground">
                  Built for Enterprise Procurement Excellence
                </h2>
                <p className="text-lg text-muted-foreground mb-8 leading-relaxed">
                  EnGenie doesn't just search—it thinks. Every recommendation considers your complete procurement ecosystem, ensuring alignment with strategic objectives and operational requirements.
                </p>

                <div className="space-y-8">
                  {benefits.map((benefit, index) => (
                    <div key={index} className="flex gap-5">
                      <div className="w-12 h-12 rounded-xl bg-blue-50 dark:bg-blue-900/20 flex items-center justify-center flex-shrink-0">
                        <benefit.icon className="w-6 h-6 text-primary" />
                      </div>
                      <div>
                        <h4 className="text-xl font-bold mb-2 text-foreground">{benefit.title}</h4>
                        <p className="text-muted-foreground leading-relaxed">
                          {benefit.description}
                        </p>
                      </div>
                    </div>
                  ))}
                </div>
              </div>
            </ScrollAnimation>
          </div>
        </div>
      </section>



      {/* Product Types Section */}
      <ScrollAnimation>
        <section className="py-16">
          <div className="relative max-w-7xl mx-auto px-6 text-center">
            <h2 className="text-3xl font-bold mb-3">Supported Product Categories</h2>
            <p className="text-base text-muted-foreground mb-8">
              Comprehensive analysis across various industrial sensor types
            </p>
            <div className="grid grid-cols-2 md:grid-cols-3 gap-4">
              {productTypes.map((type, index) => (
                <div
                  key={index}
                  className="p-4 font-medium transition-all duration-300 text-base flex items-center justify-center min-h-[50px] hover:scale-102"
                  style={{
                    backgroundColor: 'rgba(255, 255, 255, 0.35)',
                    backdropFilter: 'blur(16px)',
                    WebkitBackdropFilter: 'blur(16px)',
                    border: '1px solid rgba(255, 255, 255, 0.4)',
                    borderRadius: '0.75rem',
                    boxShadow: '0 4px 30px rgba(0, 0, 0, 0.15)',
                    transition: 'all 0.3s ease',
                    cursor: 'pointer',
                    width: '100%',
                    textAlign: 'center'
                  }}
                  onMouseEnter={(e) => {
                    e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.5)';
                    e.currentTarget.style.backdropFilter = 'blur(20px)';
                    e.currentTarget.style.WebkitBackdropFilter = 'blur(20px)';
                    e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.6)';
                    e.currentTarget.style.transform = 'scale(1.02)';
                  }}
                  onMouseLeave={(e) => {
                    e.currentTarget.style.backgroundColor = 'rgba(255, 255, 255, 0.35)';
                    e.currentTarget.style.backdropFilter = 'blur(16px)';
                    e.currentTarget.style.WebkitBackdropFilter = 'blur(16px)';
                    e.currentTarget.style.borderColor = 'rgba(255, 255, 255, 0.4)';
                    e.currentTarget.style.transform = 'scale(1)';
                  }}
                >
                  {type}
                </div>
              ))}
            </div>
          </div>
        </section>
      </ScrollAnimation>

      {/* CTA Section */}
      <ScrollAnimation>
        <section className="py-16 text-center">
          <div className="glass-card popup-blur-card p-12 max-w-4xl mx-auto border-white/30 !bg-white/15 hover:!bg-white/25 hover:shadow-2xl transition-all duration-300">
            <h2 className="text-4xl font-bold mb-6 text-gradient inline-block">
              Ready to Find Your Perfect Product?
            </h2>
            <p className="text-xl text-muted-foreground mb-8 text-center max-w-2xl mx-auto">
              Join teams who trust EnGenie to standardize selection, reduce rework, and accelerate decisions. Product type detection starts automatically upon entering your requirements.
            </p>
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <button className="btn-glass-primary px-8 py-3 text-lg rounded-full inline-flex items-center justify-center shadow-lg hover:shadow-xl hover:scale-105 transition-all" onClick={() => navigate('/signup')}>
                Create Account
                <ChevronRight className="ml-2 w-5 h-5" />
              </button>
              <button className="btn-glass-secondary px-8 py-3 text-lg rounded-full shadow-md hover:shadow-lg hover:scale-105 transition-all" onClick={() => navigate('/login')}>
                I Already Have an Account
              </button>
            </div>
          </div>
        </section>
      </ScrollAnimation>

      {/* Comprehensive Footer */}
      <footer id="contact" className="bg-gradient-to-br from-primary/10 to-secondary/10 backdrop-blur-sm border-t border-border">
        <div className="max-w-7xl mx-auto px-6 py-12">
          <div className="grid md:grid-cols-4 gap-8 mb-8">
            {/* Brand Column */}
            <div className="md:col-span-1">
              <div className="flex items-center gap-3 mb-4">
                <div className="w-10 h-10 rounded-full overflow-hidden shadow-lg">
                  <video
                    muted
                    playsInline
                    className="w-full h-full object-cover"
                  >
                    <source src="/animation.mp4" type="video/mp4" />
                  </video>
                </div>
                <span className="text-xl font-bold text-gradient-copilot">EnGenie</span>
              </div>
              <p className="text-sm text-muted-foreground leading-relaxed">
                AI-powered procurement intelligence for industrial instruments and accessories. Transforming how enterprises source and suppliers sell.
              </p>
            </div>

            {/* Solutions Column */}
            <div>
              <h4 className="font-bold mb-4">Solutions</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li><a href="#solutions" className="hover:text-primary transition-colors">For Enterprises</a></li>
                <li><a href="#solutions" className="hover:text-primary transition-colors">For Suppliers</a></li>
                <li><a href="#features" className="hover:text-primary transition-colors">Solution Engineering</a></li>
                <li><a href="#features" className="hover:text-primary transition-colors">Intelligent Search</a></li>
                <li><a href="#features" className="hover:text-primary transition-colors">Quick Chat</a></li>
              </ul>
            </div>

            {/* Company Column */}
            <div>
              <h4 className="font-bold mb-4">Company</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li><a href="#" className="hover:text-primary transition-colors">About Us</a></li>
                <li><a href="#" className="hover:text-primary transition-colors">Careers</a></li>
                <li><a href="#" className="hover:text-primary transition-colors">Blog</a></li>
                <li><a href="#" className="hover:text-primary transition-colors">Contact</a></li>
              </ul>
            </div>

            {/* Resources Column */}
            <div>
              <h4 className="font-bold mb-4">Resources</h4>
              <ul className="space-y-2 text-sm text-muted-foreground">
                <li><a href="#" className="hover:text-primary transition-colors">Documentation</a></li>
                <li><a href="#" className="hover:text-primary transition-colors">API Reference</a></li>
                <li><a href="#" className="hover:text-primary transition-colors">Support</a></li>
                <li><a href="#" className="hover:text-primary transition-colors">Privacy Policy</a></li>
              </ul>
            </div>
          </div>

          {/* Footer Bottom */}
          <div className="pt-8 border-t border-border flex flex-col md:flex-row justify-between items-center gap-4">
            <p className="text-sm text-muted-foreground">
              © 2026 EnGenie. All rights reserved.
            </p>
            <div className="flex items-center gap-4">
              <a href="#" className="w-10 h-10 rounded-lg glass-card flex items-center justify-center hover:scale-110 transition-transform" aria-label="LinkedIn">
                <Linkedin className="w-5 h-5 text-primary" />
              </a>
              <a href="#" className="w-10 h-10 rounded-lg glass-card flex items-center justify-center hover:scale-110 transition-transform" aria-label="Twitter">
                <Twitter className="w-5 h-5 text-primary" />
              </a>
              <a href="#" className="w-10 h-10 rounded-lg glass-card flex items-center justify-center hover:scale-110 transition-transform" aria-label="GitHub">
                <Github className="w-5 h-5 text-primary" />
              </a>
            </div>
          </div>
        </div>
      </footer>
    </div>
  );
};

export default Landing;
