import React from 'react'
import { Link } from 'react-router-dom'
import { ArrowRight, Activity, Layers, Zap, Eye, AlertTriangle, CheckCircle2, XCircle } from 'lucide-react'
import Button from '../components/common/Button'

const LandingPage = () => {
  return (
    <div className="flex flex-col min-h-screen bg-white">
      {/* Hero Section */}
      <section className="relative overflow-hidden bg-slate-900 text-white py-24 lg:py-32">
        <div className="absolute inset-0 bg-[url('/grid.svg')] opacity-10"></div>
        <div className="absolute top-0 right-0 w-1/2 h-full bg-gradient-to-l from-blue-900/20 to-transparent"></div>
        
        <div className="container mx-auto px-4 relative z-10">
          <div className="max-w-4xl mx-auto text-center">
            <div className="inline-flex items-center gap-2 px-4 py-2 rounded-full bg-blue-500/10 border border-blue-500/20 text-blue-300 font-medium text-sm mb-6">
              <Activity className="w-4 h-4" />
              <span>v2.0 Now with Test-Time Augmentation</span>
            </div>
            
            <h1 className="text-5xl lg:text-7xl font-bold tracking-tight mb-6 bg-clip-text text-transparent bg-gradient-to-r from-white via-blue-100 to-blue-400">
              Autonomous Landing <br /> Intelligence
            </h1>
            
            <p className="text-xl text-slate-300 mb-10 max-w-2xl mx-auto leading-relaxed">
              Real-time semantic segmentation for UAVs. We identify safe landing zones in complex unstructured environments with uncertainty-aware AI.
            </p>
            
            <div className="flex flex-col sm:flex-row gap-4 justify-center">
              <Link to="/dashboard">
                <Button size="lg" className="w-full sm:w-auto gap-2 shadow-blue-500/25">
                  Analyze Terrain <ArrowRight className="w-5 h-5" />
                </Button>
              </Link>
              <Button variant="outline" size="lg" className="w-full sm:w-auto text-white border-slate-700 hover:bg-slate-800">
                Read Research
              </Button>
            </div>
          </div>
        </div>
      </section>

      {/* WORKING SPECIMEN / DEMO SECTION */}
      <section className="py-24 bg-slate-50 relative">
        <div className="container mx-auto px-4">
          <div className="text-center mb-16">
            <h2 className="text-3xl font-bold text-slate-900 mb-4">See the Model in Action</h2>
            <p className="text-slate-600 max-w-2xl mx-auto">
              Our model processes high-resolution aerial imagery to generate pixel-level safety maps.
              Below are actual outputs from the WildUAV dataset.
            </p>
          </div>

          <div className="grid lg:grid-cols-2 gap-8 lg:gap-12 items-start max-w-6xl mx-auto">
            
            {/* Specimen 1: Forest */}
            <div className="group bg-white rounded-2xl shadow-xl shadow-slate-200/50 overflow-hidden border border-slate-100 transition-all hover:shadow-2xl hover:-translate-y-1">
              <div className="relative aspect-video bg-slate-100 overflow-hidden">
                <img 
                  src="/assets/0004.gif" 
                  alt="Forest Analysis" 
                  className="w-full h-full object-cover transform group-hover:scale-105 transition-transform duration-700"
                />
                
                {/* HUD Overlay */}
                <div className="absolute top-4 left-4 flex gap-2">
                  <span className="bg-black/60 backdrop-blur-md text-white px-3 py-1 rounded-full text-xs font-medium flex items-center gap-1">
                    <Layers className="w-3 h-3" /> Forest
                  </span>
                  <span className="bg-green-500/90 text-white px-3 py-1 rounded-full text-xs font-bold shadow-lg">
                    Safe
                  </span>
                </div>
              </div>
              
              <div className="p-8">
                <h3 className="text-lg font-bold text-slate-900 mb-2 flex items-center gap-2">
                  🌲 Complex Vegetation Handling
                </h3>
                <p className="text-slate-600 text-sm leading-relaxed mb-6">
                  The model successfully filters out tree canopies and uneven foliage (Red), identifying small clearings (Green) suitable for emergency landing.
                </p>
                
                <div className="flex items-center gap-4 text-xs font-medium text-slate-500 border-t border-slate-100 pt-4">
                  <div className="flex items-center gap-1">
                    <CheckCircle2 className="w-4 h-4 text-green-500" /> Safe Zone
                  </div>
                  <div className="flex items-center gap-1">
                    <XCircle className="w-4 h-4 text-red-500" /> Hazard
                  </div>
                  <div className="flex items-center gap-1 ml-auto">
                    <Activity className="w-4 h-4 text-blue-500" /> 92% Confidence
                  </div>
                </div>
              </div>
            </div>

            {/* Specimen 2: Open Field */}
            <div className="group bg-white rounded-2xl shadow-xl shadow-slate-200/50 overflow-hidden border border-slate-100 transition-all hover:shadow-2xl hover:-translate-y-1">
              <div className="relative aspect-video bg-slate-100 overflow-hidden">
                <img 
                  src="/assets/0130.gif" 
                  alt="Open Field Analysis" 
                  className="w-full h-full object-cover transform group-hover:scale-105 transition-transform duration-700"
                />
                 <div className="absolute top-4 left-4 flex gap-2">
                  <span className="bg-black/60 backdrop-blur-md text-white px-3 py-1 rounded-full text-xs font-medium flex items-center gap-1">
                    <Layers className="w-3 h-3" /> Open Terrain
                  </span>
                  <span className="bg-blue-500/90 text-white px-3 py-1 rounded-full text-xs font-bold shadow-lg">
                    Analyzing
                  </span>
                </div>
              </div>
              
              <div className="p-8">
                <h3 className="text-lg font-bold text-slate-900 mb-2 flex items-center gap-2">
                  🌾 Surface Regularity Logic
                </h3>
                <p className="text-slate-600 text-sm leading-relaxed mb-6">
                  In open fields, the model doesn't just look for "ground" but assesses flatness. 
                  Slight slopes or rough patches are marked unsafe (Red) despite looking flat to the naked eye.
                </p>
                
                 <div className="flex items-center gap-4 text-xs font-medium text-slate-500 border-t border-slate-100 pt-4">
                  <div className="flex items-center gap-1">
                    <Eye className="w-4 h-4 text-slate-400" /> Variance Check
                  </div>
                  <div className="flex items-center gap-1 ml-auto">
                    <Activity className="w-4 h-4 text-blue-500" /> 95% Confidence
                  </div>
                </div>
              </div>
            </div>

          </div>
        </div>
      </section>

      {/* Tech Stack / How It Works */}
      <section className="py-24 bg-white">
         <div className="container mx-auto px-4">
           <div className="text-left mb-12 border-l-4 border-blue-600 pl-6">
              <h2 className="text-3xl font-bold text-slate-900">Under the Hood</h2>
              <p className="text-slate-500 mt-2">Powered by state-of-the-art computer vision pipeline</p>
           </div>
           
           <div className="grid md:grid-cols-4 gap-8">
              {[
                {
                  icon: <Layers className="w-6 h-6 text-blue-600" />,
                  title: "YOLOv8-Nano",
                  desc: "Lightweight segmentation backbone optimized for edge devices."
                },
                {
                  icon: <Zap className="w-6 h-6 text-amber-500" />,
                  title: "Test-Time Augmentation",
                  desc: "Robustness via multi-scale inference (0.75x, 1.0x, 1.25x)."
                },
                {
                  icon: <AlertTriangle className="w-6 h-6 text-red-500" />,
                  title: "Uncertainty Estimation",
                  desc: "Variance-based filtering to reject low-confidence predictions."
                },
                {
                  icon: <Activity className="w-6 h-6 text-green-500" />,
                  title: "Superpixel Smoothing",
                  desc: "SLIC-based post-processing for coherent safety zones."
                }
              ].map((item, i) => (
                <div key={i} className="p-6 rounded-xl bg-slate-50 border border-slate-100 hover:border-blue-200 transition-colors">
                  <div className="w-12 h-12 bg-white rounded-lg shadow-sm flex items-center justify-center mb-4">
                    {item.icon}
                  </div>
                  <h3 className="font-bold text-slate-900 mb-2">{item.title}</h3>
                  <p className="text-sm text-slate-600">{item.desc}</p>
                </div>
              ))}
           </div>
         </div>
      </section>
    </div>
  )
}

export default LandingPage
