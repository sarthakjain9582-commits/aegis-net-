import React from 'react'
import { Link, useLocation } from 'react-router-dom'
import { Plane, Home, BarChart, History } from 'lucide-react'

const Navbar = () => {
  const location = useLocation()
  
  const isActive = (path) => {
    return location.pathname === path 
      ? 'text-blue-600 bg-blue-50' 
      : 'text-gray-600 hover:text-blue-600 hover:bg-gray-50'
  }

  return (
    <nav className="bg-white border-b border-gray-100 sticky top-0 z-50">
      <div className="container mx-auto px-4">
        <div className="flex items-center justify-between h-16">
          {/* Logo */}
          <Link to="/" className="flex items-center gap-2">
            <div className="p-2 bg-blue-600 rounded-lg">
              <Plane className="w-6 h-6 text-white" />
            </div>
            <span className="text-xl font-bold text-gray-900 tracking-tight">AEGIS-NET</span>
          </Link>

          {/* Navigation Links */}
          <div className="hidden md:flex items-center gap-1">
            <Link 
              to="/" 
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${isActive('/')}`}
            >
              <Home className="w-4 h-4" />
              Home
            </Link>
            <Link 
              to="/dashboard" 
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${isActive('/dashboard')}`}
            >
              <BarChart className="w-4 h-4" />
              Dashboard
            </Link>
            <Link 
              to="/history" 
              className={`flex items-center gap-2 px-4 py-2 rounded-lg text-sm font-medium transition-colors ${isActive('/history')}`}
            >
              <History className="w-4 h-4" />
              History
            </Link>
          </div>

          {/* Mobile Menu Button (Placeholder) */}
          <button className="md:hidden p-2 text-gray-600">
            <svg className="w-6 h-6" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 6h16M4 12h16M4 18h16" />
            </svg>
          </button>
        </div>
      </div>
    </nav>
  )
}

export default Navbar
