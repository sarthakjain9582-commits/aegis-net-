import React from 'react'
import { BrowserRouter as Router, Routes, Route } from 'react-router-dom'
import Navbar from './components/common/Navbar'
import LandingPage from './pages/LandingPage'
import Dashboard from './pages/Dashboard'
import History from './pages/History'

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gray-50 flex flex-col">
        <Navbar />
        <main className="flex-grow">
          <Routes>
            <Route path="/" element={<LandingPage />} />
            <Route path="/dashboard" element={<Dashboard />} />
            <Route path="/history" element={<History />} />
          </Routes>
        </main>
      </div>
    </Router>
  )
}

export default App
