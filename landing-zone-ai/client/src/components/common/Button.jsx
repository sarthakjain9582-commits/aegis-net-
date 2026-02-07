import React from 'react'

const variants = {
  primary: 'bg-blue-600 text-white hover:bg-blue-700 active:bg-blue-800 shadow-sm shadow-blue-200',
  secondary: 'bg-white text-gray-700 border border-gray-200 hover:bg-gray-50 hover:border-gray-300',
  outline: 'bg-transparent border-2 border-blue-600 text-blue-600 hover:bg-blue-50',
  ghost: 'bg-transparent text-gray-600 hover:bg-gray-100'
}

const sizes = {
  sm: 'px-3 py-1.5 text-sm',
  md: 'px-6 py-2.5 text-base',
  lg: 'px-8 py-3 text-lg'
}

const Button = ({ 
  children, 
  variant = 'primary', 
  size = 'md', 
  className = '', 
  type = 'button',
  ...props 
}) => {
  return (
    <button
      type={type}
      className={`
        inline-flex items-center justify-center 
        rounded-xl font-semibold transition-all duration-200 
        focus:outline-none focus:ring-2 focus:ring-blue-500 focus:ring-offset-2 
        disabled:opacity-50 disabled:cursor-not-allowed
        active:scale-[0.98]
        ${variants[variant]} 
        ${sizes[size]} 
        ${className}
      `}
      {...props}
    >
      {children}
    </button>
  )
}

export default Button
