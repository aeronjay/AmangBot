import { useState } from 'react'
import './index.css'
import ChatInterface from '../components/chatInterface'
function App() {

  return (
    <>
      <div className='h-screen bg-gray-100 flex items-center justify-center'>
        <ChatInterface />
      </div>
    </>
  )
}

export default App
