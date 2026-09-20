import React, { useState } from 'react';
import Navbar from './components/Navbar';
import Footer from './components/Footer';
import LandingPage from './pages/LandingPage';
import AnalyzePage from './pages/AnalyzePage';
import ModelsPage from './pages/ModelsPage';
import LivePage from './pages/LivePage';
import HistoryPage from './pages/HistoryPage';

export default function App() {
  const [activePage, setActivePage] = useState('landing');

  return (
    <div className="min-h-screen bg-[#0b0f17] text-slate-100 flex flex-col justify-between font-sans selection:bg-cyan-500 selection:text-slate-950">
      <div>
        <Navbar activePage={activePage} setActivePage={setActivePage} />
        <main className="animate-fade-in">
          {activePage === 'landing' && <LandingPage setActivePage={setActivePage} />}
          {activePage === 'analyze' && <AnalyzePage setActivePage={setActivePage} />}
          {activePage === 'models' && <ModelsPage />}
          {activePage === 'live' && <LivePage />}
          {activePage === 'history' && <HistoryPage />}
        </main>
      </div>
      <Footer setActivePage={setActivePage} />
    </div>
  );
}
