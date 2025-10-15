import React from "react";
import { BrowserRouter as Router, Routes, Route } from "react-router-dom";
import Dashboard from "./components/Dashboard";
import RecordPredict from "./components/RecordPredict";

function App() {
  return (
    <Router>
      <div className="min-h-screen bg-gradient-to-r from-blue-100 via-purple-100 to-pink-100">
        <div className="p-6">
          <Routes>
            <Route path="/" element={<Dashboard />} />
            <Route path="/predict" element={<RecordPredict />} />
          </Routes>
        </div>
      </div>
    </Router>
  );
}

export default App;