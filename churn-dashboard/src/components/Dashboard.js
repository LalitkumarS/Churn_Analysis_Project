import React, { useEffect, useState } from "react";
import axios from "axios";
import { Link, useLocation } from "react-router-dom";
import "./Dashboard.css";

export default function Dashboard() {
  const [metrics, setMetrics] = useState(null);
  const location = useLocation();

  useEffect(() => {
    axios.get("http://localhost:5000/api/metrics").then(res => {
      setMetrics(res.data);
    });
  }, []);

  if (!metrics) return (
    <div className="loading-container">
      <div className="loading-spinner"></div>
      <p className="loading-text">
        Loading metrics...
      </p>
    </div>
  );

  // Extract confusion matrix values
  const confusionMatrix = metrics.confusion_matrix;
  const [[tn, fp], [fn, tp]] = confusionMatrix;

  return (
    <div className="dashboard-container">
      {/* Centered Header */}
      <header className="dashboard-header">
        <h1 className="dashboard-title">
          📊 Telecom Churn Dashboard
        </h1>
        
        {/* Navigation Bar */}
        <nav className="dashboard-nav">
          <ul className="nav-list">
            <li className="nav-item">
              <Link 
                to="/" 
                className={`nav-link ${location.pathname === '/' ? 'active' : ''}`}
              >
                Dashboard
              </Link>
            </li>
            <li className="nav-item">
              <Link 
                to="/predict" 
                className={`nav-link ${location.pathname === '/predict' ? 'active' : ''}`}
              >
                Predict Record
              </Link>
            </li>
          </ul>
        </nav>
      </header>

      <div className="metrics-grid">
        {/* Accuracy Card */}
        <div className="metric-card">
          <h2 className="card-title">Accuracy</h2>
          <p className="metric-value">{(metrics.accuracy * 100).toFixed(2)}%</p>
          <div className="progress-container">
            <div
              className="progress-bar progress-accuracy"
              style={{ width: `${metrics.accuracy * 100}%` }}
            ></div>
          </div>
        </div>

        {/* AUC Score Card */}
        <div className="metric-card">
          <h2 className="card-title">AUC Score</h2>
          <p className="metric-value">{(metrics.auc * 100).toFixed(2)}%</p>
          <div className="progress-container">
            <div
              className="progress-bar progress-auc"
              style={{ width: `${metrics.auc * 100}%` }}
            ></div>
          </div>
        </div>

        {/* Confusion Matrix */}
        <div className="confusion-matrix-card">
          <h2 className="card-title">Confusion Matrix</h2>
          
          <div className="confusion-matrix-container">
            <div className="confusion-matrix-grid">
              {/* Empty corner */}
              <div className="matrix-corner"></div>
              
              {/* Column headers */}
              <div className="matrix-header">Predicted No</div>
              <div className="matrix-header">Predicted Yes</div>
              
              {/* Row headers and cells */}
              <div className="matrix-header">Actual No</div>
              <div className="matrix-cell cell-tn">
                <span className="cell-value">{tn}</span>
                <span className="cell-label">True Negative</span>
              </div>
              <div className="matrix-cell cell-fp">
                <span className="cell-value">{fp}</span>
                <span className="cell-label">False Positive</span>
              </div>
              
              <div className="matrix-header">Actual Yes</div>
              <div className="matrix-cell cell-fn">
                <span className="cell-value">{fn}</span>
                <span className="cell-label">False Negative</span>
              </div>
              <div className="matrix-cell cell-tp">
                <span className="cell-value">{tp}</span>
                <span className="cell-label">True Positive</span>
              </div>
            </div>

            {/* Matrix Statistics */}
            <div className="matrix-stats">
              <div className="stat-item">
                <div className="stat-value">{((tp + tn) / (tp + tn + fp + fn) * 100).toFixed(2)}%</div>
                <div className="stat-label">Overall Accuracy</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{((tp) / (tp + fp) * 100).toFixed(2)}%</div>
                <div className="stat-label">Precision</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{((tp) / (tp + fn) * 100).toFixed(2)}%</div>
                <div className="stat-label">Recall</div>
              </div>
              <div className="stat-item">
                <div className="stat-value">{((2 * tp) / (2 * tp + fp + fn) * 100).toFixed(2)}%</div>
                <div className="stat-label">F1-Score</div>
              </div>
            </div>

            {/* Legend */}
            <div className="matrix-legend">
              <div className="legend-item">
                <div className="legend-color legend-tp"></div>
                <span>True Positive (TP)</span>
              </div>
              <div className="legend-item">
                <div className="legend-color legend-fp"></div>
                <span>False Positive (FP)</span>
              </div>
              <div className="legend-item">
                <div className="legend-color legend-fn"></div>
                <span>False Negative (FN)</span>
              </div>
              <div className="legend-item">
                <div className="legend-color legend-tn"></div>
                <span>True Negative (TN)</span>
              </div>
            </div>
          </div>
        </div>
      </div>
    </div>
  );
}