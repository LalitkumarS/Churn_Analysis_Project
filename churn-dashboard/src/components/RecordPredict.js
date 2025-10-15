import React, { useState } from "react";
import "./RecordPredict.css";
import axios from "axios";

export default function RecordPredict() {
  const [inputData, setInputData] = useState({
    tenure: "",
    MonthlyCharges: "",
    TotalCharges: "",
  });
  const [result, setResult] = useState(null);
  const [loading, setLoading] = useState(false);

  const handleChange = (e) => {
    setInputData({ ...inputData, [e.target.name]: e.target.value });
  };

  const handleSubmit = async () => {
    if (!inputData.tenure || !inputData.MonthlyCharges || !inputData.TotalCharges) {
      alert("Please fill in all fields before prediction!");
      return;
    }
    try {
      setLoading(true);
      const res = await axios.post("http://localhost:5000/api/predict", inputData);
      setResult(res.data);
    } catch (err) {
      console.error("Prediction error:", err);
      setResult({ error: "Server error! Please try again later." });
    } finally {
      setLoading(false);
    }
  };
  const matchedRecord=JSON.stringify(result?.matched_row, null, 2);

  const badge = (val) => {
    if (val === null || val === undefined) return null;
    const isYes = String(val).toLowerCase() === "yes" || Number(val) === 1;
    return (
      <span
        className={`RecordPredict-badge inline-block px-3 py-1 rounded-full text-white font-semibold ${
          isYes ? "bg-red-600" : "bg-green-600"
        }`}
      >
        {isYes ? "YES" : "NO"}
      </span>
    );
  };

  return (
    <div className="RecordPredict-card max-w-lg mx-auto bg-gradient-to-br from-purple-100 via-white to-purple-50 p-6 rounded-2xl shadow-2xl space-y-4 mt-10 transition-all duration-300">
      <h2 className="RecordPredict-header text-3xl font-extrabold text-center">
        🔮 Predict Customer Churn
      </h2>

      {/* Input Fields */}
      {["tenure", "MonthlyCharges", "TotalCharges"].map((col) => (
        <div key={col} className="RecordPredict-form-group">
          <label>{col.toUpperCase()}</label>
          <input
            type="number"
            name={col}
            value={inputData[col]}
            onChange={handleChange}
            className="RecordPredict-input"
            placeholder={`Enter ${col}`}
          />
        </div>
      ))}

      {/* Predict Button */}
      <div className="RecordPredict-btn-container">
        <button
          onClick={handleSubmit}
          disabled={loading}
          className="RecordPredict-btn"
        >
          {loading ? "Predicting..." : "PREDICT"}
        </button>
      </div>

      {/* Result Section */}
      {result && (
        <div className="RecordPredict-result mt-6 p-4 bg-white rounded-lg shadow-inner text-center">
          {result.error && <p className="text-red-600 font-semibold">{result.error}</p>}

          {!result.error && (
            <>
              <h3>Dataset Result</h3>
              <div className="RecordPredict-data-grid">
                <div className="RecordPredict-data-item">
                  <div className="RecordPredict-data-label">Churn (dataset)</div>
                  <div className="RecordPredict-data-value">
                    {badge(result.churn_dataset ?? result.matched_row?.Churn ?? null)}
                  </div>
                </div>

                <div className="RecordPredict-data-item">
                  <div className="RecordPredict-data-label">Match type</div>
                  <div className="RecordPredict-data-value">
                    {result.match_type || "N/A"}
                  </div>
                </div>
              </div>

              <hr className="RecordPredict-hr my-4" />

              <h3>Model Prediction</h3>
              <div className="RecordPredict-data-grid">
                <div className="RecordPredict-data-item">
                  <div className="RecordPredict-data-label">Churn (model)</div>
                  <div className="RecordPredict-data-value">
                    {result.churn_model === null || result.churn_model === undefined
                      ? "N/A"
                      : result.churn_model === 1
                      ? "Yes"
                      : "No"}
                  </div>
                </div>

                <div className="RecordPredict-data-item">
                  <div className="RecordPredict-data-label">Probability</div>
                  <div className="RecordPredict-data-value">
                    {result.probability === null || result.probability === undefined
                      ? "N/A"
                      : (result.probability * 100).toFixed(2) + "%"}
                  </div>
                </div>
              </div>

              {result.matched_row && (
                <>
                  <hr className="RecordPredict-hr my-4" />
                  <h4 className="font-semibold">Matched Record (from dataset)</h4>
                  <pre className="RecordPredict-pre text-left text-sm mt-2 bg-gray-50 p-3 rounded-lg overflow-x-auto">
                    {matchedRecord}
                  </pre>
                </>
              )}
            </>
          )}
        </div>
      )}
    </div>
  );
}