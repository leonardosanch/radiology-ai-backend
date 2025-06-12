# 📋 Guía Frontend para Sistema IA Médica Avanzado

## 🎯 Objetivo: Interface para Análisis Ensemble Multi-Modelo

Transformar el JSON del **Sistema IA Avanzado** en interfaces médicas profesionales que muestren:

- **Análisis Ensemble** vs **Modelo Individual**
- **Consenso entre Modelos** de IA
- **Router Inteligente** y selección automática
- **Recomendaciones Médicas** avanzadas

---

## 🏗️ Arquitectura Frontend Avanzada

### **Estructura de Componentes (React/Liferay)**

```
📄 AdvancedRadiologyReport (Componente Principal)
├── 🧠 AnalysisTypeIndicator (Ensemble vs Individual)
├── 🏥 ReportHeader (Info del estudio + modelos usados)
├── 🎯 ModelSelectionSummary (Router inteligente)
├── 🤝 ConsensusAnalysis (Acuerdo entre modelos)
├── 📊 ExecutiveSummary (Resumen ejecutivo avanzado)
├── 🔍 EnsembleFindingsSection (Hallazgos con consenso)
├── 📋 DetailedMultiModelAnalysis (Análisis por modelo)
├── 🩺 AdvancedMedicalInterpretation (Interpretación IA)
├── 🎯 IntelligentRecommendations (Recomendaciones automáticas)
├── ⚡ PerformanceMetrics (Métricas de rendimiento)
├── ⚠️ SystemLimitations (Limitaciones del sistema)
└── 🏆 EnsembleConclusion (Conclusión del ensemble)
```

---

## 📊 Mapeo JSON → Interface Médica Avanzada

### 🧠 **SECCIÓN 1: INDICADOR DE ANÁLISIS**

**Datos JSON:**

```json
{
  "analysis_type": "intelligent_ensemble | single_model",
  "models_used": ["torax_model", "fracturas_model", "chexnet_model"],
  "confidence": 0.847,
  "processing_time": 2.34
}
```

**Presentación Visual:**

```jsx
// Ejemplo React
function AnalysisTypeIndicator({ analysisType, modelsUsed, confidence }) {
  const isEnsemble = analysisType === "intelligent_ensemble";

  return (
    <div className={`analysis-badge ${isEnsemble ? "ensemble" : "single"}`}>
      <div className="analysis-type">
        {isEnsemble ? (
          <>
            🧠 <strong>Análisis Ensemble</strong>
            <span className="models-count">{modelsUsed.length} Modelos IA</span>
          </>
        ) : (
          <>
            ⚡ <strong>Análisis Individual</strong>
            <span className="single-model">{modelsUsed[0]}</span>
          </>
        )}
      </div>
      <div className="confidence-indicator">
        <span>Confianza: {(confidence * 100).toFixed(1)}%</span>
        <div className="confidence-bar">
          <div
            className="confidence-fill"
            style={{ width: `${confidence * 100}%` }}
          />
        </div>
      </div>
    </div>
  );
}
```

### 🎯 **SECCIÓN 2: RESUMEN DEL ROUTER INTELIGENTE**

**Datos JSON:**

```json
{
  "image_analysis": {
    "type": "chest_xray",
    "study_type": "pa_chest",
    "quality": "excellent"
  },
  "models_used": ["torax_model", "chexnet_model"],
  "model_selection_reason": "Detected chest X-ray - selected thoracic specialists"
}
```

**Presentación Visual:**

```jsx
function ModelSelectionSummary({ imageAnalysis, modelsUsed }) {
  const getModelIcon = (model) => {
    const icons = {
      torax_model: "🫁",
      fracturas_model: "🦴",
      chexnet_model: "🩺",
      radimagenet_model: "🔬",
    };
    return icons[model] || "🤖";
  };

  return (
    <div className="model-selection">
      <h3>🎯 Selección Inteligente de Modelos</h3>
      <div className="detection-info">
        <span>📸 Detectado: {imageAnalysis.type}</span>
        <span>📊 Calidad: {imageAnalysis.quality}</span>
      </div>
      <div className="selected-models">
        <h4>Modelos Especializados Utilizados:</h4>
        <div className="models-grid">
          {modelsUsed.map((model) => (
            <div key={model} className="model-card">
              <span className="model-icon">{getModelIcon(model)}</span>
              <span className="model-name">{model.replace("_", " ")}</span>
              <span className="specialization">{getSpecialization(model)}</span>
            </div>
          ))}
        </div>
      </div>
    </div>
  );
}
```

### 🤝 **SECCIÓN 3: ANÁLISIS DE CONSENSO (Solo para Ensemble)**

**Datos JSON:**

```json
{
  "consensus_analysis": {
    "high_agreement": ["Pneumonia", "Atelectasis"],
    "moderate_agreement": ["Mass"],
    "low_agreement": ["Nodule"],
    "conflicting": [],
    "agreement_scores": {
      "Pneumonia": 0.89,
      "Atelectasis": 0.76
    }
  }
}
```

**Presentación Visual:**

```jsx
function ConsensusAnalysis({ consensus }) {
  if (!consensus) return null; // Solo mostrar para ensemble

  return (
    <div className="consensus-section">
      <h3>🤝 Consenso entre Modelos IA</h3>

      <div className="consensus-grid">
        <div className="consensus-category high">
          <div className="category-header">
            <span className="icon">✅</span>
            <span className="title">Alto Consenso</span>
            <span className="count">{consensus.high_agreement.length}</span>
          </div>
          <div className="pathologies-list">
            {consensus.high_agreement.map((pathology) => (
              <div key={pathology} className="pathology-item high-consensus">
                <span className="name">{pathology}</span>
                <span className="score">
                  {(consensus.agreement_scores[pathology] * 100).toFixed(0)}%
                  acuerdo
                </span>
              </div>
            ))}
          </div>
        </div>

        <div className="consensus-category moderate">
          <div className="category-header">
            <span className="icon">🟡</span>
            <span className="title">Consenso Moderado</span>
            <span className="count">{consensus.moderate_agreement.length}</span>
          </div>
          {/* Similar structure */}
        </div>

        <div className="consensus-category conflicting">
          <div className="category-header">
            <span className="icon">⚠️</span>
            <span className="title">Conflictivos</span>
            <span className="count">{consensus.conflicting.length}</span>
          </div>
          {consensus.conflicting.length > 0 && (
            <div className="conflict-warning">
              <p>
                ⚠️ Los modelos discrepan en estos hallazgos. Se requiere
                revisión médica adicional.
              </p>
            </div>
          )}
        </div>
      </div>
    </div>
  );
}
```

### 📊 **SECCIÓN 4: HALLAZGOS MULTI-MODELO**

**Datos JSON:**

```json
{
  "final_predictions": {
    "Pneumonia": 0.235,
    "Atelectasis": 0.167
  },
  "individual_results": [
    {
      "model_name": "torax_model",
      "predictions": { "Pneumonia": 0.28 },
      "confidence": 0.89
    },
    {
      "model_name": "chexnet_model",
      "predictions": { "Pneumonia": 0.19 },
      "confidence": 0.76
    }
  ]
}
```

**Presentación Visual:**

```jsx
function EnsembleFindingsTable({ finalPredictions, individualResults }) {
  return (
    <div className="ensemble-findings">
      <h3>📊 Hallazgos con Análisis Multi-Modelo</h3>

      <div className="findings-table">
        <table>
          <thead>
            <tr>
              <th>Patología</th>
              <th>Ensemble Final</th>
              <th>Modelos Individuales</th>
              <th>Consenso</th>
              <th>Acción Recomendada</th>
            </tr>
          </thead>
          <tbody>
            {Object.entries(finalPredictions).map(([pathology, finalScore]) => (
              <tr key={pathology} className={getConfidenceClass(finalScore)}>
                <td className="pathology-name">
                  <strong>{pathology}</strong>
                </td>
                <td className="final-score">
                  <span className="percentage">
                    {(finalScore * 100).toFixed(1)}%
                  </span>
                  <div className="confidence-bar">
                    <div
                      className="fill"
                      style={{ width: `${finalScore * 100}%` }}
                    />
                  </div>
                </td>
                <td className="individual-scores">
                  {getIndividualScores(pathology, individualResults).map(
                    (result) => (
                      <div key={result.model} className="model-score">
                        <span className="model">{result.model}</span>
                        <span className="score">
                          {(result.score * 100).toFixed(1)}%
                        </span>
                      </div>
                    )
                  )}
                </td>
                <td className="consensus-indicator">
                  {getConsensusIndicator(pathology, individualResults)}
                </td>
                <td className="recommendation">
                  {getRecommendation(finalScore)}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
    </div>
  );
}
```

### 🎯 **SECCIÓN 5: RECOMENDACIONES INTELIGENTES**

**Datos JSON:**

```json
{
  "medical_recommendation": {
    "urgency_level": "priority",
    "primary_recommendation": "Evaluación médica prioritaria",
    "immediate_actions": ["Revisión por radiólogo certificado requerida"],
    "follow_up_actions": ["Seguimiento clínico recomendado"],
    "specialist_referral": true
  }
}
```

**Presentación Visual:**

```jsx
function IntelligentRecommendations({ recommendation }) {
  const getUrgencyConfig = (level) => {
    const configs = {
      routine: { icon: "✅", color: "green", label: "Rutinario" },
      priority: { icon: "🟡", color: "orange", label: "Prioritario" },
      urgent: { icon: "🚨", color: "red", label: "Urgente" },
    };
    return configs[level] || configs.routine;
  };

  const urgencyConfig = getUrgencyConfig(recommendation.urgency_level);

  return (
    <div className="intelligent-recommendations">
      <div className="urgency-header">
        <span className={`urgency-badge ${urgencyConfig.color}`}>
          {urgencyConfig.icon} {urgencyConfig.label}
        </span>
        <h3>🎯 Recomendaciones del Sistema IA</h3>
      </div>

      <div className="primary-recommendation">
        <h4>📋 Recomendación Principal</h4>
        <p className="main-rec">{recommendation.primary_recommendation}</p>
      </div>

      {recommendation.immediate_actions?.length > 0 && (
        <div className="immediate-actions">
          <h4>🚨 Acciones Inmediatas</h4>
          <ul className="action-list immediate">
            {recommendation.immediate_actions.map((action, index) => (
              <li key={index} className="action-item">
                <span className="bullet">•</span>
                <span className="text">{action}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {recommendation.follow_up_actions?.length > 0 && (
        <div className="follow-up-actions">
          <h4>📅 Seguimiento Recomendado</h4>
          <ul className="action-list followup">
            {recommendation.follow_up_actions.map((action, index) => (
              <li key={index} className="action-item">
                <span className="bullet">•</span>
                <span className="text">{action}</span>
              </li>
            ))}
          </ul>
        </div>
      )}

      {recommendation.specialist_referral && (
        <div className="specialist-referral">
          <div className="referral-badge">
            👨‍⚕️ <strong>Referencia a Especialista Recomendada</strong>
          </div>
        </div>
      )}
    </div>
  );
}
```

### ⚡ **SECCIÓN 6: MÉTRICAS DE RENDIMIENTO AVANZADO**

**Datos JSON:**

```json
{
  "performance_metrics": {
    "total_processing_time_seconds": 2.34,
    "individual_model_times": {
      "torax_model": 0.45,
      "chexnet_model": 0.38
    },
    "ensemble_combination_time": 0.12,
    "consensus_analysis_time": 0.08
  }
}
```

**Presentación Visual:**

```jsx
function PerformanceMetrics({ metrics, analysisType }) {
  return (
    <div className="performance-section">
      <h3>⚡ Métricas de Rendimiento</h3>

      <div className="metrics-grid">
        <div className="metric-card primary">
          <span className="value">
            {metrics.total_processing_time_seconds.toFixed(2)}s
          </span>
          <span className="label">Tiempo Total</span>
        </div>

        {analysisType === "intelligent_ensemble" && (
          <>
            <div className="metric-card">
              <span className="value">
                {Object.keys(metrics.individual_model_times).length}
              </span>
              <span className="label">Modelos Ejecutados</span>
            </div>

            <div className="metric-card">
              <span className="value">
                {metrics.ensemble_combination_time.toFixed(3)}s
              </span>
              <span className="label">Tiempo Ensemble</span>
            </div>

            <div className="metric-card">
              <span className="value">
                {metrics.consensus_analysis_time.toFixed(3)}s
              </span>
              <span className="label">Análisis Consenso</span>
            </div>
          </>
        )}
      </div>

      {analysisType === "intelligent_ensemble" && (
        <div className="model-breakdown">
          <h4>Tiempo por Modelo Individual:</h4>
          <div className="model-times">
            {Object.entries(metrics.individual_model_times).map(
              ([model, time]) => (
                <div key={model} className="model-time">
                  <span className="model-name">{model.replace("_", " ")}</span>
                  <span className="time">{time.toFixed(3)}s</span>
                  <div className="time-bar">
                    <div
                      className="time-fill"
                      style={{
                        width: `${
                          (time / metrics.total_processing_time_seconds) * 100
                        }%`,
                      }}
                    />
                  </div>
                </div>
              )
            )}
          </div>
        </div>
      )}
    </div>
  );
}
```

---

## 🎨 Estilos CSS Avanzados

### **Paleta de Colores del Sistema IA**

```css
:root {
  /* Análisis Types */
  --ensemble-primary: #667eea;
  --ensemble-secondary: #764ba2;
  --single-model: #2196f3;

  /* Consenso Colors */
  --high-consensus: #4caf50;
  --moderate-consensus: #ff9800;
  --low-consensus: #2196f3;
  --conflicting: #f44336;

  /* Urgency Levels */
  --routine: #4caf50;
  --priority: #ff9800;
  --urgent: #f44336;

  /* Model Colors */
  --torax-model: #e91e63;
  --fracturas-model: #9c27b0;
  --chexnet-model: #3f51b5;
  --radimagenet-model: #00bcd4;
}
```

### **Componentes Específicos**

```css
/* Analysis Type Badge */
.analysis-badge {
  display: flex;
  align-items: center;
  padding: 15px 20px;
  border-radius: 12px;
  margin-bottom: 20px;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.analysis-badge.ensemble {
  background: linear-gradient(
    135deg,
    var(--ensemble-primary),
    var(--ensemble-secondary)
  );
  color: white;
}

.analysis-badge.single {
  background: linear-gradient(135deg, var(--single-model), #42a5f5);
  color: white;
}

/* Consensus Analysis */
.consensus-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
  gap: 20px;
  margin-top: 20px;
}

.consensus-category {
  background: white;
  border-radius: 8px;
  padding: 20px;
  box-shadow: 0 2px 8px rgba(0, 0, 0, 0.1);
}

.consensus-category.high {
  border-left: 5px solid var(--high-consensus);
}

.consensus-category.moderate {
  border-left: 5px solid var(--moderate-consensus);
}

.consensus-category.conflicting {
  border-left: 5px solid var(--conflicting);
}

/* Ensemble Findings Table */
.findings-table {
  overflow-x: auto;
  margin-top: 20px;
}

.findings-table table {
  width: 100%;
  border-collapse: collapse;
  background: white;
  border-radius: 8px;
  overflow: hidden;
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1);
}

.findings-table th {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
  padding: 15px;
  text-align: left;
  font-weight: 600;
}

.findings-table td {
  padding: 12px 15px;
  border-bottom: 1px solid #eee;
}

.findings-table tr.high-confidence {
  background: rgba(244, 67, 54, 0.05);
  border-left: 4px solid var(--urgent);
}

.findings-table tr.moderate-confidence {
  background: rgba(255, 152, 0, 0.05);
  border-left: 4px solid var(--priority);
}

/* Model Cards */
.models-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
  gap: 15px;
  margin-top: 15px;
}

.model-card {
  background: white;
  padding: 15px;
  border-radius: 8px;
  text-align: center;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
  transition: transform 0.2s;
}

.model-card:hover {
  transform: translateY(-2px);
  box-shadow: 0 4px 12px rgba(0, 0, 0, 0.15);
}

/* Performance Metrics */
.metrics-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
  gap: 15px;
  margin-top: 20px;
}

.metric-card {
  background: white;
  padding: 20px;
  border-radius: 8px;
  text-align: center;
  box-shadow: 0 2px 6px rgba(0, 0, 0, 0.1);
}

.metric-card.primary {
  background: linear-gradient(135deg, #667eea, #764ba2);
  color: white;
}

.metric-card .value {
  display: block;
  font-size: 2em;
  font-weight: bold;
  margin-bottom: 5px;
}

.metric-card .label {
  font-size: 0.9em;
  opacity: 0.8;
}
```

---

## 📱 Componente Principal Integrado

```jsx
// AdvancedRadiologyReport.jsx
import React, { useState, useEffect } from "react";

function AdvancedRadiologyReport({ analysisData }) {
  const [activeTab, setActiveTab] = useState("summary");
  const [expandedSections, setExpandedSections] = useState({});

  const isEnsemble = analysisData.analysis_type === "intelligent_ensemble";

  return (
    <div className="advanced-radiology-report">
      {/* Analysis Type Indicator */}
      <AnalysisTypeIndicator
        analysisType={analysisData.analysis_type}
        modelsUsed={analysisData.models_used || [analysisData.model_used]}
        confidence={analysisData.confidence}
        processingTime={analysisData.processing_time}
      />

      {/* Navigation Tabs */}
      <div className="report-navigation">
        <button
          className={activeTab === "summary" ? "active" : ""}
          onClick={() => setActiveTab("summary")}
        >
          📊 Resumen Ejecutivo
        </button>

        {isEnsemble && (
          <button
            className={activeTab === "consensus" ? "active" : ""}
            onClick={() => setActiveTab("consensus")}
          >
            🤝 Análisis de Consenso
          </button>
        )}

        <button
          className={activeTab === "findings" ? "active" : ""}
          onClick={() => setActiveTab("findings")}
        >
          🔍 Hallazgos Detallados
        </button>

        <button
          className={activeTab === "recommendations" ? "active" : ""}
          onClick={() => setActiveTab("recommendations")}
        >
          📝 Recomendaciones
        </button>
      </div>

      {/* Tab Content */}
      <div className="report-content">
        {activeTab === "summary" && (
          <div className="tab-pane">
            <ModelSelectionSummary
              imageAnalysis={analysisData.image_analysis}
              modelsUsed={analysisData.models_used || [analysisData.model_used]}
            />
            <ExecutiveSummary
              medicalAnalysis={analysisData.medical_analysis}
              isEnsemble={isEnsemble}
            />
          </div>
        )}

        {activeTab === "consensus" && isEnsemble && (
          <div className="tab-pane">
            <ConsensusAnalysis consensus={analysisData.consensus_analysis} />
          </div>
        )}

        {activeTab === "findings" && (
          <div className="tab-pane">
            {isEnsemble ? (
              <EnsembleFindingsTable
                finalPredictions={analysisData.final_predictions}
                individualResults={analysisData.individual_results}
              />
            ) : (
              <SingleModelFindings
                predictions={analysisData.predictions}
                modelUsed={analysisData.model_used}
              />
            )}
          </div>
        )}

        {activeTab === "recommendations" && (
          <div className="tab-pane">
            <IntelligentRecommendations
              recommendation={analysisData.medical_recommendation}
            />
          </div>
        )}
      </div>

      {/* Performance Metrics Footer */}
      <PerformanceMetrics
        metrics={analysisData.performance_metrics}
        analysisType={analysisData.analysis_type}
      />

      {/* Medical Disclaimer */}
      <div className="medical-disclaimer">
        <h4>⚠️ Importante - Sistema de Apoyo Diagnóstico</h4>
        <p>
          Este análisis ha sido generado por un sistema de inteligencia
          artificial
          {isEnsemble
            ? ` usando ${analysisData.models_used.length} modelos especializados`
            : " con modelo único"}. Los resultados requieren <strong>
            validación por profesional médico certificado
          </strong>
          antes de tomar decisiones clínicas.
        </p>
        {isEnsemble && (
          <p>
            El consenso entre múltiples modelos proporciona mayor confianza,
            pero no reemplaza el juicio clínico profesional.
          </p>
        )}
      </div>
    </div>
  );
}

export default AdvancedRadiologyReport;
```

---

## 🔄 Estados y Interacciones Avanzadas

### **Toggle Ensemble vs Individual**

```jsx
function AnalysisToggle({ onToggle, currentMode, isProcessing }) {
  return (
    <div className="analysis-toggle">
      <div className="toggle-header">
        <h3>🎯 Seleccionar Tipo de Análisis</h3>
        <p>
          Elige entre análisis ensemble (más preciso) o individual (más rápido)
        </p>
      </div>

      <div className="toggle-options">
        <button
          className={`toggle-option ${
            currentMode === "ensemble" ? "active" : ""
          }`}
          onClick={() => onToggle("ensemble")}
          disabled={isProcessing}
        >
          <div className="option-icon">🧠</div>
          <div className="option-content">
            <h4>Análisis Ensemble</h4>
            <p>Múltiples modelos + consenso</p>
            <span className="time-estimate">~2-4 segundos</span>
          </div>
        </button>

        <button
          className={`toggle-option ${
            currentMode === "single" ? "active" : ""
          }`}
          onClick={() => onToggle("single")}
          disabled={isProcessing}
        >
          <div className="option-icon">⚡</div>
          <div className="option-content">
            <h4>Análisis Individual</h4>
            <p>Modelo único especializado</p>
            <span className="time-estimate">~0.5 segundos</span>
          </div>
        </button>
      </div>
    </div>
  );
}
```

### **Loading States Avanzados**

```jsx
function AdvancedLoadingState({ analysisType, currentStep, modelsUsed }) {
  const steps =
    analysisType === "ensemble"
      ? [
          "Validando imagen",
          "Analizando con IA",
          "Generando consenso",
          "Creando reporte",
        ]
      : ["Validando imagen", "Analizando con IA", "Generando reporte"];

  return (
    <div className="advanced-loading">
      <div className="loading-header">
        <h3>
          {analysisType === "ensemble"
            ? "🧠 Ejecutando Análisis Ensemble"
            : "⚡ Ejecutando Análisis Individual"}
        </h3>
        <p>
          {analysisType === "ensemble"
            ? `Utilizando ${modelsUsed.length} modelos especializados`
            : "Análisis rápido con modelo especializado"}
        </p>
      </div>

      <div className="loading-progress">
        <div className="steps-indicator">
          {steps.map((step, index) => (
            <div
              key={index}
              className={`step ${index <= currentStep ? "completed" : ""} ${
                index === currentStep ? "active" : ""
              }`}
            >
              <div className="step-number">{index + 1}</div>
              <div className="step-label">{step}</div>
            </div>
          ))}
        </div>

        <div className="progress-bar">
          <div
            className="progress-fill"
            style={{ width: `${((currentStep + 1) / steps.length) * 100}%` }}
          />
        </div>
      </div>

      {analysisType === "ensemble" && (
        <div className="models-status">
          <h4>Estado de Modelos:</h4>
          <div className="models-list">
            {modelsUsed.map((model, index) => (
              <div
                key={model}
                className={`model-status ${
                  index <= currentStep ? "completed" : "pending"
                }`}
              >
                <span className="model-icon">{getModelIcon(model)}</span>
                <span className="model-name">{model.replace("_", " ")}</span>
                <span className="status-indicator">
                  {index <= currentStep ? "✅" : "⏳"}
                </span>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
```

---

## 🚀 Integración con Liferay

### **Portlet Principal**

```jsx
// RadiologyAIPortlet.jsx para Liferay
import { AdvancedRadiologyAIClient } from "./api/AdvancedRadiologyAIClient";

function RadiologyAIPortlet() {
  const [client] = useState(() => new AdvancedRadiologyAIClient());
  const [systemStatus, setSystemStatus] = useState(null);
  const [analysisMode, setAnalysisMode] = useState("ensemble");
  const [currentAnalysis, setCurrentAnalysis] = useState(null);
  const [isProcessing, setIsProcessing] = useState(false);
  const [uploadedFile, setUploadedFile] = useState(null);

  useEffect(() => {
    initializeSystem();
  }, []);

  const initializeSystem = async () => {
    const initialized = await client.initialize();
    if (initialized) {
      const status = await client.getSystemStatus();
      setSystemStatus(status);
    }
  };

  const handleFileUpload = async (file) => {
    setUploadedFile(file);
    setIsProcessing(true);

    try {
      const result = await client.analyzeWithEnsemble(file, {
        useEnsemble: analysisMode === "ensemble",
        includeConsensus: true,
      });

      setCurrentAnalysis(result);
    } catch (error) {
      console.error("Error en análisis:", error);
    } finally {
      setIsProcessing(false);
    }
  };

  return (
    <div className="radiology-ai-portlet">
      {/* System Status Header */}
      {systemStatus && (
        <div className="system-status-header">
          <div className="status-info">
            <h2>🧠 Sistema IA Médica Avanzado</h2>
            <div className="status-badges">
              <span
                className={`status-badge ${
                  systemStatus.healthy ? "healthy" : "error"
                }`}
              >
                {systemStatus.healthy ? "✅ Operacional" : "❌ Error"}
              </span>
              <span className="models-badge">
                🤖 {systemStatus.loadedModels}/{systemStatus.totalModels}{" "}
                Modelos
              </span>
              <span className="capabilities-badge">
                🎯 {Object.keys(systemStatus.capabilities).length} Capacidades
              </span>
            </div>
          </div>

          <div className="active-models">
            <h4>Modelos Activos:</h4>
            <div className="models-chips">
              {systemStatus.activeModels.map((model) => (
                <span key={model} className="model-chip">
                  {getModelIcon(model)} {model.replace("_", " ")}
                </span>
              ))}
            </div>
          </div>
        </div>
      )}

      {/* Analysis Mode Toggle */}
      <AnalysisToggle
        currentMode={analysisMode}
        onToggle={setAnalysisMode}
        isProcessing={isProcessing}
      />

      {/* File Upload Area */}
      <div className="upload-section">
        <FileUploadArea
          onFileSelect={handleFileUpload}
          acceptedFormats={[
            ".jpg",
            ".jpeg",
            ".png",
            ".dcm",
            ".dicom",
            ".tiff",
            ".bmp",
          ]}
          maxSize={50} // MB
          disabled={isProcessing}
        />
      </div>

      {/* Processing State */}
      {isProcessing && (
        <AdvancedLoadingState
          analysisType={analysisMode}
          modelsUsed={systemStatus?.activeModels || []}
          currentStep={1}
        />
      )}

      {/* Analysis Results */}
      {currentAnalysis && (
        <AdvancedRadiologyReport analysisData={currentAnalysis} />
      )}
    </div>
  );
}
```

### **Configuración específica para Liferay**

```javascript
// liferay-portlet-config.js
Liferay.Portlet.ready("/radiology-ai-portlet", function (portletId, node) {
  // Configuración específica para Liferay
  const portletConfig = {
    apiBaseUrl: window.location.origin + "/api/v1",
    liferayUserId: Liferay.ThemeDisplay.getUserId(),
    companyId: Liferay.ThemeDisplay.getCompanyId(),
    scopeGroupId: Liferay.ThemeDisplay.getScopeGroupId(),
  };

  // Inicializar portlet con configuración de Liferay
  ReactDOM.render(
    React.createElement(RadiologyAIPortlet, {
      config: portletConfig,
      permissions: {
        canAnalyze: Liferay.ThemeDisplay.getPermissionChecker().hasPermission(
          "RADIOLOGY_AI",
          "ANALYZE"
        ),
        canViewReports:
          Liferay.ThemeDisplay.getPermissionChecker().hasPermission(
            "RADIOLOGY_AI",
            "VIEW_REPORTS"
          ),
      },
    }),
    node.one(".radiology-ai-portlet-container").getDOMNode()
  );
});
```

---

## 📊 Componentes de Visualización Avanzada

### **Gráfico de Consenso**

```jsx
// ConsensusChart.jsx
import { Chart as ChartJS, ArcElement, Tooltip, Legend } from "chart.js";
import { Doughnut } from "react-chartjs-2";

ChartJS.register(ArcElement, Tooltip, Legend);

function ConsensusChart({ consensus }) {
  const data = {
    labels: [
      "Alto Consenso",
      "Consenso Moderado",
      "Bajo Consenso",
      "Conflictivos",
    ],
    datasets: [
      {
        data: [
          consensus.high_agreement.length,
          consensus.moderate_agreement.length,
          consensus.low_agreement.length,
          consensus.conflicting.length,
        ],
        backgroundColor: [
          "#4caf50", // Verde - Alto consenso
          "#ff9800", // Naranja - Moderado
          "#2196f3", // Azul - Bajo
          "#f44336", // Rojo - Conflictivos
        ],
        borderWidth: 2,
        borderColor: "#fff",
      },
    ],
  };

  const options = {
    responsive: true,
    plugins: {
      legend: {
        position: "bottom",
        labels: {
          padding: 20,
          usePointStyle: true,
        },
      },
      tooltip: {
        callbacks: {
          label: function (context) {
            const label = context.label || "";
            const value = context.parsed;
            const total = context.dataset.data.reduce((a, b) => a + b, 0);
            const percentage = ((value / total) * 100).toFixed(1);
            return `${label}: ${value} hallazgos (${percentage}%)`;
          },
        },
      },
    },
  };

  return (
    <div className="consensus-chart">
      <h4>📊 Distribución de Consenso</h4>
      <div className="chart-container">
        <Doughnut data={data} options={options} />
      </div>
      <div className="chart-summary">
        <p>
          <strong>Interpretación:</strong>{" "}
          {consensus.high_agreement.length > 0
            ? `${consensus.high_agreement.length} hallazgos con alto acuerdo entre modelos`
            : "No hay hallazgos con alto consenso"}
        </p>
      </div>
    </div>
  );
}
```

### **Timeline de Procesamiento**

```jsx
// ProcessingTimeline.jsx
function ProcessingTimeline({ metrics, analysisType }) {
  const getTimelineSteps = () => {
    if (analysisType === "intelligent_ensemble") {
      return [
        { name: "Validación", time: 0.02, color: "#2196f3" },
        {
          name: "Procesamiento Imagen",
          time: metrics.image_processing_time_seconds || 0.5,
          color: "#ff9800",
        },
        ...Object.entries(metrics.individual_model_times || {}).map(
          ([model, time]) => ({
            name: model.replace("_", " "),
            time: time,
            color: getModelColor(model),
          })
        ),
        {
          name: "Ensemble",
          time: metrics.ensemble_combination_time || 0.1,
          color: "#4caf50",
        },
        {
          name: "Consenso",
          time: metrics.consensus_analysis_time || 0.08,
          color: "#9c27b0",
        },
        { name: "Reporte", time: 0.05, color: "#607d8b" },
      ];
    } else {
      return [
        { name: "Validación", time: 0.02, color: "#2196f3" },
        {
          name: "Procesamiento",
          time: metrics.image_processing_time_seconds || 0.4,
          color: "#ff9800",
        },
        {
          name: "Análisis IA",
          time: metrics.ai_inference_time_seconds || 0.05,
          color: "#f44336",
        },
        { name: "Reporte", time: 0.03, color: "#607d8b" },
      ];
    }
  };

  const steps = getTimelineSteps();
  const totalTime = steps.reduce((sum, step) => sum + step.time, 0);

  return (
    <div className="processing-timeline">
      <h4>⏱️ Timeline de Procesamiento</h4>

      <div className="timeline-bar">
        {steps.map((step, index) => (
          <div
            key={index}
            className="timeline-segment"
            style={{
              width: `${(step.time / totalTime) * 100}%`,
              backgroundColor: step.color,
            }}
            title={`${step.name}: ${step.time.toFixed(3)}s`}
          />
        ))}
      </div>

      <div className="timeline-labels">
        {steps.map((step, index) => (
          <div key={index} className="timeline-label">
            <div
              className="label-indicator"
              style={{ backgroundColor: step.color }}
            />
            <span className="label-name">{step.name}</span>
            <span className="label-time">{step.time.toFixed(3)}s</span>
          </div>
        ))}
      </div>

      <div className="timeline-summary">
        <strong>Tiempo Total: {totalTime.toFixed(3)}s</strong>
        {analysisType === "intelligent_ensemble" && (
          <span className="ensemble-note">
            (Overhead ensemble:{" "}
            {(
              ((totalTime - steps[1].time - steps[2].time) / totalTime) *
              100
            ).toFixed(1)}
            %)
          </span>
        )}
      </div>
    </div>
  );
}
```

### **Comparador Lado a Lado**

```jsx
// EnsembleComparison.jsx
function EnsembleComparison({ ensembleResult, singleResult }) {
  return (
    <div className="ensemble-comparison">
      <h3>🔄 Comparación: Ensemble vs Individual</h3>

      <div className="comparison-grid">
        {/* Ensemble Column */}
        <div className="comparison-column ensemble">
          <div className="column-header">
            <h4>🧠 Análisis Ensemble</h4>
            <span className="models-count">
              {ensembleResult.models_used.length} Modelos
            </span>
          </div>

          <div className="comparison-metrics">
            <div className="metric">
              <span className="label">Tiempo:</span>
              <span className="value">
                {ensembleResult.processing_time.toFixed(2)}s
              </span>
            </div>
            <div className="metric">
              <span className="label">Confianza:</span>
              <span className="value">
                {(ensembleResult.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="metric">
              <span className="label">Consenso:</span>
              <span className="value">
                {ensembleResult.consensus_analysis.high_agreement.length}{" "}
                acuerdos
              </span>
            </div>
          </div>

          <div className="top-findings">
            <h5>🔍 Principales Hallazgos:</h5>
            {Object.entries(ensembleResult.final_predictions)
              .sort(([, a], [, b]) => b - a)
              .slice(0, 3)
              .map(([pathology, confidence]) => (
                <div key={pathology} className="finding-item">
                  <span className="pathology">{pathology}</span>
                  <span className="confidence">
                    {(confidence * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
          </div>
        </div>

        {/* Individual Column */}
        <div className="comparison-column individual">
          <div className="column-header">
            <h4>⚡ Análisis Individual</h4>
            <span className="model-name">{singleResult.model_used}</span>
          </div>

          <div className="comparison-metrics">
            <div className="metric">
              <span className="label">Tiempo:</span>
              <span className="value">
                {singleResult.inference_time.toFixed(2)}s
              </span>
            </div>
            <div className="metric">
              <span className="label">Confianza:</span>
              <span className="value">
                {(singleResult.confidence * 100).toFixed(1)}%
              </span>
            </div>
            <div className="metric">
              <span className="label">Modelo:</span>
              <span className="value">Único</span>
            </div>
          </div>

          <div className="top-findings">
            <h5>🔍 Principales Hallazgos:</h5>
            {Object.entries(singleResult.predictions)
              .sort(([, a], [, b]) => b - a)
              .slice(0, 3)
              .map(([pathology, confidence]) => (
                <div key={pathology} className="finding-item">
                  <span className="pathology">{pathology}</span>
                  <span className="confidence">
                    {(confidence * 100).toFixed(1)}%
                  </span>
                </div>
              ))}
          </div>
        </div>
      </div>

      {/* Comparison Summary */}
      <div className="comparison-summary">
        <h4>📊 Resumen Comparativo</h4>
        <div className="summary-points">
          <div className="summary-point">
            <span className="icon">⚡</span>
            <span className="text">
              Individual es{" "}
              <strong>
                {(
                  ensembleResult.processing_time / singleResult.inference_time
                ).toFixed(1)}
                x más rápido
              </strong>
            </span>
          </div>
          <div className="summary-point">
            <span className="icon">🎯</span>
            <span className="text">
              Ensemble proporciona <strong>validación cruzada</strong> entre{" "}
              {ensembleResult.models_used.length} modelos
            </span>
          </div>
          <div className="summary-point">
            <span className="icon">🤝</span>
            <span className="text">
              <strong>
                {ensembleResult.consensus_analysis.high_agreement.length}{" "}
                hallazgos
              </strong>{" "}
              con alto consenso
            </span>
          </div>
        </div>
      </div>
    </div>
  );
}
```

---

## 🎯 Utilidades y Helpers

### **Funciones de Mapeo**

```javascript
// utils/modelUtils.js
export const getModelIcon = (modelName) => {
  const icons = {
    torax_model: "🫁",
    fracturas_model: "🦴",
    chexnet_model: "🩺",
    radimagenet_model: "🔬",
  };
  return icons[modelName] || "🤖";
};

export const getModelColor = (modelName) => {
  const colors = {
    torax_model: "#e91e63",
    fracturas_model: "#9c27b0",
    chexnet_model: "#3f51b5",
    radimagenet_model: "#00bcd4",
  };
  return colors[modelName] || "#607d8b";
};

export const getSpecialization = (modelName) => {
  const specializations = {
    torax_model: "Patologías Torácicas",
    fracturas_model: "Detección de Fracturas",
    chexnet_model: "Especialista en Neumonía",
    radimagenet_model: "Análisis Universal",
  };
  return specializations[modelName] || "Modelo Especializado";
};

export const getConfidenceClass = (confidence) => {
  if (confidence >= 0.7) return "high-confidence";
  if (confidence >= 0.3) return "moderate-confidence";
  return "low-confidence";
};
```

### **Formatters y Transformadores**

```javascript
// utils/formatters.js
export const formatConfidence = (confidence) => {
  return `${(confidence * 100).toFixed(1)}%`;
};

export const formatTime = (seconds) => {
  if (seconds < 1) {
    return `${(seconds * 1000).toFixed(0)}ms`;
  }
  return `${seconds.toFixed(2)}s`;
};

export const getUrgencyConfig = (urgencyLevel) => {
  const configs = {
    routine: {
      icon: "✅",
      color: "green",
      label: "Rutinario",
      description: "Seguimiento normal recomendado",
    },
    priority: {
      icon: "🟡",
      color: "orange",
      label: "Prioritario",
      description: "Requiere atención médica en 24-48h",
    },
    urgent: {
      icon: "🚨",
      color: "red",
      label: "Urgente",
      description: "Requiere atención médica inmediata",
    },
  };
  return configs[urgencyLevel] || configs.routine;
};

export const transformAnalysisData = (rawData) => {
  // Transformar datos del JSON para compatibilidad con componentes
  return {
    ...rawData,
    isEnsemble: rawData.analysis_type === "intelligent_ensemble",
    modelsUsed: rawData.models_used || [rawData.model_used],
    confidencePercentage: formatConfidence(rawData.confidence),
    formattedTime: formatTime(rawData.processing_time),
  };
};
```

---

## 📱 Responsive Design Avanzado

### **Breakpoints Médicos**

```css
/* Responsive breakpoints específicos para uso médico */
:root {
  --mobile-medical: 480px; /* Smartphones médicos */
  --tablet-medical: 768px; /* Tablets hospitalarios */
  --desktop-medical: 1024px; /* Workstations médicas */
  --large-medical: 1440px; /* Monitores radiológicos */
}

/* Mobile First - Uso en urgencias */
@media screen and (max-width: 480px) {
  .advanced-radiology-report {
    padding: 10px;
    font-size: 14px;
  }

  .analysis-badge {
    flex-direction: column;
    text-align: center;
    padding: 15px;
  }

  .consensus-grid {
    grid-template-columns: 1fr;
    gap: 15px;
  }

  .findings-table {
    font-size: 12px;
  }

  .findings-table th,
  .findings-table td {
    padding: 8px 4px;
  }

  /* Ocultar columnas menos importantes en móvil */
  .findings-table .individual-scores,
  .findings-table .consensus-indicator {
    display: none;
  }

  .models-grid {
    grid-template-columns: 1fr;
  }

  .metrics-grid {
    grid-template-columns: repeat(2, 1fr);
  }
}

/* Tablet - Uso en salas médicas */
@media screen and (min-width: 481px) and (max-width: 768px) {
  .consensus-grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .models-grid {
    grid-template-columns: repeat(2, 1fr);
  }

  .comparison-grid {
    grid-template-columns: 1fr;
    gap: 20px;
  }

  .report-navigation {
    flex-wrap: wrap;
    gap: 5px;
  }

  .report-navigation button {
    flex: 1;
    min-width: 120px;
  }
}

/* Desktop - Workstations médicas */
@media screen and (min-width: 769px) and (max-width: 1024px) {
  .advanced-radiology-report {
    max-width: 1000px;
    margin: 0 auto;
  }

  .consensus-grid {
    grid-template-columns: repeat(3, 1fr);
  }

  .findings-table {
    font-size: 14px;
  }
}

/* Large screens - Monitores radiológicos */
@media screen and (min-width: 1025px) {
  .advanced-radiology-report {
    max-width: 1200px;
    margin: 0 auto;
    padding: 30px;
  }

  .consensus-grid {
    grid-template-columns: repeat(4, 1fr);
  }

  .findings-table {
    font-size: 15px;
  }

  .findings-table th,
  .findings-table td {
    padding: 15px;
  }

  /* Mostrar funcionalidades avanzadas en pantallas grandes */
  .advanced-features {
    display: block;
  }

  .detailed-metrics {
    display: grid;
    grid-template-columns: repeat(3, 1fr);
    gap: 20px;
  }
}

/* Print styles para reportes médicos */
@media print {
  .advanced-radiology-report {
    background: white;
    color: black;
    font-size: 12pt;
    line-height: 1.4;
  }

  .report-navigation,
  .interactive-elements,
  .performance-metrics {
    display: none;
  }

  .consensus-grid,
  .models-grid {
    grid-template-columns: 1fr;
    gap: 10px;
  }

  .findings-table {
    border: 1px solid black;
  }

  .findings-table th {
    background: #f0f0f0 !important;
    color: black !important;
  }

  .page-break {
    page-break-before: always;
  }

  .medical-disclaimer {
    font-size: 10pt;
    border: 2px solid black;
    padding: 10px;
    margin-top: 20px;
  }
}
```

---

## 🔧 Testing y Debugging

### **Componente de Debug**

```jsx
// DebugPanel.jsx - Solo para desarrollo
function DebugPanel({ analysisData, systemStatus }) {
  const [showDebug, setShowDebug] = useState(false);

  if (process.env.NODE_ENV !== "development") {
    return null;
  }

  return (
    <div className="debug-panel">
      <button className="debug-toggle" onClick={() => setShowDebug(!showDebug)}>
        🔧 Debug Panel
      </button>

      {showDebug && (
        <div className="debug-content">
          <div className="debug-section">
            <h4>System Status</h4>
            <pre>{JSON.stringify(systemStatus, null, 2)}</pre>
          </div>

          <div className="debug-section">
            <h4>Analysis Data</h4>
            <pre>{JSON.stringify(analysisData, null, 2)}</pre>
          </div>

          <div className="debug-section">
            <h4>Component State</h4>
            <div className="debug-info">
              <p>Analysis Type: {analysisData?.analysis_type}</p>
              <p>Models Used: {analysisData?.models_used?.join(", ")}</p>
              <p>Processing Time: {analysisData?.processing_time}s</p>
              <p>Confidence: {analysisData?.confidence}</p>
            </div>
          </div>
        </div>
      )}
    </div>
  );
}
```

### **Mock Data para Testing**

```javascript
// mockData.js - Datos para testing de componentes
export const mockEnsembleData = {
  analysis_id: "ensemble-test-123",
  analysis_type: "intelligent_ensemble",
  models_used: ["torax_model", "chexnet_model", "fracturas_model"],
  confidence: 0.847,
  processing_time: 2.34,

  final_predictions: {
    Pneumonia: 0.235,
    Atelectasis: 0.167,
    Mass: 0.089,
    Fracture: 0.045,
  },

  individual_results: [
    {
      model_name: "torax_model",
      predictions: { Pneumonia: 0.28, Atelectasis: 0.19 },
      confidence: 0.89,
      inference_time: 0.45,
    },
    {
      model_name: "chexnet_model",
      predictions: { Pneumonia: 0.19, Atelectasis: 0.12 },
      confidence: 0.76,
      inference_time: 0.38,
    },
  ],

  consensus_analysis: {
    high_agreement: ["Pneumonia", "Atelectasis"],
    moderate_agreement: ["Mass"],
    low_agreement: ["Fracture"],
    conflicting: [],
    agreement_scores: {
      Pneumonia: 0.89,
      Atelectasis: 0.76,
      Mass: 0.62,
    },
  },

  medical_recommendation: {
    urgency_level: "priority",
    primary_recommendation: "Evaluación médica prioritaria recomendada",
    immediate_actions: [
      "Revisión por radiólogo certificado requerida",
      "Correlación con historia clínica y examen físico",
    ],
    follow_up_actions: ["Seguimiento clínico recomendado"],
    specialist_referral: true,
  },

  performance_metrics: {
    total_processing_time_seconds: 2.34,
    individual_model_times: {
      torax_model: 0.45,
      chexnet_model: 0.38,
      fracturas_model: 0.42,
    },
    ensemble_combination_time: 0.12,
    consensus_analysis_time: 0.08,
  },
};

export const mockSingleData = {
  analysis_id: "single-test-456",
  analysis_type: "single_model",
  model_used: "torax_model",
  confidence: 0.78,
  inference_time: 0.45,

  predictions: {
    Pneumonia: 0.28,
    Atelectasis: 0.19,
    Mass: 0.12,
    Cardiomegaly: 0.08,
  },
};

export const mockSystemStatus = {
  systemType: "IntelligentMedicalRouter",
  totalModels: 4,
  loadedModels: 4,
  activeModels: [
    "torax_model",
    "fracturas_model",
    "chexnet_model",
    "radimagenet_model",
  ],
  capabilities: {
    intelligent_routing: true,
    ensemble_analysis: true,
    consensus_validation: true,
    medical_recommendations: true,
  },
  healthy: true,
};
```

---

## 🎓 Documentación para Desarrolladores

### **Guía de Implementación Rápida**

````markdown
# 🚀 Quick Start - Sistema IA Avanzado Frontend

## 1. Instalación de Dependencias

```bash
npm install react react-dom
npm install chart.js react-chartjs-2
npm install axios  # Para API calls
```
````

## 2. Estructura de Archivos

```
src/
├── components/
│   ├── AdvancedRadiologyReport.jsx
│   ├── AnalysisTypeIndicator.jsx
│   ├── ConsensusAnalysis.jsx
│   ├── EnsembleFindingsTable.jsx
│   └── IntelligentRecommendations.jsx
├── utils/
│   ├── modelUtils.js
│   ├── formatters.js
│   └── apiClient.js
├── styles/
│   ├── main.css
│   ├── components.css
│   └── responsive.css
└── data/
    └── mockData.js
```

## 3. Uso Básico

```jsx
import AdvancedRadiologyReport from "./components/AdvancedRadiologyReport";

function App() {
  const [analysisData, setAnalysisData] = useState(null);

  return (
    <div>
      {analysisData && <AdvancedRadiologyReport analysisData={analysisData} />}
    </div>
  );
}
```

## 4. Personalización

- Modificar colores en `:root` CSS variables
- Ajustar breakpoints en `responsive.css`
- Personalizar iconos en `modelUtils.js`
- Adaptar textos médicos según institución

````

### **API Reference Rápida**

```javascript
// Endpoints principales para frontend
const API_ENDPOINTS = {
  // Sistema IA
  HEALTH: '/api/v1/analysis/health',
  MODEL_INFO: '/api/v1/analysis/model/info',
  MODELS_STATUS: '/api/v1/ai/models/status',

  // Análisis
  UPLOAD_ENSEMBLE: '/api/v1/analysis/upload',
  UPLOAD_SINGLE: '/api/v1/analysis/upload?use_ensemble=false',
  DEMO_ENSEMBLE: '/api/v1/analysis/demo',
  DEMO_SINGLE: '/api/v1/analysis/demo?use_ensemble=false',

  // Utilidades
  CAPABILITIES: '/api/v1/ai/capabilities',
  STATISTICS: '/api/v1/analysis/statistics'
};

// Ejemplo de uso
const analyzeImage = async (file, useEnsemble = true) => {
  const formData = new
````
