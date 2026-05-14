import Layout from "../components/layout/Layout";

import {
  useEffect,
  useState,
} from "react";

import "../estilos/pages.css";

import {

  getModelsRequest,

  trainModelRequest,

  generateDatasetRequest,

} from "../services/iaLabService";

function LaboratorioEstudiante() {

  // =====================================
  // STATES
  // =====================================

  const [models, setModels] =
    useState([]);

  const [selectedModel, setSelectedModel] =
    useState("");

  const [datasetSize, setDatasetSize] =
    useState(500);

  const [datasetType, setDatasetType] =
    useState("rendimiento");

  const [datasetInfo, setDatasetInfo] =
    useState(null);

  const [sessionId, setSessionId] =
    useState("");

  const [trainingResult, setTrainingResult] =
    useState(null);

  // eslint-disable-next-line no-unused-vars
  const [predictionInputs, setPredictionInputs] =
    useState({});

  const [predictionResult, setPredictionResult] =
    useState(null);

  const [loadingTrain, setLoadingTrain] =
    useState(false);

  const [loadingDataset, setLoadingDataset] =
    useState(false);

  const [error, setError] =
    useState("");

  // =====================================
  // LOAD MODELS
  // =====================================

  useEffect(() => {

    const loadModels = async () => {

      try {

        const response =
          await getModelsRequest();

        setModels(
          response.models
        );

        if (
          response.models.length > 0
        ) {

          setSelectedModel(
            response.models[0].name
          );
        }

      } catch (error) {

        console.error(error);

        setError(
          "Error cargando modelos"
        );
      }
    };

    loadModels();

  }, []);

  // =====================================
  // GENERATE DATASET
  // =====================================

  const handleGenerateDataset =
    async () => {

      try {

        setLoadingDataset(true);

        setError("");

        setTrainingResult(null);

        setPredictionResult(null);

        const response =
          await generateDatasetRequest(

            datasetSize,

            datasetType
          );

        setDatasetInfo(response);

        setSessionId(
          response.session_id
        );

      } catch (error) {

        console.error(error);

        setError(
          "Error generando dataset"
        );

      } finally {

        setLoadingDataset(false);
      }
    };

  // =====================================
  // TRAIN MODEL
  // =====================================

  const handleTrain =
    async () => {

      if (!sessionId) {

        setError(
          "Primero debes generar un dataset"
        );

        return;
      }

      try {

        setLoadingTrain(true);

        setError("");

        const response =
          await trainModelRequest({

            session_id:
              sessionId,

            model_name:
              selectedModel,

            params: {},

            test_size: 0.2,
          });

        setTrainingResult(
          response
        );

      } catch (error) {

        console.error(error);

        setError(
          "Error entrenando modelo"
        );

      } finally {

        setLoadingTrain(false);
      }
    };

  // =====================================
  // INPUTS
  // =====================================

  const handleInputChange = (
    feature,
    value
  ) => {

    setPredictionInputs((prev) => ({

      ...prev,

      [feature]:
        Number(value),

    }));
  };

  // =====================================
  // MOCK PREDICTION
  // =====================================

  const handlePrediction = () => {

    const probability =
      Math.floor(
        Math.random() * 30
      ) + 70;

    setPredictionResult({

      prediction:
        probability > 80
          ? "Aprobado"
          : "En riesgo",

      probability,
    });
  };

  return (

    <Layout>

      <div className="lab-page">

        {/* HEADER */}

        <header className="lab-header">

          <div>

            <span className="lab-badge">
              Laboratorio IA
            </span>

            <h1>
              Machine Learning Educativo
            </h1>

            <p>
              Genera datasets dinámicos,
              entrena modelos reales
              y realiza predicciones
              académicas.
            </p>

          </div>

        </header>

        {/* ERROR */}

        {
          error && (

            <div className="lab-card">

              <p
                style={{
                  color: "#fca5a5",
                }}
              >
                {error}
              </p>

            </div>
          )
        }

        {/* MAIN */}

        <main className="lab-layout">

          {/* =====================================
              DATASET CONFIG
          ====================================== */}

          <section className="lab-card">

            <div className="card-title">

              <span>01</span>

              <div>

                <h2>
                  Generar dataset
                </h2>

                <p>
                  Crea datasets
                  educativos dinámicos.
                </p>

              </div>

            </div>

            <div className="form-grid">

              <div className="lab-form-group">

                <label>
                  Tipo de dataset
                </label>

                <select
                  value={datasetType}
                  onChange={(e) =>

                    setDatasetType(
                      e.target.value
                    )
                  }
                >

                  <option value="rendimiento">
                    Rendimiento académico
                  </option>

                  <option value="riesgo">
                    Riesgo académico
                  </option>

                  <option value="programacion">
                    Programación
                  </option>

                  <option value="asistencia">
                    Asistencia
                  </option>

                  <option value="becas">
                    Becas
                  </option>

                  <option value="desercion">
                    Deserción
                  </option>

                  <option value="matematica">
                    Matemática
                  </option>

                  <option value="ia">
                    Inteligencia Artificial
                  </option>

                  <option value="redes">
                    Redes
                  </option>

                  <option value="algoritmos">
                    Algoritmos
                  </option>

                </select>

              </div>

              <div className="lab-form-group">

                <label>
                  Cantidad de registros
                </label>

                <select
                  value={datasetSize}
                  onChange={(e) =>

                    setDatasetSize(
                      Number(
                        e.target.value
                      )
                    )
                  }
                >

                  <option value={100}>
                    100 registros
                  </option>

                  <option value={300}>
                    300 registros
                  </option>

                  <option value={500}>
                    500 registros
                  </option>

                  <option value={1000}>
                    1000 registros
                  </option>

                </select>

              </div>

            </div>

            <button
              className="train-btn"
              onClick={
                handleGenerateDataset
              }
              disabled={
                loadingDataset
              }
            >

              {
                loadingDataset

                  ? "Generando..."

                  : "Generar Dataset"
              }

            </button>

          </section>

          {/* =====================================
              DATASET INFO
          ====================================== */}

          {
            datasetInfo && (

              <section className="lab-card">

                <div className="card-title">

                  <span>02</span>

                  <div>

                    <h2>
                      Dataset generado
                    </h2>

                    <p>
                      Información y visualización
                      del dataset generado.
                    </p>

                  </div>

                </div>

                {/* METRICS */}

                <div className="metrics-grid">

                  <div className="metric">

                    <strong>
                      {
                        datasetInfo.total_records
                      }
                    </strong>

                    <span>
                      Registros
                    </span>

                  </div>

                  <div className="metric">

                    <strong>
                      {
                        datasetInfo.total_features
                      }
                    </strong>

                    <span>
                      Variables
                    </span>

                  </div>

                  <div className="metric">

                    <strong>
                      {
                        Object.keys(
                          datasetInfo.target_classes || {}
                        ).length
                      }
                    </strong>

                    <span>
                      Clases
                    </span>

                  </div>

                </div>

                {/* TABLE */}

                <div
                  className="table-wrapper"
                  style={{
                    marginTop: "24px",
                  }}
                >

                  <table>

                    <thead>

                      <tr>

                        {
                          datasetInfo.preview?.[0] &&

                          Object.keys(
                            datasetInfo.preview[0]
                          ).map((col) => (

                            <th key={col}>
                              {col}
                            </th>
                          ))
                        }

                      </tr>

                    </thead>

                    <tbody>

                      {
                        datasetInfo.preview?.map(
                          (
                            row,
                            index
                          ) => (

                            <tr key={index}>

                              {
                                Object.values(
                                  row
                                ).map(
                                  (
                                    value,
                                    i
                                  ) => (

                                    <td key={i}>
                                      {String(value)}
                                    </td>
                                  )
                                )
                              }

                            </tr>
                          )
                        )
                      }

                    </tbody>

                  </table>

                </div>

              </section>
            )
          }

          {/* =====================================
              MODEL TRAINING
          ====================================== */}

          {
            datasetInfo && (

              <section className="lab-card">

                <div className="card-title">

                  <span>03</span>

                  <div>

                    <h2>
                      Entrenar modelo
                    </h2>

                    <p>
                      Ejecuta Machine Learning.
                    </p>

                  </div>

                </div>

                <div className="lab-form-group">

                  <label>
                    Modelo IA
                  </label>

                  <select
                    value={selectedModel}
                    onChange={(e) =>

                      setSelectedModel(
                        e.target.value
                      )
                    }
                  >

                    {
                      models.map((model) => (

                        <option
                          key={model.name}
                          value={model.name}
                        >
                          {model.name}
                        </option>
                      ))
                    }

                  </select>

                </div>

                <button
                  className="train-btn"
                  onClick={handleTrain}
                  disabled={loadingTrain}
                >

                  {
                    loadingTrain

                      ? "Entrenando..."

                      : "Entrenar Modelo"
                  }

                </button>

              </section>
            )
          }

          {/* =====================================
              RESULTS
          ====================================== */}

          {
            trainingResult && (

              <section className="lab-card">

                <div className="card-title">

                  <span>04</span>

                  <div>

                    <h2>
                      Resultados IA
                    </h2>

                    <p>
                      Métricas del modelo.
                    </p>

                  </div>

                </div>

                <div className="metrics-grid">

                  <div className="metric">

                    <strong>

                      {
                        (
                          trainingResult.accuracy * 100
                        ).toFixed(2)
                      }%

                    </strong>

                    <span>
                      Accuracy
                    </span>

                  </div>

                  <div className="metric">

                    <strong>

                      {
                        (
                          trainingResult.precision * 100
                        ).toFixed(2)
                      }%

                    </strong>

                    <span>
                      Precision
                    </span>

                  </div>

                  <div className="metric">

                    <strong>

                      {
                        (
                          trainingResult.recall * 100
                        ).toFixed(2)
                      }%

                    </strong>

                    <span>
                      Recall
                    </span>

                  </div>

                </div>

              </section>
            )
          }

          {/* =====================================
              PREDICTION
          ====================================== */}

          {
            trainingResult && (

              <section className="lab-card">

                <div className="card-title">

                  <span>05</span>

                  <div>

                    <h2>
                      Predicción IA
                    </h2>

                    <p>
                      Simula predicciones.
                    </p>

                  </div>

                </div>

                <div className="form-grid">

                  {
                    trainingResult.feature_names?.map(
                      (feature) => (

                        <div
                          key={feature}
                          className="lab-form-group"
                        >

                          <label>
                            {feature}
                          </label>

                          <input
                            type="number"
                            placeholder={feature}
                            onChange={(e) =>

                              handleInputChange(
                                feature,
                                e.target.value
                              )
                            }
                          />

                        </div>
                      )
                    )
                  }

                </div>

                <button
                  className="train-btn"
                  onClick={handlePrediction}
                >

                  Realizar Predicción

                </button>

                {
                  predictionResult && (

                    <div
                      className="result-message"
                      style={{
                        marginTop: "20px",
                      }}
                    >

                      <span>🤖</span>

                      <p>

                        Resultado:

                        <strong>

                          {" "}
                          {
                            predictionResult.prediction
                          }

                        </strong>

                        {" "}(
                        {
                          predictionResult.probability
                        }%)

                      </p>

                    </div>
                  )
                }

              </section>
            )
          }

        </main>

      </div>

    </Layout>
  );
}

export default LaboratorioEstudiante;