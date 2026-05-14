import Layout from "../components/layout/Layout";
import {
  useEffect,
  useState
} from "react";

import "../estilos/pages.css";

import {
  getModelsRequest,
  uploadCSVRequest,
  getDatasetInfoRequest,
  trainModelRequest,
} from "../services/iaLabService";

function Laboratorio() {

  // =========================================
  // STATES
  // =========================================

  const [file, setFile] =
    useState(null);

  const [sessionId, setSessionId] =
    useState("");

  const [datasetInfo, setDatasetInfo] =
    useState(null);

  const [models, setModels] =
    useState([]);

  const [selectedModel, setSelectedModel] =
    useState("");

  const [targetColumn, setTargetColumn] =
    useState("aprobado");

  const [trainingResult, setTrainingResult] =
    useState(null);

  const [predictionInputs, setPredictionInputs] =
  useState({});

  const [predictionResult, setPredictionResult] =
    useState(null);

  const [loadingPrediction, setLoadingPrediction] =
    useState(false);

  const [loadingUpload, setLoadingUpload] =
    useState(false);

  const [loadingTrain, setLoadingTrain] =
    useState(false);

  const [error, setError] =
    useState("");

  // =========================================
  // LOAD MODELS
  // =========================================

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

  // =========================================
  // FILE CHANGE
  // =========================================

  const handleFileChange = (
    e
  ) => {

    const selectedFile =
      e.target.files[0];

    if (!selectedFile) return;

    setFile(selectedFile);
  };

  // =========================================
  // UPLOAD DATASET
  // =========================================

  const handleUpload =
    async () => {

      if (!file) {

        setError(
          "Selecciona un archivo CSV"
        );

        return;
      }

      try {

        setLoadingUpload(true);

        setError("");

        const response =
          await uploadCSVRequest(
            file,
            targetColumn
          );

        setSessionId(
          response.session_id
        );

        const datasetResponse =
          await getDatasetInfoRequest(
            response.session_id,
            targetColumn
          );

        setDatasetInfo(
          datasetResponse
        );

      } catch (error) {

        console.error(error);

        setError(
          error.response?.data?.detail ||
          "Error subiendo dataset"
        );

      } finally {

        setLoadingUpload(false);
      }
    };

  // =========================================
  // TRAIN MODEL
  // =========================================

  const handleTrain =
    async () => {

      if (!sessionId) {

        setError(
          "Debes subir un dataset"
        );

        return;
      }

      try {

        setLoadingTrain(true);

        setError("");

        const response =
          await trainModelRequest({
            session_id: sessionId,
            model_name: selectedModel,
            params: {},
            test_size: 0.2,
          });

        setTrainingResult(
          response
        );

      } catch (error) {

        console.error(error);

        setError(
          error.response?.data?.detail ||
          "Error entrenando modelo"
        );

      } finally {

        setLoadingTrain(false);
      }
    };


  // =========================================
  // HANDLE PREDICTION INPUT
  // =========================================

  const handlePredictionInput = (
    feature,
    value
  ) => {

    setPredictionInputs((prev) => ({

      ...prev,

      [feature]: Number(value),

    }));
  };

  // =========================================
  // PREDICT
  // =========================================

  const handlePredict =
    async () => {

      try {

        setLoadingPrediction(true);

        const response =
          // eslint-disable-next-line no-undef
          await predictRequest({

            session_id: sessionId,

            features:
              predictionInputs,

          });

        setPredictionResult(
          response
        );

      } catch (error) {

        console.error(error);

        setError(
          error.response?.data?.detail ||
          "Error realizando predicción"
        );

      } finally {

        setLoadingPrediction(false);
      }
    };


    

  return (

  <Layout>

    <div className="lab-page">

      {/* =====================================
          HEADER
      ====================================== */}

      <header className="lab-header">

        <div>

          <span className="lab-badge">
            Laboratorio IA
          </span>

          <h1>
            Entrena modelos reales
            de Machine Learning
          </h1>

          <p>
            Sube datasets CSV,
            analiza estadísticas,
            entrena modelos sklearn
            y realiza predicciones
            académicas reales.
          </p>

        </div>

      </header>

      {/* =====================================
          ERROR
      ====================================== */}

      {
        error && (

          <div
            className="lab-card"
            style={{
              marginBottom: "24px",
              border:
                "1px solid rgba(239,68,68,.4)",
            }}
          >

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

      {/* =====================================
          MAIN
      ====================================== */}

      <main className="lab-layout">

        {/* =====================================
            UPLOAD
        ====================================== */}

        <section className="lab-card">

          <div className="card-title">

            <span>01</span>

            <div>

              <h2>
                Subir dataset
              </h2>

              <p>
                Carga datasets CSV
                educativos para entrenar
                modelos predictivos.
              </p>

            </div>

          </div>

          <label className="upload-box">

            <input
              type="file"
              accept=".csv"
              onChange={
                handleFileChange
              }
            />

            <div className="upload-icon">
              📁
            </div>

            <h3>

              {
                file
                  ? file.name
                  : "Selecciona tu dataset"
              }

            </h3>

            <p>
              Formato soportado:
              CSV
            </p>

          </label>

          <button
            className="train-btn"
            onClick={
              handleUpload
            }
            disabled={
              loadingUpload
            }
            style={{
              marginTop: "18px",
            }}
          >

            {
              loadingUpload
                ? "Subiendo..."
                : "Subir Dataset"
            }

          </button>

        </section>

        {/* =====================================
            CONFIG
        ====================================== */}

        <section className="lab-card config-card">

          <div className="card-title">

            <span>02</span>

            <div>

              <h2>
                Configurar entrenamiento
              </h2>

              <p>
                Selecciona el modelo
                de IA y configura
                el entrenamiento.
              </p>

            </div>

          </div>

          <div className="form-grid">

            {
              datasetInfo && (

                <div className="lab-form-group">

                  <label>
                    Variable objetivo
                  </label>

                  <select
                    value={
                      targetColumn
                    }
                    onChange={(e) =>
                      setTargetColumn(
                        e.target.value
                      )
                    }
                  >

                    {
                      datasetInfo.feature_names
                        ?.map((col) => (

                          <option
                            key={col}
                            value={col}
                          >
                            {col}
                          </option>
                        ))
                    }

                  </select>

                </div>
              )
            }

            <div className="lab-form-group">

              <label>
                Modelo IA
              </label>

              <select
                value={
                  selectedModel
                }
                onChange={(e) =>
                  setSelectedModel(
                    e.target.value
                  )
                }
              >

                {
                  models.map(
                    (model) => (

                      <option
                        key={
                          model.name
                        }
                        value={
                          model.name
                        }
                      >
                        {model.name}
                      </option>
                    )
                  )
                }

              </select>

            </div>

          </div>

          <button
            className="train-btn"
            onClick={
              handleTrain
            }
            disabled={
              loadingTrain
            }
          >

            {
              loadingTrain
                ? "Entrenando..."
                : "Entrenar Modelo"
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

                <span>03</span>

                <div>

                  <h2>
                    Información del dataset
                  </h2>

                  <p>
                    Vista previa y
                    análisis básico.
                  </p>

                </div>

              </div>

              <div
                className="metrics-grid"
                style={{
                  marginBottom: "20px",
                }}
              >

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

              <div className="table-wrapper">

                <table>

                  <thead>

                    <tr>

                      {
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
                      datasetInfo.preview.map(
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
                                    {
                                      value
                                    }
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
            RESULTS
        ====================================== */}

        {
          trainingResult && (

            <section className="lab-card">

              <div className="card-title">

                <span>04</span>

                <div>

                  <h2>
                    Resultados del modelo
                  </h2>

                  <p>
                    Métricas generadas
                    por sklearn.
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

              <div
                className="metrics-grid"
                style={{
                  marginTop: "16px",
                }}
              >

                <div className="metric">

                  <strong>

                    {
                      (
                        trainingResult.f1_score * 100
                      ).toFixed(2)
                    }%

                  </strong>

                  <span>
                    F1 Score
                  </span>

                </div>

                <div className="metric">

                  <strong>
                    {
                      trainingResult.train_samples
                    }
                  </strong>

                  <span>
                    Train Samples
                  </span>

                </div>

                <div className="metric">

                  <strong>
                    {
                      trainingResult.test_samples
                    }
                  </strong>

                  <span>
                    Test Samples
                  </span>

                </div>

              </div>

              <div className="result-message">

                <span>✅</span>

                <p>

                  Modelo entrenado
                  correctamente usando:

                  <strong>

                    {" "}
                    {
                      trainingResult.model_name
                    }

                  </strong>

                </p>

              </div>

            </section>
          )
        }

        {
          trainingResult && (

            <section className="lab-card">

              <div className="card-title">

                <span>05</span>

                <div>

                  <h2>
                    Realizar predicción
                  </h2>

                  <p>
                    Ingresa valores para
                    obtener predicciones
                    académicas usando IA.
                  </p>

                </div>

              </div>

              <div className="form-grid">

                {
                  trainingResult.feature_names.map(
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
                          step="any"
                          placeholder={`Valor de ${feature}`}
                          onChange={(e) =>

                            handlePredictionInput(
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
                onClick={handlePredict}
                disabled={loadingPrediction}
              >

                {
                  loadingPrediction
                    ? "Prediciendo..."
                    : "Realizar Predicción"
                }

              </button>

              {
                predictionResult && (

                  <div
                    className="result-message"
                    style={{
                      marginTop: "24px",
                    }}
                  >

                    <span>🤖</span>

                    <div>

                      <p>

                        Predicción:

                        <strong>

                          {" "}
                          {
                            predictionResult.prediction
                          }

                        </strong>

                      </p>

                      {
                        predictionResult.probabilities && (

                          <div
                            style={{
                              marginTop: "12px",
                            }}
                          >

                            {
                              Object.entries(

                                predictionResult.probabilities

                              ).map(

                                ([key, value]) => (

                                  <p key={key}>

                                    Clase {key}:

                                    {" "}

                                    <strong>

                                      {
                                        (
                                          value * 100
                                        ).toFixed(2)
                                      }%

                                    </strong>

                                  </p>
                                )
                              )
                            }

                          </div>
                        )
                      }

                    </div>

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

export default Laboratorio;