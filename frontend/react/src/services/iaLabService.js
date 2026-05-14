import API from "../api/axios";

// ======================================
// MODELOS DISPONIBLES
// ======================================

export const getModelsRequest =
  async () => {

    const response =
      await API.get(
        "/api/v1/ia-lab/models"
      );

    return response.data;
  };

// ======================================
// SUBIR CSV
// ======================================

export const uploadCSVRequest =
  async (
    file,
    targetColumn = "aprobado"
  ) => {

    const formData =
      new FormData();

    formData.append(
      "file",
      file
    );

    const response =
      await API.post(

        `/api/v1/ia-lab/dataset/upload?target_column=${targetColumn}`,

        formData,

        {
          headers: {
            "Content-Type":
              "multipart/form-data",
          },
        }
      );

    return response.data;
  };

// ======================================
// INFO DATASET
// ======================================

export const getDatasetInfoRequest =
  async (
    sessionId,
    targetColumn = "aprobado"
  ) => {

    const response =
      await API.get(

        `/api/v1/ia-lab/dataset/${sessionId}/info?target_column=${targetColumn}`

      );

    return response.data;
  };

// ======================================
// TRAIN MODEL
// ======================================

export const trainModelRequest =
  async (data) => {

    const response =
      await API.post(
        "/api/v1/ia-lab/train",
        data
      );

    return response.data;
  };

// ======================================
// PREDICT
// ======================================

export const predictRequest =
  async (data) => {

    const response =
      await API.post(
        "/api/v1/ia-lab/predict",
        data
      );

    return response.data;
  };


// ======================================
// GENERAR DATASET SIMULADO
// ======================================

export const generateDatasetRequest =
  async (
    nSamples,
    datasetType
  ) => {

    const response =
      await API.post(

        "/api/v1/ia-lab/dataset/generate",

        {
          n_samples: nSamples,
          dataset_type: datasetType,
        }
      );

    return response.data;
  };