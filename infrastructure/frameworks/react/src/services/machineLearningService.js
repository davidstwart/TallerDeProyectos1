import API from "../api/axios";

export const trainModelRequest = async (
  data
) => {

  const response = await API.post(
    "/machine-learning/train",
    data
  );

  return response.data;
};

export const predictRequest = async (
  data
) => {

  const response = await API.post(
    "/machine-learning/predict",
    data
  );

  return response.data;
};

export const getModelsRequest = async () => {

  const response = await API.get(
    "/machine-learning/models"
  );

  return response.data;
};

export const getPredictionsRequest = async () => {

  const response = await API.get(
    "/machine-learning/predictions"
  );

  return response.data;
};