import API from "../api/axios";

export const uploadDatasetRequest = async (
  file
) => {

  const formData = new FormData();

  formData.append("file", file);

  const response = await API.post(
    "/datasets/upload",
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

export const getDatasetsRequest = async () => {

  const response = await API.get(
    "/datasets"
  );

  return response.data;
};

export const getDatasetByIdRequest = async (
  datasetId
) => {

  const response = await API.get(
    `/datasets/${datasetId}`
  );

  return response.data;
};

export const deleteDatasetRequest = async (
  datasetId
) => {

  const response = await API.delete(
    `/datasets/${datasetId}`
  );

  return response.data;
};