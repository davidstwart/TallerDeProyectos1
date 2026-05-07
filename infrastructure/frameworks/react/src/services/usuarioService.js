import API from "../api/axios";

export const getUsuariosRequest = async () => {

  const response = await API.get(
    "/usuarios"
  );

  return response.data;
};

export const createUsuarioRequest = async (
  data
) => {

  const response = await API.post(
    "/usuarios",
    data
  );

  return response.data;
};

export const updateUsuarioRequest = async (
  usuarioId,
  data
) => {

  const response = await API.put(
    `/usuarios/${usuarioId}`,
    data
  );

  return response.data;
};

export const deleteUsuarioRequest = async (
  usuarioId
) => {

  const response = await API.delete(
    `/usuarios/${usuarioId}`
  );

  return response.data;
};

export const getDocentesRequest = async () => {

  const response = await API.get(
    "/usuarios/docentes"
  );

  return response.data;
};

export const getEstudiantesRequest = async () => {

  const response = await API.get(
    "/usuarios/estudiantes"
  );

  return response.data;
};