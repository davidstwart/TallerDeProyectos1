import API from "../api/axios";

// ======================================
// GET TEMAS
// ======================================

export const getTemasRequest =
  async () => {

    const response =
      await API.get(
        "/temas"
      );

    return response.data;
  };

// ======================================
// CREATE TEMA
// ======================================

export const createTemaRequest =
  async (data) => {

    const response =
      await API.post(
        "/temas",
        data
      );

    return response.data;
  };